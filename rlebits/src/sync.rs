/// More scalable alternatives to global spin mutex for RleBits
///
/// This demonstrates several approaches for better scalability

use crate::{RleBits, RleBitsError};
use spin::{RwLock, Mutex};

/// Original thread-safe wrapper using a simple mutex
/// 
/// This is the simplest approach but can become a bottleneck under high contention.
/// All operations (reads and writes) serialize through a single mutex.
/// 
/// **Performance characteristics:**
/// - Read scalability: Poor (all reads serialize)
/// - Write scalability: Poor (all writes serialize)  
/// - Complexity: Simple
/// - Best for: Low contention scenarios
#[cfg(feature = "thread-safe")]
#[derive(Debug)]
pub struct ThreadSafeRleBits {
    inner: Mutex<RleBits>,
}

#[cfg(feature = "thread-safe")]
impl ThreadSafeRleBits {
    pub const fn new(limit: usize) -> Self {
        Self {
            inner: Mutex::new(RleBits::new(limit)),
        }
    }

    pub fn get(&self, index: usize) -> Option<u8> {
        let guard = self.inner.lock();
        guard.get(index)
    }

    pub fn get_run(&self, n: usize) -> usize {
        let guard = self.inner.lock();
        guard.get_run(n)
    }

    pub fn set(&self, index: usize, value: u8) -> Result<(), RleBitsError> {
        let mut guard = self.inner.lock();
        guard.set(index, value)
    }

    pub fn set_range(&self, index: usize, len: usize, value: u8) -> Result<(), RleBitsError> {
        let mut guard = self.inner.lock();
        guard.set_range(index, len, value)
    }

    pub fn sanity_check(&self) -> usize {
        let guard = self.inner.lock();
        guard.sanity_check()
    }

    #[cfg(all(test, feature = "std"))]
    pub fn dump(&self) {
        let guard = self.inner.lock();
        guard.dump();
    }

    /// For advanced use cases where you need to perform multiple operations atomically
    pub fn with_lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut RleBits) -> R,
    {
        let mut guard = self.inner.lock();
        f(&mut *guard)
    }
}

// ThreadSafeRleBits is Send and Sync because spin::Mutex is Send and Sync
#[cfg(feature = "thread-safe")]
unsafe impl Send for ThreadSafeRleBits {}
#[cfg(feature = "thread-safe")]
unsafe impl Sync for ThreadSafeRleBits {}

/// Read-Write Lock approach
/// Allows multiple concurrent readers, exclusive writers
#[cfg(feature = "thread-safe")]
#[derive(Debug)]
pub struct RwLockRleBits {
    inner: RwLock<RleBits>,
}

#[cfg(feature = "thread-safe")]
impl RwLockRleBits {
    pub const fn new(limit: usize) -> Self {
        Self {
            inner: RwLock::new(RleBits::new(limit)),
        }
    }

    /// Multiple threads can read concurrently
    pub fn get(&self, index: usize) -> Option<u8> {
        let guard = self.inner.read();
        guard.get(index)
    }

    pub fn get_run(&self, n: usize) -> usize {
        let guard = self.inner.read();
        guard.get_run(n)
    }

    pub fn sanity_check(&self) -> usize {
        let guard = self.inner.read();
        guard.sanity_check()
    }

    /// Writers get exclusive access
    pub fn set(&self, index: usize, value: u8) -> Result<(), RleBitsError> {
        let mut guard = self.inner.write();
        guard.set(index, value)
    }

    pub fn set_range(&self, index: usize, len: usize, value: u8) -> Result<(), RleBitsError> {
        let mut guard = self.inner.write();
        guard.set_range(index, len, value)
    }
}

/// Alternative 2: Sharded/Segmented approach
/// Divide the bit array into segments, each with its own mutex
/// This reduces contention when operations target different segments
#[cfg(feature = "thread-safe")]
#[derive(Debug)]
pub struct ShardedRleBits {
    shards: [Mutex<RleBits>; 8], // 8 shards for example
    shard_size: usize,
}

#[cfg(feature = "thread-safe")]
impl ShardedRleBits {
    pub fn new(limit: usize) -> Self {
        let shard_size = (limit + 7) / 8; // Round up division
        Self {
            shards: [
                Mutex::new(RleBits::new(shard_size)),
                Mutex::new(RleBits::new(shard_size)),
                Mutex::new(RleBits::new(shard_size)),
                Mutex::new(RleBits::new(shard_size)),
                Mutex::new(RleBits::new(shard_size)),
                Mutex::new(RleBits::new(shard_size)),
                Mutex::new(RleBits::new(shard_size)),
                Mutex::new(RleBits::new(shard_size)),
            ],
            shard_size,
        }
    }

    fn get_shard_and_index(&self, index: usize) -> (usize, usize) {
        let shard_id = index / self.shard_size;
        let local_index = index % self.shard_size;
        (shard_id.min(7), local_index) // Clamp to last shard
    }

    pub fn get(&self, index: usize) -> Option<u8> {
        let (shard_id, local_index) = self.get_shard_and_index(index);
        let guard = self.shards[shard_id].lock();
        guard.get(local_index)
    }

    pub fn set(&self, index: usize, value: u8) -> Result<(), RleBitsError> {
        let (shard_id, local_index) = self.get_shard_and_index(index);
        let mut guard = self.shards[shard_id].lock();
        guard.set(local_index, value)
    }

    // Note: set_range becomes more complex as it may span multiple shards
    pub fn set_range(&self, index: usize, len: usize, value: u8) -> Result<(), RleBitsError> {
        let end = index + len;
        let start_shard = index / self.shard_size;
        let end_shard = (end - 1) / self.shard_size;

        if start_shard == end_shard {
            // Single shard operation
            let (shard_id, local_index) = self.get_shard_and_index(index);
            let mut guard = self.shards[shard_id].lock();
            guard.set_range(local_index, len, value)
        } else {
            // Multi-shard operation - need to lock multiple shards
            // This requires careful ordering to avoid deadlocks
            for shard_id in start_shard..=end_shard.min(7) {
                let range_start = if shard_id == start_shard {
                    index % self.shard_size
                } else {
                    0
                };
                let range_end = if shard_id == end_shard {
                    (end - 1) % self.shard_size + 1
                } else {
                    self.shard_size
                };
                let range_len = range_end - range_start;

                let mut guard = self.shards[shard_id].lock();
                guard.set_range(range_start, range_len, value)?;
            }
            Ok(())
        }
    }
}

/// Alternative 3: Lock-free approach using atomic operations
/// This would require a completely different internal representation
/// using atomic primitives instead of a mutex-protected array
use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "thread-safe")]
#[derive(Debug)]
pub struct AtomicRleBits {
    // This is a simplified example - real implementation would be more complex
    // Could use a lock-free linked list or atomic arrays
    data: [AtomicUsize; 200], // Atomic version of runs array
    length: AtomicUsize,
}

#[cfg(feature = "thread-safe")]
impl AtomicRleBits {
    pub fn new(limit: usize) -> Self {
        // Initialize array element by element since AtomicUsize is not Copy
        const INIT: AtomicUsize = AtomicUsize::new(0);
        let mut data = [INIT; 200];
        data[0] = AtomicUsize::new(limit);

        Self {
            data,
            length: AtomicUsize::new(limit),
        }
    }

    pub fn get(&self, index: usize) -> Option<u8> {
        if index >= self.length.load(Ordering::Acquire) {
            return None;
        }

        // This is a simplified version - real implementation would need
        // more sophisticated lock-free algorithms to maintain consistency
        let mut start = 0;
        for (i, run) in self.data.iter().enumerate() {
            let len = run.load(Ordering::Acquire);
            if len == 0 && i > 0 {
                break;
            }
            if index < start + len {
                return Some((i % 2) as u8);
            }
            start += len;
        }
        None
    }

    // Note: Implementing atomic set operations requires complex 
    // compare-and-swap loops and is quite involved
}

/// Alternative 4: Thread-local storage with periodic synchronization
/// Each thread maintains local changes and periodically syncs with global state
/// This is only available in std environments

#[cfg(all(feature = "thread-safe", feature = "std"))]
pub mod thread_local_approach {
    use super::*;

    extern crate std;
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        static LOCAL_CHANGES: RefCell<HashMap<usize, u8>> = RefCell::new(HashMap::new());
    }

    #[derive(Debug)]
    pub struct EventuallyConsistentRleBits {
        global: RwLock<RleBits>,
    }

    impl EventuallyConsistentRleBits {
        pub fn new(limit: usize) -> Self {
            Self {
                global: RwLock::new(RleBits::new(limit)),
            }
        }

        /// Fast local write - no synchronization
        pub fn set_local(&self, index: usize, value: u8) {
            LOCAL_CHANGES.with(|changes| {
                changes.borrow_mut().insert(index, value);
            });
        }

        /// Read with local overrides
        pub fn get(&self, index: usize) -> Option<u8> {
            // Check local changes first
            let local_value = LOCAL_CHANGES.with(|changes| {
                changes.borrow().get(&index).copied()
            });

            if let Some(value) = local_value {
                Some(value)
            } else {
                // Fall back to global state
                let guard = self.global.read();
                guard.get(index)
            }
        }

        /// Synchronize local changes to global state
        pub fn sync(&self) -> Result<(), RleBitsError> {
            LOCAL_CHANGES.with(|changes| {
                let mut local = changes.borrow_mut();
                if !local.is_empty() {
                    let mut guard = self.global.write();
                    for (&index, &value) in local.iter() {
                        guard.set(index, value)?;
                    }
                    local.clear();
                }
                Ok(())
            })
        }
    }
}
