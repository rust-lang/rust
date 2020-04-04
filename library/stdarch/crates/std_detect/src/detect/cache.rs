//! Caches run-time feature detection so that it only needs to be computed
//! once.

#![allow(dead_code)] // not used on all platforms

use crate::sync::atomic::Ordering;

use crate::sync::atomic::AtomicUsize;

/// Sets the `bit` of `x`.
#[inline]
const fn set_bit(x: u64, bit: u32) -> u64 {
    x | 1 << bit
}

/// Tests the `bit` of `x`.
#[inline]
const fn test_bit(x: u64, bit: u32) -> bool {
    x & (1 << bit) != 0
}

/// Unset the `bit of `x`.
#[inline]
const fn unset_bit(x: u64, bit: u32) -> u64 {
    x & !(1 << bit)
}

/// Maximum number of features that can be cached.
const CACHE_CAPACITY: u32 = 62;

/// This type is used to initialize the cache
#[derive(Copy, Clone)]
pub(crate) struct Initializer(u64);

#[allow(clippy::use_self)]
impl Default for Initializer {
    fn default() -> Self {
        Initializer(0)
    }
}

// NOTE: the `debug_assert!` would catch that we do not add more Features than
// the one fitting our cache.
impl Initializer {
    /// Tests the `bit` of the cache.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn test(self, bit: u32) -> bool {
        debug_assert!(
            bit < CACHE_CAPACITY,
            "too many features, time to increase the cache size!"
        );
        test_bit(self.0, bit)
    }

    /// Sets the `bit` of the cache.
    #[inline]
    pub(crate) fn set(&mut self, bit: u32) {
        debug_assert!(
            bit < CACHE_CAPACITY,
            "too many features, time to increase the cache size!"
        );
        let v = self.0;
        self.0 = set_bit(v, bit);
    }

    /// Unsets the `bit` of the cache.
    #[inline]
    pub(crate) fn unset(&mut self, bit: u32) {
        debug_assert!(
            bit < CACHE_CAPACITY,
            "too many features, time to increase the cache size!"
        );
        let v = self.0;
        self.0 = unset_bit(v, bit);
    }
}

/// This global variable is a cache of the features supported by the CPU.
// Note: on x64, we only use the first slot
static CACHE: [Cache; 2] = [Cache::uninitialized(), Cache::uninitialized()];

/// Feature cache with capacity for `usize::MAX - 1` features.
///
/// Note: the last feature bit is used to represent an
/// uninitialized cache.
///
/// Note: we can use `Relaxed` atomic operations, because we are only interested
/// in the effects of operations on a single memory location. That is, we only
/// need "modification order", and not the full-blown "happens before". However,
/// we use `SeqCst` just to be on the safe side.
struct Cache(AtomicUsize);

impl Cache {
    const CAPACITY: u32 = (core::mem::size_of::<usize>() * 8 - 1) as u32;
    const MASK: usize = (1 << Cache::CAPACITY) - 1;

    /// Creates an uninitialized cache.
    #[allow(clippy::declare_interior_mutable_const)]
    const fn uninitialized() -> Self {
        Cache(AtomicUsize::new(usize::MAX))
    }
    /// Is the cache uninitialized?
    #[inline]
    pub(crate) fn is_uninitialized(&self) -> bool {
        self.0.load(Ordering::SeqCst) == usize::MAX
    }

    /// Is the `bit` in the cache set?
    #[inline]
    pub(crate) fn test(&self, bit: u32) -> bool {
        test_bit(self.0.load(Ordering::SeqCst) as u64, bit)
    }

    /// Initializes the cache.
    #[inline]
    fn initialize(&self, value: usize) {
        self.0.store(value, Ordering::SeqCst);
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "std_detect_env_override")] {
        #[inline(never)]
        fn initialize(mut value: Initializer) {
            if let Ok(disable) = crate::env::var("RUST_STD_DETECT_UNSTABLE") {
                for v in disable.split(" ") {
                    let _ = super::Feature::from_str(v).map(|v| value.unset(v as u32));
                }
            }
            do_initialize(value);
        }
    } else {
        #[inline]
        fn initialize(value: Initializer) {
            do_initialize(value);
        }
    }
}

#[inline]
fn do_initialize(value: Initializer) {
    CACHE[0].initialize((value.0) as usize & Cache::MASK);
    CACHE[1].initialize((value.0 >> Cache::CAPACITY) as usize & Cache::MASK);
}

/// Tests the `bit` of the storage. If the storage has not been initialized,
/// initializes it with the result of `f()`.
///
/// On its first invocation, it detects the CPU features and caches them in the
/// `CACHE` global variable as an `AtomicU64`.
///
/// It uses the `Feature` variant to index into this variable as a bitset. If
/// the bit is set, the feature is enabled, and otherwise it is disabled.
///
/// If the feature `std_detect_env_override` is enabled looks for the env
/// variable `RUST_STD_DETECT_UNSTABLE` and uses its its content to disable
/// Features that would had been otherwise detected.
#[inline]
pub(crate) fn test<F>(bit: u32, f: F) -> bool
where
    F: FnOnce() -> Initializer,
{
    let (bit, idx) = if bit < Cache::CAPACITY {
        (bit, 0)
    } else {
        (bit - Cache::CAPACITY, 1)
    };

    if CACHE[idx].is_uninitialized() {
        initialize(f())
    }
    CACHE[idx].test(bit)
}
