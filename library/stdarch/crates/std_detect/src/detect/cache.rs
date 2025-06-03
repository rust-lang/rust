//! Caches run-time feature detection so that it only needs to be computed
//! once.

#![allow(dead_code)] // not used on all platforms

use core::sync::atomic::Ordering;

use core::sync::atomic::AtomicUsize;

/// Sets the `bit` of `x`.
#[inline]
const fn set_bit(x: u128, bit: u32) -> u128 {
    x | 1 << bit
}

/// Tests the `bit` of `x`.
#[inline]
const fn test_bit(x: u128, bit: u32) -> bool {
    x & (1 << bit) != 0
}

/// Unset the `bit of `x`.
#[inline]
const fn unset_bit(x: u128, bit: u32) -> u128 {
    x & !(1 << bit)
}

/// Maximum number of features that can be cached.
const CACHE_CAPACITY: u32 = 93;

/// This type is used to initialize the cache
// The derived `Default` implementation will initialize the field to zero,
// which is what we want.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub(crate) struct Initializer(u128);

// NOTE: the `debug_assert!` would catch that we do not add more Features than
// the one fitting our cache.
impl Initializer {
    /// Tests the `bit` of the cache.
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
// Note: the third slot is only used in x86
// Another Slot can be added if needed without any change to `Initializer`
static CACHE: [Cache; 3] = [
    Cache::uninitialized(),
    Cache::uninitialized(),
    Cache::uninitialized(),
];

/// Feature cache with capacity for `size_of::<usize>() * 8 - 1` features.
///
/// Note: 0 is used to represent an uninitialized cache, and (at least) the most
/// significant bit is set on any cache which has been initialized.
///
/// Note: we use `Relaxed` atomic operations, because we are only interested in
/// the effects of operations on a single memory location. That is, we only need
/// "modification order", and not the full-blown "happens before".
struct Cache(AtomicUsize);

impl Cache {
    const CAPACITY: u32 = (core::mem::size_of::<usize>() * 8 - 1) as u32;
    const MASK: usize = (1 << Cache::CAPACITY) - 1;
    const INITIALIZED_BIT: usize = 1usize << Cache::CAPACITY;

    /// Creates an uninitialized cache.
    #[allow(clippy::declare_interior_mutable_const)]
    const fn uninitialized() -> Self {
        Cache(AtomicUsize::new(0))
    }

    /// Is the `bit` in the cache set? Returns `None` if the cache has not been initialized.
    #[inline]
    pub(crate) fn test(&self, bit: u32) -> Option<bool> {
        let cached = self.0.load(Ordering::Relaxed);
        if cached == 0 {
            None
        } else {
            Some(test_bit(cached as u128, bit))
        }
    }

    /// Initializes the cache.
    #[inline]
    fn initialize(&self, value: usize) -> usize {
        debug_assert_eq!((value & !Cache::MASK), 0);
        self.0
            .store(value | Cache::INITIALIZED_BIT, Ordering::Relaxed);
        value
    }
}

#[inline]
fn initialize(value: Initializer) -> Initializer {
    CACHE[0].initialize((value.0) as usize & Cache::MASK);
    CACHE[1].initialize((value.0 >> Cache::CAPACITY) as usize & Cache::MASK);
    CACHE[2].initialize((value.0 >> (2 * Cache::CAPACITY)) as usize & Cache::MASK);
    value
}

// We only have to detect features once, and it's fairly costly, so hint to LLVM
// that it should assume that cache hits are more common than misses (which is
// the point of caching). It's possibly unfortunate that this function needs to
// reach across modules like this to call `os::detect_features`, but it produces
// the best code out of several attempted variants.
//
// The `Initializer` that the cache was initialized with is returned, so that
// the caller can call `test()` on it without having to load the value from the
// cache again.
#[cold]
fn detect_and_initialize() -> Initializer {
    initialize(super::os::detect_features())
}

/// Tests the `bit` of the storage. If the storage has not been initialized,
/// initializes it with the result of `os::detect_features()`.
///
/// On its first invocation, it detects the CPU features and caches them in the
/// `CACHE` global variable as an `AtomicU64`.
///
/// It uses the `Feature` variant to index into this variable as a bitset. If
/// the bit is set, the feature is enabled, and otherwise it is disabled.
#[inline]
pub(crate) fn test(bit: u32) -> bool {
    let (relative_bit, idx) = if bit < Cache::CAPACITY {
        (bit, 0)
    } else if bit < 2 * Cache::CAPACITY {
        (bit - Cache::CAPACITY, 1)
    } else {
        (bit - 2 * Cache::CAPACITY, 2)
    };
    CACHE[idx]
        .test(relative_bit)
        .unwrap_or_else(|| detect_and_initialize().test(bit))
}
