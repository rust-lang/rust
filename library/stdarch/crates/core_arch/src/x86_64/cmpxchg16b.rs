use crate::sync::atomic::Ordering;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Compares and exchange 16 bytes (128 bits) of data atomically.
///
/// This intrinsic corresponds to the `cmpxchg16b` instruction on `x86_64`
/// processors. It performs an atomic compare-and-swap, updating the `ptr`
/// memory location to `val` if the current value in memory equals `old`.
///
/// # Return value
///
/// This function returns the previous value at the memory location. If it is
/// equal to `old` then the memory was updated to `new`.
///
/// # Memory Orderings
///
/// This atomic operation has the same semantics of memory orderings as
/// `AtomicUsize::compare_exchange` does, only operating on 16 bytes of memory
/// instead of just a pointer.
///
/// The failure ordering must be [`Ordering::SeqCst`], [`Ordering::Acquire`] or
/// [`Ordering::Relaxed`].
///
/// For more information on memory orderings here see the `compare_exchange`
/// documentation for other `Atomic*` types in the standard library.
///
/// # Unsafety
///
/// This method is unsafe because it takes a raw pointer and will attempt to
/// read and possibly write the memory at the pointer. The pointer must also be
/// aligned on a 16-byte boundary.
///
/// This method also requires the `cmpxchg16b` CPU feature to be available at
/// runtime to work correctly. If the CPU running the binary does not actually
/// support `cmpxchg16b` and the program enters an execution path that
/// eventually would reach this function the behavior is undefined.
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
#[cfg_attr(test, assert_instr(cmpxchg16b, success = Ordering::SeqCst, failure = Ordering::SeqCst))]
#[target_feature(enable = "cmpxchg16b")]
#[stable(feature = "cmpxchg16b_intrinsic", since = "1.67.0")]
pub unsafe fn cmpxchg16b(
    dst: *mut u128,
    old: u128,
    new: u128,
    success: Ordering,
    failure: Ordering,
) -> u128 {
    debug_assert!(dst as usize % 16 == 0);

    let res = crate::sync::atomic::atomic_compare_exchange(dst, old, new, success, failure);
    res.unwrap_or_else(|x| x)
}
