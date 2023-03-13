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
    use crate::{intrinsics, sync::atomic::Ordering::*};

    debug_assert!(dst as usize % 16 == 0);

    // Copied from `atomic_compare_exchange` in `core`.
    // https://github.com/rust-lang/rust/blob/f8a2e49/library/core/src/sync/atomic.rs#L3046-L3079
    let (val, _ok) = match (success, failure) {
        (Relaxed, Relaxed) => intrinsics::atomic_cxchg_relaxed_relaxed(dst, old, new),
        (Relaxed, Acquire) => intrinsics::atomic_cxchg_relaxed_acquire(dst, old, new),
        (Relaxed, SeqCst) => intrinsics::atomic_cxchg_relaxed_seqcst(dst, old, new),
        (Acquire, Relaxed) => intrinsics::atomic_cxchg_acquire_relaxed(dst, old, new),
        (Acquire, Acquire) => intrinsics::atomic_cxchg_acquire_acquire(dst, old, new),
        (Acquire, SeqCst) => intrinsics::atomic_cxchg_acquire_seqcst(dst, old, new),
        (Release, Relaxed) => intrinsics::atomic_cxchg_release_relaxed(dst, old, new),
        (Release, Acquire) => intrinsics::atomic_cxchg_release_acquire(dst, old, new),
        (Release, SeqCst) => intrinsics::atomic_cxchg_release_seqcst(dst, old, new),
        (AcqRel, Relaxed) => intrinsics::atomic_cxchg_acqrel_relaxed(dst, old, new),
        (AcqRel, Acquire) => intrinsics::atomic_cxchg_acqrel_acquire(dst, old, new),
        (AcqRel, SeqCst) => intrinsics::atomic_cxchg_acqrel_seqcst(dst, old, new),
        (SeqCst, Relaxed) => intrinsics::atomic_cxchg_seqcst_relaxed(dst, old, new),
        (SeqCst, Acquire) => intrinsics::atomic_cxchg_seqcst_acquire(dst, old, new),
        (SeqCst, SeqCst) => intrinsics::atomic_cxchg_seqcst_seqcst(dst, old, new),
        (_, AcqRel) => panic!("there is no such thing as an acquire-release failure ordering"),
        (_, Release) => panic!("there is no such thing as a release failure ordering"),

        // `atomic::Ordering` is non_exhaustive. It warns when `core_arch` is built as a part of `core`.
        #[allow(unreachable_patterns)]
        (_, _) => unreachable!(),
    };
    val
}
