use sync::atomic::Ordering;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Compare and exchange 16 bytes (128 bits) of data atomically.
///
/// This intrinsic corresponds to the `cmpxchg16b` instruction on x86_64
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
/// This atomic operations has the same semantics of memory orderings as
/// `AtomicUsize::compare_exchange` does, only operating on 16 bytes of memory
/// instead of just a pointer.
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
///
/// The `success` ordering must also be stronger or equal to `failure`, or this
/// function call is undefined. See the `Atomic*` documentation's
/// `compare_exchange` function for more information. When `compare_exchange`
/// panics, this is undefined behavior. Currently this function aborts the
/// process with an undefined instruction.
#[inline]
#[cfg_attr(test, assert_instr(cmpxchg16b, success = Ordering::SeqCst, failure = Ordering::SeqCst))]
#[target_feature(enable = "cmpxchg16b")]
#[cfg(not(stage0))]
pub unsafe fn cmpxchg16b(
    dst: *mut u128,
    old: u128,
    new: u128,
    success: Ordering,
    failure: Ordering,
) -> u128 {
    use intrinsics;
    use sync::atomic::Ordering::*;

    debug_assert!(dst as usize % 16 == 0);

    let (val, _ok) = match (success, failure) {
        (Acquire, Acquire) => intrinsics::atomic_cxchg_acq(dst, old, new),
        (Release, Relaxed) => intrinsics::atomic_cxchg_rel(dst, old, new),
        (AcqRel, Acquire) => intrinsics::atomic_cxchg_acqrel(dst, old, new),
        (Relaxed, Relaxed) => intrinsics::atomic_cxchg_relaxed(dst, old, new),
        (SeqCst, SeqCst) => intrinsics::atomic_cxchg(dst, old, new),
        (Acquire, Relaxed) => intrinsics::atomic_cxchg_acq_failrelaxed(dst, old, new),
        (AcqRel, Relaxed) => intrinsics::atomic_cxchg_acqrel_failrelaxed(dst, old, new),
        (SeqCst, Relaxed) => intrinsics::atomic_cxchg_failrelaxed(dst, old, new),
        (SeqCst, Acquire) => intrinsics::atomic_cxchg_failacq(dst, old, new),

        // The above block is all copied from libcore, and this statement is
        // also copied from libcore except that it's a panic in libcore and we
        // have a little bit more of a lightweight panic here.
        _ => ::core_arch::x86::ud2(),
    };
    val
}
