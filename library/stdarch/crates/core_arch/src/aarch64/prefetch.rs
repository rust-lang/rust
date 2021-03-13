#[cfg(test)]
use stdarch_test::assert_instr;

extern "C" {
    #[link_name = "llvm.prefetch"]
    fn prefetch(p: *const i8, rw: i32, loc: i32, ty: i32);
}

/// See [`prefetch`](fn._prefetch.html).
pub const _PREFETCH_READ: i32 = 0;

/// See [`prefetch`](fn._prefetch.html).
pub const _PREFETCH_WRITE: i32 = 1;

/// See [`prefetch`](fn._prefetch.html).
pub const _PREFETCH_LOCALITY0: i32 = 0;

/// See [`prefetch`](fn._prefetch.html).
pub const _PREFETCH_LOCALITY1: i32 = 1;

/// See [`prefetch`](fn._prefetch.html).
pub const _PREFETCH_LOCALITY2: i32 = 2;

/// See [`prefetch`](fn._prefetch.html).
pub const _PREFETCH_LOCALITY3: i32 = 3;

/// Fetch the cache line that contains address `p` using the given `RW` and `LOCALITY`.
///
/// The `RW` must be one of:
///
/// * [`_PREFETCH_READ`](constant._PREFETCH_READ.html): the prefetch is preparing
///   for a read.
///
/// * [`_PREFETCH_WRITE`](constant._PREFETCH_WRITE.html): the prefetch is preparing
///   for a write.
///
/// The `LOCALITY` must be one of:
///
/// * [`_PREFETCH_LOCALITY0`](constant._PREFETCH_LOCALITY0.html): Streaming or
///   non-temporal prefetch, for data that is used only once.
///
/// * [`_PREFETCH_LOCALITY1`](constant._PREFETCH_LOCALITY1.html): Fetch into level 3 cache.
///
/// * [`_PREFETCH_LOCALITY2`](constant._PREFETCH_LOCALITY2.html): Fetch into level 2 cache.
///
/// * [`_PREFETCH_LOCALITY3`](constant._PREFETCH_LOCALITY3.html): Fetch into level 1 cache.
///
/// The prefetch memory instructions signal to the memory system that memory accesses
/// from a specified address are likely to occur in the near future. The memory system
/// can respond by taking actions that are expected to speed up the memory access when
/// they do occur, such as preloading the specified address into one or more caches.
/// Because these signals are only hints, it is valid for a particular CPU to treat
/// any or all prefetch instructions as a NOP.
///
///
/// [Arm's documentation](https://developer.arm.com/documentation/den0024/a/the-a64-instruction-set/memory-access-instructions/prefetching-memory?lang=en)
#[inline(always)]
#[cfg_attr(test, assert_instr("prfm pldl1strm", RW = _PREFETCH_READ, LOCALITY = _PREFETCH_LOCALITY0))]
#[cfg_attr(test, assert_instr("prfm pldl3keep", RW = _PREFETCH_READ, LOCALITY = _PREFETCH_LOCALITY1))]
#[cfg_attr(test, assert_instr("prfm pldl2keep", RW = _PREFETCH_READ, LOCALITY = _PREFETCH_LOCALITY2))]
#[cfg_attr(test, assert_instr("prfm pldl1keep", RW = _PREFETCH_READ, LOCALITY = _PREFETCH_LOCALITY3))]
#[cfg_attr(test, assert_instr("prfm pstl1strm", RW = _PREFETCH_WRITE, LOCALITY = _PREFETCH_LOCALITY0))]
#[cfg_attr(test, assert_instr("prfm pstl3keep", RW = _PREFETCH_WRITE, LOCALITY = _PREFETCH_LOCALITY1))]
#[cfg_attr(test, assert_instr("prfm pstl2keep", RW = _PREFETCH_WRITE, LOCALITY = _PREFETCH_LOCALITY2))]
#[cfg_attr(test, assert_instr("prfm pstl1keep", RW = _PREFETCH_WRITE, LOCALITY = _PREFETCH_LOCALITY3))]
#[rustc_legacy_const_generics(1, 2)]
// FIXME: Replace this with the standard ACLE __pld/__pldx/__pli/__plix intrinsics
pub unsafe fn _prefetch<const RW: i32, const LOCALITY: i32>(p: *const i8) {
    // We use the `llvm.prefetch` instrinsic with `cache type` = 1 (data cache).
    static_assert_imm1!(RW);
    static_assert_imm2!(LOCALITY);
    prefetch(p, RW, LOCALITY, 1);
}
