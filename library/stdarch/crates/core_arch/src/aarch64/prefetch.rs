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

/// Fetch the cache line that contains address `p` using the given `rw` and `locality`.
///
/// The `rw` must be one of:
///
/// * [`_PREFETCH_READ`](constant._PREFETCH_READ.html): the prefetch is preparing
///   for a read.
///
/// * [`_PREFETCH_WRITE`](constant._PREFETCH_WRITE.html): the prefetch is preparing
///   for a write.
///
/// The `locality` must be one of:
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
#[cfg_attr(test, assert_instr("prfm pldl1strm", rw = _PREFETCH_READ, locality = _PREFETCH_LOCALITY0))]
#[cfg_attr(test, assert_instr("prfm pldl3keep", rw = _PREFETCH_READ, locality = _PREFETCH_LOCALITY1))]
#[cfg_attr(test, assert_instr("prfm pldl2keep", rw = _PREFETCH_READ, locality = _PREFETCH_LOCALITY2))]
#[cfg_attr(test, assert_instr("prfm pldl1keep", rw = _PREFETCH_READ, locality = _PREFETCH_LOCALITY3))]
#[cfg_attr(test, assert_instr("prfm pstl1strm", rw = _PREFETCH_WRITE, locality = _PREFETCH_LOCALITY0))]
#[cfg_attr(test, assert_instr("prfm pstl3keep", rw = _PREFETCH_WRITE, locality = _PREFETCH_LOCALITY1))]
#[cfg_attr(test, assert_instr("prfm pstl2keep", rw = _PREFETCH_WRITE, locality = _PREFETCH_LOCALITY2))]
#[cfg_attr(test, assert_instr("prfm pstl1keep", rw = _PREFETCH_WRITE, locality = _PREFETCH_LOCALITY3))]
#[rustc_args_required_const(1, 2)]
pub unsafe fn _prefetch(p: *const i8, rw: i32, locality: i32) {
    // We use the `llvm.prefetch` instrinsic with `cache type` = 1 (data cache).
    // `rw` and `strategy` are based on the function parameters.
    macro_rules! pref {
        ($rdwr:expr, $local:expr) => {
            match ($rdwr, $local) {
                (0, 0) => prefetch(p, 0, 0, 1),
                (0, 1) => prefetch(p, 0, 1, 1),
                (0, 2) => prefetch(p, 0, 2, 1),
                (0, 3) => prefetch(p, 0, 3, 1),
                (1, 0) => prefetch(p, 1, 0, 1),
                (1, 1) => prefetch(p, 1, 1, 1),
                (1, 2) => prefetch(p, 1, 2, 1),
                (1, 3) => prefetch(p, 1, 3, 1),
                (_, _) => panic!(
                    "Illegal (rw, locality) pair in prefetch, value ({}, {}).",
                    $rdwr, $local
                ),
            }
        };
    }
    pref!(rw, locality);
}
