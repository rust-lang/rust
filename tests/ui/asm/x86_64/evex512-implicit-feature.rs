//@ build-pass
//@ only-x86_64
//@ compile-flags: --crate-type=lib -C target-cpu=skylake

use std::arch::x86_64::*;

#[target_feature(enable = "avx512f")]
#[no_mangle]
pub unsafe fn test(res: *mut f64, p: *const f64) {
    let arg = _mm512_load_pd(p);
    _mm512_store_pd(res, _mm512_fmaddsub_pd(arg, arg, arg));
}
