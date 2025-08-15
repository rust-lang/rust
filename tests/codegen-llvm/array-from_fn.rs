//@ revisions: NORMAL OPT
//@ [NORMAL] compile-flags: -C opt-level=0 -C debuginfo=2
//@ [NORMAL] filecheck-flags: --check-prefix=NORMAL
//@ [OPT] compile-flags: -C opt-level=s -C debuginfo=0
//@ [OPT] filecheck-flags: --check-prefix=OPT

#![crate_type = "lib"]
#![feature(array_from_fn)]

#[no_mangle]
pub fn iota() -> [u8; 16] {
    // OPT-NOT: core..array..Guard
    // NORMAL: core..array..Guard
    std::array::from_fn(|i| i as _)
}
