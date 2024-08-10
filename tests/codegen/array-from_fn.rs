//@ revisions: NORMAL OPT
//@ [NORMAL] compile-flags: -C opt-level=0 -C debuginfo=2
//@ [OPT] compile-flags: -C opt-level=s -C debuginfo=0

#![crate_type = "lib"]
#![feature(array_from_fn)]

#[no_mangle]
pub fn iota() -> [u8; 16] {
    // OPT-NOT: core..array..Guard
    // NORMAL: core..array..Guard
    std::array::from_fn(|i| i as _)
}
