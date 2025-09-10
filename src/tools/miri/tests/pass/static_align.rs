#![feature(static_align)]

// When a static uses `align(N)`, its address should be a multiple of `N`.

#[rustc_align_static(256)]
static FOO: u64 = 0;

#[rustc_align_static(512)]
static BAR: u64 = 0;

fn main() {
    assert!(core::ptr::from_ref(&FOO).addr().is_multiple_of(256));
    assert!(core::ptr::from_ref(&BAR).addr().is_multiple_of(512));
}
