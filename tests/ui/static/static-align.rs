//@ run-pass
#![feature(static_align)]

#[rustc_align_static(64)]
static A: u8 = 0;

#[rustc_align_static(64)]
static B: u8 = 0;

#[rustc_align_static(128)]
#[no_mangle]
static EXPORTED: u64 = 0;

unsafe extern "C" {
    #[rustc_align_static(128)]
    #[link_name = "EXPORTED"]
    static C: u64;
}

fn main() {
    assert!(core::ptr::from_ref(&A).addr().is_multiple_of(64));
    assert!(core::ptr::from_ref(&B).addr().is_multiple_of(64));

    assert!(core::ptr::from_ref(&EXPORTED).addr().is_multiple_of(128));
    unsafe { assert!(core::ptr::from_ref(&C).addr().is_multiple_of(128)) };
}
