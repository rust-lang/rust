//@ run-pass

#![feature(core_intrinsics, intrinsics)]

// NOTE(eddyb) `#[inline(never)]` and returning `fn` pointers from functions is
// done to force codegen (of the reification-to-`fn`-ptr shims around intrinsics).

#[inline(never)]
fn a() -> unsafe fn(isize) -> usize {
    let f: unsafe fn(isize) -> usize = std::mem::transmute;
    f
}

#[inline(never)]
fn b() -> unsafe fn(isize) -> usize {
    let f = std::mem::transmute as unsafe fn(isize) -> usize;
    f
}

#[inline(never)]
fn c() -> [fn(bool) -> bool; 2] {
    let fs = [
        std::intrinsics::likely,
        std::intrinsics::unlikely,
    ];
    fs
}

fn main() {
    unsafe {
        assert_eq!(a()(-1), !0);
        assert_eq!(b()(-1), !0);
    }

    let [likely_ptr, unlikely_ptr] = c();
    assert!(likely_ptr(true));
    assert!(unlikely_ptr(true));
}
