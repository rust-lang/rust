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
fn c() -> [unsafe fn(f32) -> f32; 2] {
    let fs = [
        std::intrinsics::floorf32,
        std::intrinsics::log2f32,
    ];
    fs
}

fn main() {
    unsafe {
        assert_eq!(a()(-1), !0);
        assert_eq!(b()(-1), !0);

        let [floorf32_ptr, log2f32_ptr] = c();
        assert_eq!(floorf32_ptr(1.5), 1.0);
        assert_eq!(log2f32_ptr(2.0), 1.0);
    }
}
