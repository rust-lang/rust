// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, platform_intrinsics)]

#[repr(simd, packed)]
pub struct Simd<T, const N: usize>([T; N]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct FullSimd<T, const N: usize>([T; N]);

extern "platform-intrinsic" {
    fn simd_mul<T>(a: T, b: T) -> T;
}

// non-powers-of-two have padding and need to be expanded to full vectors
fn load<T, const N: usize>(v: Simd<T, N>) -> FullSimd<T, N> {
    unsafe {
        let mut tmp = core::mem::MaybeUninit::<FullSimd<T, N>>::uninit();
        std::ptr::copy_nonoverlapping(&v as *const _, tmp.as_mut_ptr().cast(), 1);
        tmp.assume_init()
    }
}

// CHECK-LABEL: @square_packed
#[no_mangle]
pub fn square_packed(x: Simd<f32, 3>) -> FullSimd<f32, 3> {
    // CHECK: align 4 dereferenceable(12) %x
    let x = load(x);
    unsafe { simd_mul(x, x) }
}
