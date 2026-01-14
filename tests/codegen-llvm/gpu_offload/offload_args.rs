//@ compile-flags: -Zoffload=Args -Zunstable-options -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test is meant to verify that we are able to map cpu argument to a device, and pass those to
// a gpu library like cuBLAS or rocblas. We don't really want to link those libraries in CI, and we
// neither want to deal with the creation or destruction of handles that those require since it's
// just noise. We do however test that we can combine host pointer (like alpha, beta) with device
// pointers (A, x, y). We also test std support while already at it.

#![allow(internal_features, non_camel_case_types, non_snake_case)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]

const N: i32 = 10;

fn main() {
    let mut A: [f32; 3 * 2] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut x: [f32; 3] = [1.0, 1.0, 1.0];
    let mut y: [f32; 2] = [0.0, 0.0];
    for _ in 0..N {
        core::intrinsics::offload_args::<_, _, ()>(rocblas_sgemv_wrapper, (&mut A, &mut x, &mut y));
    }
    println!("{:?}", y);
}

unsafe extern "C" {
    pub fn fake_gpublas_sgemv(
        m: i32,
        n: i32,
        alpha: *const f32,
        A: *const [f32; 6],
        lda: i32,
        x: *const [f32; 3],
        incx: i32,
        beta: *const f32,
        y: *mut [f32; 2],
        incy: i32,
    ) -> i32;
}

#[inline(never)]
pub fn rocblas_sgemv_wrapper(A: &mut [f32; 6], x: &mut [f32; 3], y: &mut [f32; 2]) -> () {
    let m: i32 = 2;
    let n: i32 = 3;
    let incx: i32 = 1;
    let incy: i32 = 1;
    let lda = m;
    // those two by default should be host ptr:
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    unsafe {
        let st_res = rocblas_sgemv(
            m,
            n,
            &alpha as *const f32,
            A,
            lda,
            x,
            incx,
            &beta as *const f32,
            y,
            incy,
        );
        assert_eq!(st_res, 1);
    };
}
