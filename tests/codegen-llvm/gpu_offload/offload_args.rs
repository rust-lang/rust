//@ compile-flags: -Zoffload=Args -Zno-link -Zunstable-options -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-offload

// This test is meant to verify that we are able to map cpu argument to a device, and pass those to
// a gpu library like cuBLAS or rocblas. We don't really want to link those libraries in CI, and we
// neither want to deal with the creation or destruction of handles that those require since it's
// just noise. We do however test that we can combine host pointer (like alpha, beta) with device
// pointers (A, x, y). We also test std support while already at it.
// FIXME(offload): We should be able remove the no_mangle from the wrapper if we mark it as used.

#![allow(internal_features, non_camel_case_types, non_snake_case)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]

fn main() {
    let mut A: [f32; 3 * 2] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut x: [f32; 3] = [1.0, 1.0, 1.0];
    let mut y: [f32; 2] = [0.0, 0.0];
    for _ in 0..10 {
        core::intrinsics::offload_args::<_, _, ()>(rocblas_sgemv_wrapper, (&mut A, &mut x, &mut y));
        // CHECK-LABEL: ; offload_args::main
        // CHECK:   call void @__tgt_target_data_begin_mapper(
        // CHECK-NEXT: [[A:%.*]] = call ptr @omp_get_mapped_ptr(ptr nonnull %A, i32 0)
        // CHECK-NEXT: [[X:%.*]] = call ptr @omp_get_mapped_ptr(ptr nonnull %x, i32 0)
        // CHECK-NEXT: [[Y:%.*]] = call ptr @omp_get_mapped_ptr(ptr nonnull %y, i32 0)
        // CHECK-NEXT: call {{.*}}void @rocblas_sgemv_wrapper(ptr [[A]], ptr [[X]], ptr [[Y]])
        // CHECK-NEXT: call void @__tgt_target_data_end_mapper(
    }
    println!("{:?}", y);
}

unsafe extern "C" {
    pub fn fake_gpublas_sgemv(
        m: i32,
        n: i32,
        alpha: *const f32,
        A: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: *const f32,
        y: *mut f32,
        incy: i32,
    ) -> i32;
}

#[unsafe(no_mangle)]
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

    // CHECK-LABEL: define {{.*}}void @rocblas_sgemv_wrapper(ptr{{.*}} %A, ptr{{.*}} %x, ptr{{.*}} %y)
    // CHECK-DAG: %alpha = alloca [4 x i8]
    // CHECK-DAG: %beta  = alloca [4 x i8]
    // CHECK-DAG: store float 1.000000e+00, ptr %alpha
    // CHECK-DAG: store float 1.000000e+00, ptr %beta
    // CHECK: call noundef i32 @fake_gpublas_sgemv(i32 noundef 2, i32 noundef 3, ptr{{.*}} %alpha, ptr{{.*}} %A, i32 noundef 2, ptr{{.*}} %x, i32 noundef 1, ptr{{.*}} %beta, ptr{{.*}} %y, i32 noundef 1)

    unsafe {
        let st_res = fake_gpublas_sgemv(
            m,
            n,
            &alpha as *const f32,
            A.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            &beta as *const f32,
            y.as_mut_ptr(),
            incy,
        );
        assert_eq!(st_res, 1);
    };
}
