//@ compile-flags: -Copt-level=3

// See https://github.com/rust-lang/rust/issues/135802

#![crate_type = "lib"]

enum Void {}

// Should be ABI-compatible with T, but wasn't prior to the PR adding this test.
#[repr(transparent)]
struct NoReturn<T>(T, Void);

// Returned by invisible reference (in most ABIs)
#[allow(dead_code)]
struct Large(u64, u64, u64);

extern "Rust" {
    fn opaque() -> NoReturn<Large>;
    fn opaque_with_arg(rsi: u32) -> NoReturn<Large>;
}

// CHECK-LABEL: @test_uninhabited_ret_by_ref
#[no_mangle]
pub fn test_uninhabited_ret_by_ref() {
    // CHECK: %_1 = alloca [24 x i8], align {{8|4}}
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 24, )?}}ptr nonnull %_1)
    // CHECK-NEXT: call void @opaque({{.*}} sret([24 x i8]) {{.*}} %_1) #2
    // CHECK-NEXT: unreachable
    unsafe {
        opaque();
    }
}

// CHECK-LABEL: @test_uninhabited_ret_by_ref_with_arg
#[no_mangle]
pub fn test_uninhabited_ret_by_ref_with_arg(rsi: u32) {
    // CHECK: %_2 = alloca [24 x i8], align {{8|4}}
    // CHECK-NEXT: call void @llvm.lifetime.start.p0({{(i64 24, )?}}ptr nonnull %_2)
    // CHECK-NEXT: call void @opaque_with_arg({{.*}} sret([24 x i8]) {{.*}} %_2, i32 noundef{{( signext)?}} %rsi) #2
    // CHECK-NEXT: unreachable
    unsafe {
        opaque_with_arg(rsi);
    }
}
