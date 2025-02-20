#![crate_type = "lib"]
#![no_builtins]

type T = [u8; 256];

#[no_mangle]
pub fn f() -> [i32; 1000] {
    // CHECK: call void @llvm.memset.inline.{{.*}}(ptr nonnull align 4 %{{.*}}, i8 0, i64 4000, i1 false)
    [0; 1000]
}

#[no_mangle]
pub fn g(x: &[i32; 1000], y: &mut [i32; 1000]) {
    // CHECK: call void @llvm.memcpy.inline.{{.*}}(ptr nonnull align 4 %{{.*}}, ptr nonnull align 4 %{{.*}}, i64 4000, i1 false)
    *y = *x;
}
