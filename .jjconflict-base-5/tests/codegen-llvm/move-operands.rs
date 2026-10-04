// Verify that optimized MIR only copies `a` once.
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]

type T = [u8; 256];

#[no_mangle]
pub fn f(a: T, b: fn(_: T, _: T)) {
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 1 %{{.*}}, ptr align 1 %{{.*}}, {{.*}} 256, i1 false)
    // CHECK-NOT: call void @llvm.memcpy.{{.*}}(ptr align 1 %{{.*}}, ptr align 1 %{{.*}}, {{.*}} 256, i1 false)
    b(a, a)
}
