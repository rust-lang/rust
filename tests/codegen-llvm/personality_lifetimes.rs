//@ ignore-msvc
//@ needs-unwind

//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]

struct S;

impl Drop for S {
    #[inline(never)]
    fn drop(&mut self) {}
}

#[inline(never)]
fn might_unwind() {}

// CHECK-LABEL: @test
#[no_mangle]
pub fn test() {
    let _s = S;
    // Check that the personality slot alloca gets a lifetime start in each cleanup block, not just
    // in the first one.
    // CHECK: [[SLOT:%[0-9]+]] = alloca [{{[0-9]+}} x i8]
    // CHECK-LABEL: cleanup:
    // CHECK: call void @llvm.lifetime.start.{{.*}}({{.*}})
    // CHECK-LABEL: cleanup1:
    // CHECK: call void @llvm.lifetime.start.{{.*}}({{.*}})
    might_unwind();
    let _t = S;
    might_unwind();
}
