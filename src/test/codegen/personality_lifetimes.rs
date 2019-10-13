// ignore-msvc

// compile-flags: -O -C no-prepopulate-passes

#![crate_type="lib"]

struct S;

impl Drop for S {
    fn drop(&mut self) {
    }
}

fn might_unwind() {
}

// CHECK-LABEL: @test
#[no_mangle]
pub fn test() {
    let _s = S;
    // Check that the personality slot alloca gets a lifetime start in each cleanup block, not just
    // in the first one.
    // CHECK: [[SLOT:%[0-9]+]] = alloca { i8*, i32 }
    // CHECK-LABEL: cleanup:
    // CHECK: [[BITCAST:%[0-9]+]] = bitcast { i8*, i32 }* [[SLOT]] to i8*
    // CHECK-NEXT: call void @llvm.lifetime.start.{{.*}}({{.*}}, i8* [[BITCAST]])
    // CHECK-LABEL: cleanup1:
    // CHECK: [[BITCAST1:%[0-9]+]] = bitcast { i8*, i32 }* [[SLOT]] to i8*
    // CHECK-NEXT: call void @llvm.lifetime.start.{{.*}}({{.*}}, i8* [[BITCAST1]])
    might_unwind();
    let _t = S;
    might_unwind();
}
