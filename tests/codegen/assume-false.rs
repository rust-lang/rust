//@ compile-flags: -Cno-prepopulate-passes -Zmir-opt-level=0 -O -Zinline-mir

#![crate_type = "lib"]

#[no_mangle]
pub fn unwrap_unchecked(x: Option<i32>) -> i32 {
    // CHECK-LABEL: define{{.*}} i32 @unwrap_unchecked
    // CHECK-NOT: call void @llvm.assume(i1 false)
    // CHECK: store i1 true, ptr poison, align 1
    // CHECK-NOT: call void @llvm.assume(i1 false)
    // CHECK: ret
    // CHECK }
    unsafe { x.unwrap_unchecked() }
}
