// compile-flags: -C opt-level=0 -Z emit-invariant-markers=yes
#![crate_type="lib"]

#[no_mangle]
pub fn foo(x: &i32) {
    // CHECK-LABEL: @foo
    // CHECK: call {{.*}} @llvm.invariant.start{{[^(]*}}(i64 4
    // CHECK: call void @{{.*}}drop{{.*}}
    // CHECK: call void @llvm.invariant.end
    // CHECK: ret void
    drop(x);
}
