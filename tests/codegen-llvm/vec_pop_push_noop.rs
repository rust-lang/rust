//@ compile-flags: -Copt-level=3
//@ revisions: new old
//@ [old] max-llvm-major-version: 21
//@ [new] min-llvm-version: 22

#![crate_type = "lib"]

#[no_mangle]
// CHECK-LABEL: @noop(
pub fn noop(v: &mut Vec<u8>) {
    // CHECK-NOT: grow_one
    // CHECK-NOT: call
    // old: tail call void @llvm.assume
    // CHECK-NOT: grow_one
    // CHECK-NOT: call
    // CHECK: {{ret|[}]}}
    if let Some(x) = v.pop() {
        v.push(x)
    }
}

#[no_mangle]
// CHECK-LABEL: @push_byte(
pub fn push_byte(v: &mut Vec<u8>) {
    // CHECK: call {{.*}}grow_one
    v.push(3);
}
