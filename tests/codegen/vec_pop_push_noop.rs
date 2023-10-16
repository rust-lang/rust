// compile-flags: -O

#![crate_type = "lib"]

#[no_mangle]
// CHECK-LABEL: @noop(
pub fn noop(v: &mut Vec<u8>) {
    // CHECK-NOT: reserve_for_push
    // CHECK-NOT: call
    // CHECK: tail call void @llvm.assume
    // CHECK-NOT: reserve_for_push
    // CHECK-NOT: call
    // CHECK: ret
    if let Some(x) = v.pop() {
        v.push(x)
    }
}

#[no_mangle]
// CHECK-LABEL: @push_byte(
pub fn push_byte(v: &mut Vec<u8>) {
    // CHECK: call {{.*}}reserve_for_push
    v.push(3);
}
