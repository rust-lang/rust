// compile-flags: -O

#![crate_type = "lib"]

#[no_mangle]
pub fn push(v: &mut Vec<u8>) {
    let _ = v.reserve(4);
    // CHECK-NOT: call {{.*}}reserve_for_push
    v.push(1);
    // CHECK-NOT: call {{.*}}reserve_for_push
    v.push(2);
    // CHECK-NOT: call {{.*}}reserve_for_push
    v.push(3);
    // CHECK-NOT: call {{.*}}reserve_for_push
    v.push(4);
}