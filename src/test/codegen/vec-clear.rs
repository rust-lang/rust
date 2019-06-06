// ignore-test

// FIXME:
// LLVM can't optimize some loops with unknown number of iterations because of
// @llvm.sideeffect() (see also #59546)

// ignore-debug: the debug assertions get in the way
// compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @vec_clear
#[no_mangle]
pub fn vec_clear(x: &mut Vec<u32>) {
    // CHECK-NOT: load
    // CHECK-NOT: icmp
    x.clear()
}
