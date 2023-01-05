// ignore-debug: the debug assertions get in the way
// no-system-llvm
// compile-flags: -O
#![crate_type="lib"]

#[no_mangle]
pub fn get_len() -> usize {
    // CHECK-LABEL: @get_len
    // CHECK-NOT: call
    // CHECK-NOT: invoke
    [1, 2, 3].iter().collect::<Vec<_>>().len()
}
