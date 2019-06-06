// ignore-debug: the debug assertions get in the way
// no-system-llvm
// compile-flags: -O
#![crate_type="lib"]

#[no_mangle]
pub fn get_len() -> usize {
    // CHECK-COUNT-1: {{^define}}
    [1, 2, 3].iter().collect::<Vec<_>>().len()
}
