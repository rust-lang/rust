//
// min-llvm-version: 10.0.1
// compile-flags: -C opt-level=3
#![crate_type="lib"]

#[no_mangle]
pub fn alloc_test(a: &String) {
    // CHECK-LABEL: @alloc_test
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void
    let x = String::clone(a);
    drop(x);
}
