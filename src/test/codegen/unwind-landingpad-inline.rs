// no-system-llvm: needs #92110 + patch for Rust alloc/dealloc functions
// compile-flags: -Copt-level=3
#![crate_type = "lib"]

// This test checks that we can inline drop_in_place in
// unwind landing pads. Without this, the box pointers escape,
// and LLVM will not optimize out the pointer comparison.
// See https://github.com/rust-lang/rust/issues/46515

// Everything should be optimized out.
// CHECK-LABEL: @check_no_escape_in_landingpad
// CHECK: start:
// CHECK-NEXT: ret void
#[no_mangle]
pub fn check_no_escape_in_landingpad(f: fn()) {
    let x = &*Box::new(0);
    let y = &*Box::new(0);

    if x as *const _ == y as *const _ {
        f();
    }
}
