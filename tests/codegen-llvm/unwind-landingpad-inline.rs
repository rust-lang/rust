//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// This test checks that we can inline drop_in_place in
// unwind landing pads.

// Without inlining, the box pointers escape via the call to drop_in_place,
// and LLVM will not optimize out the pointer comparison.
// With inlining, everything should be optimized out.
// See https://github.com/rust-lang/rust/issues/46515
// CHECK-LABEL: @check_no_escape_in_landingpad
// CHECK: start:
// CHECK-NEXT: ; call __rustc::__rust_no_alloc_shim_is_unstable_v2
// CHECK-NEXT: tail call void @[[NO_ALLOC_SHIM:_R.+__rust_no_alloc_shim_is_unstable_v2]]()
// CHECK-NEXT: ; call __rustc::__rust_no_alloc_shim_is_unstable_v2
// CHECK-NEXT: tail call void @[[NO_ALLOC_SHIM]]()
// CHECK-NEXT: ret void
#[no_mangle]
pub fn check_no_escape_in_landingpad(f: fn()) {
    let x = &*Box::new(0);
    let y = &*Box::new(0);

    if x as *const _ == y as *const _ {
        f();
    }
}

// Without inlining, the compiler can't tell that
// dropping an empty string (in a landing pad) does nothing.
// With inlining, the landing pad should be optimized out.
// See https://github.com/rust-lang/rust/issues/87055
// CHECK-LABEL: @check_eliminate_noop_drop
// CHECK: call void %g()
// CHECK-NEXT: ret void
#[no_mangle]
pub fn check_eliminate_noop_drop(g: fn()) {
    let _var = String::new();
    g();
}
