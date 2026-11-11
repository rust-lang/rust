//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @test
// CHECK-NEXT: start:
// CHECK-NEXT: tail call void @ext_fn0()
#[no_mangle]
pub fn test() {
    test_inner(Some(inner0));
}

fn test_inner(f_maybe: Option<fn()>) {
    if let Some(f) = f_maybe {
        f();
    }
}

fn inner0() {
    unsafe { ext_fn0() };
}

extern "C" {
    fn ext_fn0();
}
