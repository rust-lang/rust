//@ needs-unwind - depends on landing pads being optimized away, so not useful to run without it
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#[inline(never)]
fn inner(_: &dyn Sync) {}

fn wrapper<T: Sync>(val: T) {
    inner(&val);
}

// Verify that there are no landing pads produced.
// CHECK-LABEL: unused_drop_pre_llvm::wrapper::<u32>
// CHECk-NOT: resume
// CHECk-NOT: landingpad
// The next line checks for the } that ends the function definition
// CHECK-LABEL: {{^[}]}}
#[inline(never)]
pub fn wrapper_u32() {
    wrapper(1u32);
}

#[inline(never)]
pub fn wrapper_u32_manual(x: u32) {
    inner(&x);
}
