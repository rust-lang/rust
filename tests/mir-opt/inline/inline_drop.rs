// Test for inlining with an indirect destination place.
//
//@ test-mir-pass: Inline
//@ edition: 2021
//@ needs-unwind
#![crate_type = "lib"]

#[inline(never)]
fn thing() {}

struct CallThingOnDrop;

impl Drop for CallThingOnDrop {
    #[inline]
    fn drop(&mut self) {
        thing();
    }
}

// EMIT_MIR inline_drop.drop_both_arguments.Inline.diff
pub fn drop_both_arguments(_a: CallThingOnDrop, _b: CallThingOnDrop) {
    // CHECK: drop(_2)
    // CHECK: drop(_1)
}
