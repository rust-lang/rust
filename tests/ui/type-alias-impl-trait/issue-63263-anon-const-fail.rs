// Regression test for issue #63263.
// Tests that we properly handle closures with an explicit return type
// that return an opaque type.

// check-pass

#![feature(type_alias_impl_trait, inline_const)]

pub type Closure = impl FnOnce();

#[defines(Closure)]
fn main() {
    const {
        let x: Closure = || {};
    }
}
