// Regression test for issue #63263.
// Tests that we properly handle closures with an explicit return type
// that return an opaque type.

#![feature(type_alias_impl_trait, inline_const)]

pub type Closure = impl FnOnce();

fn main() {
    const {
        let x: Closure = || {};
        //~^ ERROR: opaque type constrained without being represented in the signature
    }
}
