// Regression test for issue #63263.
// Tests that we properly handle closures with an explicit return type
// that return an opaque type.

// check-pass

#![feature(min_type_alias_impl_trait, type_alias_impl_trait)]
//~^ WARN incomplete

pub type Closure = impl FnOnce();

fn main() {
    || -> Closure { || () };
}
