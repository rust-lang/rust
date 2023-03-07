// Regression test for issue #63263.
// Tests that we properly handle closures with an explicit return type
// that return an opaque type.

// check-pass

#![feature(type_alias_impl_trait)]

fn main() {
    pub type Closure = impl FnOnce();
    || -> Closure { || () };
}
