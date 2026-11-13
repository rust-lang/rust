//@ revisions: default full next
//@[default] check-pass
//@[full] check-fail
//@[next] check-pass
//@[full] compile-flags: -Zassumptions-on-binders -Znext-solver
//@[next] compile-flags: -Znext-solver

// FIXME(-Zassumptions-on-binders): full mode should accept this local borrow.
// Solver constraint conversion does not account for the implicit body lifetime.

#![allow(dead_code)]

trait Trait {
    type Assoc;
}

trait Object {
    fn method(&self) {}
}

impl<T> Object for T {}

fn body_bound<T: Trait>(value: T::Assoc) {
    let object: &dyn Object = &value;
    //[full]~^ ERROR unable to satisfy constraints because it cannot be proven
    object.method();
}

fn main() {}
