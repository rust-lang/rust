//@ compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
#![allow(unconditional_recursion)]

// Test for trait-system-refactor-initiative#182 making sure
// that we don't incorrectly normalize to rigid aliases if the
// opaque type only has a trait bound.

trait Id {
    type This;
}
impl<T> Id for Vec<T> {
    type This = Vec<T>;
}
fn to_assoc<T: Id>(x: T) -> <T as Id>::This {
    todo!()
}

fn mirror<T>(x: Vec<T>) -> impl Id {
    let x = to_assoc(mirror(x));
    // `?x` equals `<opaque::<T> as Id>::This`. We should not infer `?x`
    // to be a rigid alias here.
    let _: Vec<u32> = x;
    x
}
fn main() {}
