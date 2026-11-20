//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
#![allow(unconditional_recursion)]
// Regression test for trait-system-refactor-initiative#182.

trait Id {
    type This;
}
impl<T> Id for Vec<T> {
    type This = Vec<T>;
}
fn to_assoc<T: Id>(x: T) -> <T as Id>::This {
    todo!()
}

fn mirror<T>(x: Vec<T>) -> impl Id<This = Vec<T>> {
    let x = to_assoc(mirror(x));
    // `?x` equals `<opaque::<T> as Id>::This`. We need to eagerly infer the
    // type of `?x` to prevent this method call from resulting in an error.
    x.len();
    x
}
fn main() {}
