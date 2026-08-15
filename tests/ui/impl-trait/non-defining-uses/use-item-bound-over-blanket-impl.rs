//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
#![allow(unconditional_recursion)]

// Regression test for trait-system-refactor-initiative#182.

trait Id {
    type This;
}
impl<T> Id for T {
    type This = T;
}
fn to_assoc<T>(x: T) -> <T as Id>::This {
    x
}

fn mirror<T>(x: Vec<T>) -> impl Id<This = Vec<T>> {
    let x = to_assoc(mirror(x));
    // `?x` equals `<opaque::<T> as Id>::This`. We need to eagerly infer the
    // type of `?x` to prevent this method call from resulting in an error.
    //
    // We could use both the item bound to normalize to `Vec<T>`, or the
    // blanket impl to normalize to `opaque::<T>`. We have to go with the
    // item bound.
    x.len();
    x
}
fn main() {}
