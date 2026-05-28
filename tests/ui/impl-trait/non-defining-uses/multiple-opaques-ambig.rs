//@ compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
#![allow(unconditional_recursion)]

// Regression test for trait-system-refactor-initiative#182. If multiple
// opaque types result in different item bounds, do not apply them.

trait Trait<T> {}
impl<T, U> Trait<T> for U {}

fn impls_trait<T: Trait<U>, U>(_: T) -> U {
    todo!()
}

fn overlap<T, U>() -> (impl Trait<T>, impl Trait<U>) {
    let mut x = overlap::<T, U>().0;
    x = overlap::<T, U>().1;
    let u = impls_trait(x);
    let _: u32 = u;
    ((), ())
}
fn main() {}
