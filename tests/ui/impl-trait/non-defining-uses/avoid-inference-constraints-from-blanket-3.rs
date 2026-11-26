//@ compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
#![allow(unconditional_recursion)]

// Regression test for trait-system-refactor-initiative#205 and #229. Avoid
// constraining other impl arguments when applying blanket impls,
// especially if the nested where-bounds of the blanket impl don't
// actually apply for the opaque.

trait Trait<T> {}

impl<T: Copy> Trait<u32> for T {}
impl Trait<u64> for String {}
fn impls_trait<T: Trait<U>, U>(_: T) {}

fn test() -> impl Sized {
    let x = test();
    impls_trait(x);
    String::new()
}
fn main() {}
