//@ compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
#![allow(unconditional_recursion)]

// Regression test for trait-system-refactor-initiative#205. Avoid
// constraining other impl arguments when applying blanket impls,
// especially if the nested where-bounds of the blanket impl don't
// actually apply for the opaque.

// FIXME(-Znext-solver): This currently incompletely constrains the
// argument of `opaque: Trait<?x>` using the blanket impl of trait.
// Ideally we don't do that.

trait Trait<T> {}

impl<T: Copy> Trait<u32> for T {}
impl Trait<u64> for String {}
fn impls_trait<T: Trait<U>, U>(_: T) {}

fn test() -> impl Sized {
    let x = test();
    impls_trait(x); //~ ERROR the trait bound `String: Trait<u32>` is not satisfied
    String::new()
}
fn main() {}
