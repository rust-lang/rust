//! Regression test for <https://github.com/rust-lang/rust/issues/159557>.
//! An async fn in a trait with a where-clause referencing an associated type
//! must not ICE in HIR wf-checking when the next solver is enabled.
//@ compile-flags: -Znext-solver=globally
//@ edition: 2024
//@ dont-require-annotations: ERROR

trait Foo
where
    <Self as Mirror>::Assoc: Clone,
{
    async fn e() {}
}

trait Mirror {
    type Assoc;
}

fn main() {}
