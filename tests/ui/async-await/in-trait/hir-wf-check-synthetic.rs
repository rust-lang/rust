//! Regression test for <https://github.com/rust-lang/rust/issues/159557>.
//! An async fn in a trait with a where-clause referencing an associated type
//! must not ICE in HIR wf-checking when the next solver is enabled.
//@ compile-flags: -Znext-solver=globally
//@ edition: 2024

//~^^^^^^  ERROR E0277
//~| ERROR E0277

trait Foo
//~^ ERROR E0277
where
    <Self as Mirror>::Assoc: Clone,
    //~^ ERROR E0277
    //~| ERROR E0277
{
    async fn e() {}
    //~^ ERROR E0277
    //~| ERROR E0277
    //~| ERROR the type `impl Future<Output = ()>` is not well-formed
    //~| ERROR the type `impl Future<Output = ()>` is not well-formed
    //~| ERROR the type `impl Future<Output = ()>` is not well-formed
    //~| ERROR E0271
}

trait Mirror {
    type Assoc;
}

fn main() {}
