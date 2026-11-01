// Regression test for <https://github.com/rust-lang/rust/issues/139873>.

// Test that we don't try to get the (nonexistent) name of the RPITIT in `Trait::foo`
// when emitting an error for a missing associated item `Trait::Output`.

trait Trait {
    fn foo() -> impl Sized;
    fn bar() -> Self::Output;
    //~^ ERROR associated type `Output` not found for `Self`
}

fn main() {}
