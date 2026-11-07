//! Regression test for <https://github.com/rust-lang/rust/issues/148620>.

trait Trait<'a> {
    const K: &'a ();
}

fn foo() -> dyn Trait<'r, K = {}> {
    //~^ ERROR use of undeclared lifetime name `'r`
    //~| ERROR associated const equality is incomplete
    {
        6
        //~^ ERROR mismatched types
    }
    5
}

fn main() {}
