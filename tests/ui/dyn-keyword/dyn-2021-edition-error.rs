//@ edition:2021

fn function(x: &SomeTrait, y: Box<SomeTrait>) {
    //~^ ERROR expected a type, found a trait
    //~| ERROR expected a type, found a trait
    let _x: &SomeTrait = todo!();
    //~^ ERROR expected a type, found a trait
}

// Regression test for <https://github.com/rust-lang/rust/issues/138211>.
extern "C" {
    fn foo() -> *const SomeTrait;
    //~^ ERROR expected a type, found a trait
}

trait SomeTrait {}

fn main() {}
