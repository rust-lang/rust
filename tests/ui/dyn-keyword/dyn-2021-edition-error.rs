//@ edition:2021

fn function(x: &SomeTrait, y: Box<SomeTrait>) {
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    let _x: &SomeTrait = todo!();
    //~^ ERROR trait objects must include the `dyn` keyword
}

trait SomeTrait {}

fn main() {}
