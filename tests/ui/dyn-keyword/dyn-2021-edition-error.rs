//@ edition:2021

fn function(x: &SomeTrait, y: Box<SomeTrait>) {
    //~^ ERROR expected a type, found a trait
    //~| ERROR expected a type, found a trait
    let _x: &SomeTrait = todo!();
    //~^ ERROR expected a type, found a trait
}

trait SomeTrait {}

fn main() {}
