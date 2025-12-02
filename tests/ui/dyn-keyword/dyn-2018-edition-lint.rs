//@ edition:2018
#![allow(todo_macro_calls)]
#[deny(bare_trait_objects)]

fn function(x: &SomeTrait, y: Box<SomeTrait>) {
    //~^ ERROR trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    //~| ERROR trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    let _x: &SomeTrait = todo!();
    //~^ ERROR trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}

trait SomeTrait {}

fn main() {}
