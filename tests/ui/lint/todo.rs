//@ aux-build:external_macros.rs

#![deny(todo_macro_uses)]

extern crate external_macros;

use external_macros::external_todo;

fn std_todo() {
    todo!(); //~ ERROR `todo!()` macro used
    todo!("whatever"); //~ ERROR `todo!()` macro used
    //~^ ERROR `todo!()` macro used
}

macro_rules! locally_expanded_todo {
    {} => {
        todo!("who'd have thunk?") //~ ERROR `todo!()` macro used
        //~^ ERROR `todo!()` macro used
    };
}

#[allow(todo_macro_uses)]
fn allowed_todo() {
    todo!();
}

fn main() {
    std_todo();
    locally_expanded_todo!();
    allowed_todo();
    external_todo!();
}
