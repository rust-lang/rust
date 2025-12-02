#![allow(todo_macro_calls)]
fn main() {
    todo!();
}

fn other(_: i32)) {} //~ ERROR unexpected closing delimiter: `)`
