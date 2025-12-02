#![allow(todo_macro_calls)]
#[macro_export]
macro_rules! external_todo {
    {} => { todo!() };
}
