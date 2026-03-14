#![allow(todo_macro_uses)]
#[macro_export]
macro_rules! external_todo {
    {} => { todo!() };
}
