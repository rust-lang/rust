// compile-fail
// edition:2018
// compile-flags: --crate-type lib

#![feature(async_await)]

pub const async fn x() {}
//~^ ERROR expected identifier, found reserved keyword `async`
//~^^ expected `:`, found keyword `fn`
