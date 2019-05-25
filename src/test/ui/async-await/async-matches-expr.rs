// compile-pass
// edition:2018

#![feature(async_await, await_macro)]

macro_rules! match_expr {
    ($x:expr) => {}
}

fn main() {
    match_expr!(async {});
    match_expr!(async || {});
}
