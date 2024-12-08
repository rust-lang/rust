//@ edition:2021
#![feature(stmt_expr_attributes)]
#![feature(coroutines)]

fn main() {
    let _closure = #[track_caller] || {}; //~ `#[track_caller]` on closures
    let _coroutine = #[coroutine] #[track_caller] || { yield; }; //~ `#[track_caller]` on closures
    let _future = #[track_caller] async {}; //~ `#[track_caller]` on closures
}
