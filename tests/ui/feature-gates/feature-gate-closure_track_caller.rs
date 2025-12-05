//@ edition:2021
#![feature(stmt_expr_attributes)]
#![feature(coroutines)]

fn main() {
    let _closure = #[track_caller] || {}; //~ ERROR `#[track_caller]` on closures
    let _coroutine = #[coroutine] #[track_caller] || { yield; }; //~ ERROR `#[track_caller]` on closures
    let _future = #[track_caller] async {}; //~ ERROR `#[track_caller]` on closures
}
