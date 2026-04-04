// Test for #151358, assertion failed: !worker_thread.is_null()
//~^ ERROR cycle detected when looking up span for `Default`
//
//@ compile-flags: -Z threads=2
//@ compare-output-by-lines
#![allow(todo_macro_calls)]

trait Default {}
use std::num::NonZero;
fn main() {
    NonZero();
    todo!();
}
