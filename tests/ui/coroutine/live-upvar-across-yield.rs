//@ run-pass

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let b = |_| 3;
    let mut a = #[coroutine] || {
        b(yield);
    };
    Pin::new(&mut a).resume(());
}
