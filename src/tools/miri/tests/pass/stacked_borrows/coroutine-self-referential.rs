// See https://github.com/rust-lang/unsafe-code-guidelines/issues/148:
// this fails when Stacked Borrows is strictly applied even to `!Unpin` types.
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn firstn() -> impl Coroutine<Yield = u64, Return = ()> {
    #[coroutine]
    static move || {
        let mut num = 0;
        let num = &mut num;

        yield *num;
        *num += 1; // would fail here

        yield *num;
        *num += 1;

        yield *num;
        *num += 1;
    }
}

fn main() {
    let mut coroutine_iterator = firstn();
    let mut pin = unsafe { Pin::new_unchecked(&mut coroutine_iterator) };
    let mut sum = 0;
    while let CoroutineState::Yielded(x) = pin.as_mut().resume(()) {
        sum += x;
    }
    assert_eq!(sum, 3);
}
