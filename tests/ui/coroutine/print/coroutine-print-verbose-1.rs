//@ compile-flags: -Zverbose-internals

// Same as: tests/ui/coroutine/issue-68112.stderr

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::{
    cell::RefCell,
    sync::Arc,
    pin::Pin,
    ops::{Coroutine, CoroutineState},
};

pub struct Ready<T>(Option<T>);
impl<T: Unpin> Coroutine<()> for Ready<T> {
    type Return = T;
    type Yield = ();
    fn resume(mut self: Pin<&mut Self>, _args: ()) -> CoroutineState<(), T> {
        CoroutineState::Complete(self.0.take().unwrap())
    }
}
pub fn make_gen1<T>(t: T) -> Ready<T> {
    Ready(Some(t))
}

fn require_send(_: impl Send) {}

fn make_non_send_coroutine() -> impl Coroutine<Return = Arc<RefCell<i32>>> {
    make_gen1(Arc::new(RefCell::new(0)))
}

fn test1() {
    let send_gen = #[coroutine] || {
        let _non_send_gen = make_non_send_coroutine();
        yield;
    };
    require_send(send_gen);
    //~^ ERROR coroutine cannot be sent between threads
}

pub fn make_gen2<T>(t: T) -> impl Coroutine<Return = T> {
    #[coroutine] || {
        yield;
        t
    }
}
fn make_non_send_coroutine2() -> impl Coroutine<Return = Arc<RefCell<i32>>> {
    make_gen2(Arc::new(RefCell::new(0)))
}

fn test2() {
    let send_gen = #[coroutine] || {
        let _non_send_gen = make_non_send_coroutine2();
        yield;
    };
    require_send(send_gen);
    //~^ ERROR `RefCell<i32>` cannot be shared between threads safely
}

fn main() {}
