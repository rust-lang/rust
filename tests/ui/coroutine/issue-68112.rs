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
//~^ NOTE required by a bound
//~| NOTE required by a bound
//~| NOTE required by this bound
//~| NOTE required by this bound

fn make_non_send_coroutine() -> impl Coroutine<Return = Arc<RefCell<i32>>> {
    make_gen1(Arc::new(RefCell::new(0)))
}

fn test1() {
    let send_gen = #[coroutine] || {
        let _non_send_gen = make_non_send_coroutine();
        //~^ NOTE not `Send`
        yield;
        //~^ NOTE yield occurs here
        //~| NOTE value is used across a yield
    };
    require_send(send_gen);
    //~^ ERROR coroutine cannot be sent between threads
    //~| NOTE not `Send`
    //~| NOTE use `std::sync::RwLock` instead
}

pub fn make_gen2<T>(t: T) -> impl Coroutine<Return = T> {
//~^ NOTE appears within the type
//~| NOTE expansion of desugaring
    #[coroutine] || { //~ NOTE used within this coroutine
        yield;
        t
    }
}
fn make_non_send_coroutine2() -> impl Coroutine<Return = Arc<RefCell<i32>>> { //~ NOTE appears within the type
//~^ NOTE expansion of desugaring
    make_gen2(Arc::new(RefCell::new(0)))
}

fn test2() {
    let send_gen = #[coroutine] || { //~ NOTE used within this coroutine
        let _non_send_gen = make_non_send_coroutine2();
        yield;
    };
    require_send(send_gen);
    //~^ ERROR `RefCell<i32>` cannot be shared between threads safely
    //~| NOTE `RefCell<i32>` cannot be shared between threads safely
    //~| NOTE required for
    //~| NOTE use `std::sync::RwLock` instead
}

fn main() {}
