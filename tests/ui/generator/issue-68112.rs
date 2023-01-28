// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
#![feature(generators, generator_trait)]

use std::{
    cell::RefCell,
    sync::Arc,
    pin::Pin,
    ops::{Generator, GeneratorState},
};

pub struct Ready<T>(Option<T>);
impl<T: Unpin> Generator<()> for Ready<T> {
    type Return = T;
    type Yield = ();
    fn resume(mut self: Pin<&mut Self>, _args: ()) -> GeneratorState<(), T> {
        GeneratorState::Complete(self.0.take().unwrap())
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

fn make_non_send_generator() -> impl Generator<Return = Arc<RefCell<i32>>> {
    make_gen1(Arc::new(RefCell::new(0)))
}

fn test1() {
    let send_gen = || {
        let _non_send_gen = make_non_send_generator();
        //~^ NOTE not `Send`
        yield;
        //~^ NOTE yield occurs here
        //~| NOTE value is used across a yield
    }; //[no_drop_tracking,drop_tracking]~ NOTE later dropped here
    require_send(send_gen);
    //~^ ERROR generator cannot be sent between threads
    //~| NOTE not `Send`
    //~| NOTE use `std::sync::RwLock` instead
}

pub fn make_gen2<T>(t: T) -> impl Generator<Return = T> {
//~^ NOTE appears within the type
//~| NOTE expansion of desugaring
    || { //~ NOTE used within this generator
        yield;
        t
    }
}
fn make_non_send_generator2() -> impl Generator<Return = Arc<RefCell<i32>>> { //~ NOTE appears within the type
//~^ NOTE expansion of desugaring
    make_gen2(Arc::new(RefCell::new(0)))
}

fn test2() {
    let send_gen = || { //~ NOTE used within this generator
        let _non_send_gen = make_non_send_generator2();
        yield;
    };
    require_send(send_gen);
    //~^ ERROR `RefCell<i32>` cannot be shared between threads safely
    //~| NOTE `RefCell<i32>` cannot be shared between threads safely
    //~| NOTE required for
    //[no_drop_tracking,drop_tracking]~| NOTE required by a bound introduced by this call
    //~| NOTE captures the following types
    //~| NOTE use `std::sync::RwLock` instead
}

fn main() {}
