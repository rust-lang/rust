// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// compile-flags: -Zverbose

// Same as: tests/ui/generator/issue-68112.stderr

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

fn make_non_send_generator() -> impl Generator<Return = Arc<RefCell<i32>>> {
    make_gen1(Arc::new(RefCell::new(0)))
}

fn test1() {
    let send_gen = || {
        let _non_send_gen = make_non_send_generator();
        yield;
    };
    require_send(send_gen);
    //~^ ERROR generator cannot be sent between threads
}

pub fn make_gen2<T>(t: T) -> impl Generator<Return = T> {
    || {
        yield;
        t
    }
}
fn make_non_send_generator2() -> impl Generator<Return = Arc<RefCell<i32>>> {
    make_gen2(Arc::new(RefCell::new(0)))
}

fn test2() {
    let send_gen = || {
        let _non_send_gen = make_non_send_generator2();
        yield;
    };
    require_send(send_gen);
    //~^ ERROR `RefCell<i32>` cannot be shared between threads safely
}

fn main() {}
