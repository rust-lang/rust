// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A type representing values that may be computed concurrently and operations
//! for working with them.
//!
//! # Examples
//!
//! ```
//! use std::sync::Future;
//!
//! // a fake, for now
//! fn fib(n: u32) -> u32 { 42 };
//!
//! let mut delayed_fib = Future::spawn(move || fib(5000));
//!
//! // do stuff...
//!
//! println!("fib(5000) = {}", delayed_fib.get())
//! ```

#![allow(missing_docs)]
#![unstable(feature = "std_misc",
            reason = "futures as-is have yet to be deeply reevaluated with recent \
                      core changes to Rust's synchronization story, and will likely \
                      become stable in the future but are unstable until that time")]

use core::prelude::*;
use core::mem::replace;

use self::FutureState::*;
use sync::mpsc::{Receiver, channel};
use thunk::{Thunk};
use thread;

/// A type encapsulating the result of a computation which may not be complete
pub struct Future<A> {
    state: FutureState<A>,
}

enum FutureState<A> {
    Pending(Thunk<'static,(),A>),
    Evaluating,
    Forced(A)
}

/// Methods on the `future` type
impl<A:Clone> Future<A> {
    pub fn get(&mut self) -> A {
        //! Get the value of the future.
        (*(self.get_ref())).clone()
    }
}

impl<A> Future<A> {
    /// Gets the value from this future, forcing evaluation.
    pub fn into_inner(mut self) -> A {
        self.get_ref();
        let state = replace(&mut self.state, Evaluating);
        match state {
            Forced(v) => v,
            _ => panic!( "Logic error." ),
        }
    }

    pub fn get_ref<'a>(&'a mut self) -> &'a A {
        /*!
        * Executes the future's closure and then returns a reference
        * to the result.  The reference lasts as long as
        * the future.
        */
        match self.state {
            Forced(ref v) => return v,
            Evaluating => panic!("Recursive forcing of future!"),
            Pending(_) => {
                match replace(&mut self.state, Evaluating) {
                    Forced(_) | Evaluating => panic!("Logic error."),
                    Pending(f) => {
                        self.state = Forced(f.invoke(()));
                        self.get_ref()
                    }
                }
            }
        }
    }

    pub fn from_value(val: A) -> Future<A> {
        /*!
         * Create a future from a value.
         *
         * The value is immediately available and calling `get` later will
         * not block.
         */

        Future {state: Forced(val)}
    }

    pub fn from_fn<F>(f: F) -> Future<A>
        where F : FnOnce() -> A, F : Send + 'static
    {
        /*!
         * Create a future from a function.
         *
         * The first time that the value is requested it will be retrieved by
         * calling the function.  Note that this function is a local
         * function. It is not spawned into another task.
         */

        Future {state: Pending(Thunk::new(f))}
    }
}

impl<A:Send+'static> Future<A> {
    pub fn from_receiver(rx: Receiver<A>) -> Future<A> {
        /*!
         * Create a future from a port
         *
         * The first time that the value is requested the task will block
         * waiting for the result to be received on the port.
         */

        Future::from_fn(move || {
            rx.recv().unwrap()
        })
    }

    pub fn spawn<F>(blk: F) -> Future<A>
        where F : FnOnce() -> A, F : Send + 'static
    {
        /*!
         * Create a future from a unique closure.
         *
         * The closure will be run in a new task and its result used as the
         * value of the future.
         */

        let (tx, rx) = channel();

        thread::spawn(move || {
            // Don't panic if the other end has hung up
            let _ = tx.send(blk());
        });

        Future::from_receiver(rx)
    }
}

#[cfg(test)]
mod test {
    use prelude::v1::*;
    use sync::mpsc::channel;
    use sync::Future;
    use thread;

    #[test]
    fn test_from_value() {
        let mut f = Future::from_value("snail".to_string());
        assert_eq!(f.get(), "snail");
    }

    #[test]
    fn test_from_receiver() {
        let (tx, rx) = channel();
        tx.send("whale".to_string()).unwrap();
        let mut f = Future::from_receiver(rx);
        assert_eq!(f.get(), "whale");
    }

    #[test]
    fn test_from_fn() {
        let mut f = Future::from_fn(move|| "brail".to_string());
        assert_eq!(f.get(), "brail");
    }

    #[test]
    fn test_interface_get() {
        let mut f = Future::from_value("fail".to_string());
        assert_eq!(f.get(), "fail");
    }

    #[test]
    fn test_interface_unwrap() {
        let f = Future::from_value("fail".to_string());
        assert_eq!(f.into_inner(), "fail");
    }

    #[test]
    fn test_get_ref_method() {
        let mut f = Future::from_value(22);
        assert_eq!(*f.get_ref(), 22);
    }

    #[test]
    fn test_spawn() {
        let mut f = Future::spawn(move|| "bale".to_string());
        assert_eq!(f.get(), "bale");
    }

    #[test]
    #[should_panic]
    fn test_future_panic() {
        let mut f = Future::spawn(move|| panic!());
        let _x: String = f.get();
    }

    #[test]
    fn test_sendable_future() {
        let expected = "schlorf";
        let (tx, rx) = channel();
        let f = Future::spawn(move|| { expected });
        let _t = thread::spawn(move|| {
            let mut f = f;
            tx.send(f.get()).unwrap();
        });
        assert_eq!(rx.recv().unwrap(), expected);
    }
}
