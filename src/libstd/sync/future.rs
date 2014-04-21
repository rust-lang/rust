// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * A type representing values that may be computed concurrently and
 * operations for working with them.
 *
 * # Example
 *
 * ```rust
 * use std::sync::Future;
 * # fn fib(n: uint) -> uint {42};
 * # fn make_a_sandwich() {};
 * let mut delayed_fib = Future::spawn(proc() { fib(5000) });
 * make_a_sandwich();
 * println!("fib(5000) = {}", delayed_fib.get())
 * ```
 */

#![allow(missing_doc)]

use core::prelude::*;
use core::mem::replace;

use comm::{Receiver, channel};
use task::spawn;

/// A type encapsulating the result of a computation which may not be complete
pub struct Future<A> {
    state: FutureState<A>,
}

enum FutureState<A> {
    Pending(proc():Send -> A),
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
    pub fn unwrap(mut self) -> A {
        self.get_ref();
        let state = replace(&mut self.state, Evaluating);
        match state {
            Forced(v) => v,
            _ => fail!( "Logic error." ),
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
            Evaluating => fail!("Recursive forcing of future!"),
            Pending(_) => {
                match replace(&mut self.state, Evaluating) {
                    Forced(_) | Evaluating => fail!("Logic error."),
                    Pending(f) => {
                        self.state = Forced(f());
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

    pub fn from_fn(f: proc():Send -> A) -> Future<A> {
        /*!
         * Create a future from a function.
         *
         * The first time that the value is requested it will be retrieved by
         * calling the function.  Note that this function is a local
         * function. It is not spawned into another task.
         */

        Future {state: Pending(f)}
    }
}

impl<A:Send> Future<A> {
    pub fn from_receiver(rx: Receiver<A>) -> Future<A> {
        /*!
         * Create a future from a port
         *
         * The first time that the value is requested the task will block
         * waiting for the result to be received on the port.
         */

        Future::from_fn(proc() {
            rx.recv()
        })
    }

    pub fn spawn(blk: proc():Send -> A) -> Future<A> {
        /*!
         * Create a future from a unique closure.
         *
         * The closure will be run in a new task and its result used as the
         * value of the future.
         */

        let (tx, rx) = channel();

        spawn(proc() {
            // Don't fail if the other end has hung up
            let _ = tx.send_opt(blk());
        });

        Future::from_receiver(rx)
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use sync::Future;
    use task;
    use comm::{channel, Sender};

    #[test]
    fn test_from_value() {
        let mut f = Future::from_value("snail".to_string());
        assert_eq!(f.get(), "snail".to_string());
    }

    #[test]
    fn test_from_receiver() {
        let (tx, rx) = channel();
        tx.send("whale".to_string());
        let mut f = Future::from_receiver(rx);
        assert_eq!(f.get(), "whale".to_string());
    }

    #[test]
    fn test_from_fn() {
        let mut f = Future::from_fn(proc() "brail".to_string());
        assert_eq!(f.get(), "brail".to_string());
    }

    #[test]
    fn test_interface_get() {
        let mut f = Future::from_value("fail".to_string());
        assert_eq!(f.get(), "fail".to_string());
    }

    #[test]
    fn test_interface_unwrap() {
        let f = Future::from_value("fail".to_string());
        assert_eq!(f.unwrap(), "fail".to_string());
    }

    #[test]
    fn test_get_ref_method() {
        let mut f = Future::from_value(22i);
        assert_eq!(*f.get_ref(), 22);
    }

    #[test]
    fn test_spawn() {
        let mut f = Future::spawn(proc() "bale".to_string());
        assert_eq!(f.get(), "bale".to_string());
    }

    #[test]
    #[should_fail]
    fn test_futurefail() {
        let mut f = Future::spawn(proc() fail!());
        let _x: String = f.get();
    }

    #[test]
    fn test_sendable_future() {
        let expected = "schlorf";
        let f = Future::spawn(proc() { expected });
        task::spawn(proc() {
            let mut f = f;
            let actual = f.get();
            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn test_dropped_future_doesnt_fail() {
        struct Bomb(Sender<bool>);

        local_data_key!(LOCAL: Bomb)

        impl Drop for Bomb {
            fn drop(&mut self) {
                let Bomb(ref tx) = *self;
                tx.send(task::failing());
            }
        }

        // Spawn a future, but drop it immediately. When we receive the result
        // later on, we should never view the task as having failed.
        let (tx, rx) = channel();
        drop(Future::spawn(proc() {
            LOCAL.replace(Some(Bomb(tx)));
        }));

        // Make sure the future didn't fail the task.
        assert!(!rx.recv());
    }
}
