// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
 * ~~~
 * let delayed_fib = future::spawn {|| fib(5000) };
 * make_a_sandwich();
 * io::println(fmt!("fib(5000) = %?", delayed_fib.get()))
 * ~~~
 */

use core::cast;
use core::cell::Cell;
use core::comm::{PortOne, oneshot, send_one};
use core::pipes::recv;
use core::task;

#[doc = "The future type"]
pub struct Future<A> {
    priv mut state: FutureState<A>,
}

// FIXME(#2829) -- futures should not be copyable, because they close
// over ~fn's that have pipes and so forth within!
#[unsafe_destructor]
impl<A> Drop for Future<A> {
    fn finalize(&self) {}
}

priv enum FutureState<A> {
    Pending(~fn() -> A),
    Evaluating,
    Forced(A)
}

/// Methods on the `future` type
pub impl<A:Copy> Future<A> {
    fn get(&self) -> A {
        //! Get the value of the future
        *(self.get_ref())
    }
}

pub impl<A> Future<A> {
    fn get_ref<'a>(&'a self) -> &'a A {
        /*!
        * Executes the future's closure and then returns a borrowed
        * pointer to the result.  The borrowed pointer lasts as long as
        * the future.
        */
        unsafe {
            match self.state {
                Forced(ref mut v) => { return cast::transmute(v); }
                Evaluating => fail!(~"Recursive forcing of future!"),
                Pending(_) => {}
            }

            let mut state = Evaluating;
            self.state <-> state;
            match state {
                Forced(_) | Evaluating => fail!(~"Logic error."),
                Pending(f) => {
                    self.state = Forced(f());
                    self.get_ref()
                }
            }
        }
    }
}

pub fn from_value<A>(val: A) -> Future<A> {
    /*!
     * Create a future from a value
     *
     * The value is immediately available and calling `get` later will
     * not block.
     */

    Future {state: Forced(val)}
}

pub fn from_port<A:Owned>(port: PortOne<A>) -> Future<A> {
    /*!
     * Create a future from a port
     *
     * The first time that the value is requested the task will block
     * waiting for the result to be received on the port.
     */

    let port = Cell(port);
    do from_fn || {
        let port = port.take().unwrap();
        match recv(port) {
            oneshot::send(data) => data
        }
    }
}

pub fn from_fn<A>(f: ~fn() -> A) -> Future<A> {
    /*!
     * Create a future from a function.
     *
     * The first time that the value is requested it will be retreived by
     * calling the function.  Note that this function is a local
     * function. It is not spawned into another task.
     */

    Future {state: Pending(f)}
}

pub fn spawn<A:Owned>(blk: ~fn() -> A) -> Future<A> {
    /*!
     * Create a future from a unique closure.
     *
     * The closure will be run in a new task and its result used as the
     * value of the future.
     */

    let (port, chan) = oneshot();

    let chan = Cell(chan);
    do task::spawn {
        let chan = chan.take();
        send_one(chan, blk());
    }

    return from_port(port);
}

#[allow(non_implicitly_copyable_typarams)]
#[cfg(test)]
mod test {

    use future::*;

    use core::comm::{oneshot, send_one};
    use core::task;

    #[test]
    fn test_from_value() {
        let f = from_value(~"snail");
        assert!(f.get() == ~"snail");
    }

    #[test]
    fn test_from_port() {
        let (po, ch) = oneshot();
        send_one(ch, ~"whale");
        let f = from_port(po);
        assert!(f.get() == ~"whale");
    }

    #[test]
    fn test_from_fn() {
        let f = from_fn(|| ~"brail");
        assert!(f.get() == ~"brail");
    }

    #[test]
    fn test_interface_get() {
        let f = from_value(~"fail");
        assert!(f.get() == ~"fail");
    }

    #[test]
    fn test_get_ref_method() {
        let f = from_value(22);
        assert!(*f.get_ref() == 22);
    }

    #[test]
    fn test_spawn() {
        let f = spawn(|| ~"bale");
        assert!(f.get() == ~"bale");
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_futurefail() {
        let f = spawn(|| fail!());
        let _x: ~str = f.get();
    }

    #[test]
    fn test_sendable_future() {
        let expected = ~"schlorf";
        let f = do spawn { copy expected };
        do task::spawn || {
            let actual = f.get();
            assert!(actual == expected);
        }
    }
}
