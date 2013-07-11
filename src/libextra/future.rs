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
 * ~~~ {.rust}
 * # fn fib(n: uint) -> uint {42};
 * # fn make_a_sandwich() {};
 * let mut delayed_fib = extra::future::spawn (|| fib(5000) );
 * make_a_sandwich();
 * println(fmt!("fib(5000) = %?", delayed_fib.get()))
 * ~~~
 */

#[allow(missing_doc)];


use std::cast;
use std::cell::Cell;
use std::comm::{PortOne, oneshot, send_one, recv_one};
use std::task;
use std::util::replace;

#[doc = "The future type"]
pub struct Future<A> {
    priv state: FutureState<A>,
}

// n.b. It should be possible to get rid of this.
// Add a test, though -- tjc
// FIXME(#2829) -- futures should not be copyable, because they close
// over ~fn's that have pipes and so forth within!
#[unsafe_destructor]
impl<A> Drop for Future<A> {
    fn drop(&self) {}
}

priv enum FutureState<A> {
    Pending(~fn() -> A),
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
    pub fn get_ref<'a>(&'a mut self) -> &'a A {
        /*!
        * Executes the future's closure and then returns a borrowed
        * pointer to the result.  The borrowed pointer lasts as long as
        * the future.
        */
        unsafe {
            {
                match self.state {
                    Forced(ref mut v) => { return cast::transmute(v); }
                    Evaluating => fail!("Recursive forcing of future!"),
                    Pending(_) => {}
                }
            }
            {
                let state = replace(&mut self.state, Evaluating);
                match state {
                    Forced(_) | Evaluating => fail!("Logic error."),
                    Pending(f) => {
                        self.state = Forced(f());
                        cast::transmute(self.get_ref())
                    }
                }
            }
        }
    }
}

pub fn from_value<A>(val: A) -> Future<A> {
    /*!
     * Create a future from a value.
     *
     * The value is immediately available and calling `get` later will
     * not block.
     */

    Future {state: Forced(val)}
}

pub fn from_port<A:Send>(port: PortOne<A>) -> Future<A> {
    /*!
     * Create a future from a port
     *
     * The first time that the value is requested the task will block
     * waiting for the result to be received on the port.
     */

    let port = Cell::new(port);
    do from_fn {
        recv_one(port.take())
    }
}

pub fn from_fn<A>(f: ~fn() -> A) -> Future<A> {
    /*!
     * Create a future from a function.
     *
     * The first time that the value is requested it will be retrieved by
     * calling the function.  Note that this function is a local
     * function. It is not spawned into another task.
     */

    Future {state: Pending(f)}
}

pub fn spawn<A:Send>(blk: ~fn() -> A) -> Future<A> {
    /*!
     * Create a future from a unique closure.
     *
     * The closure will be run in a new task and its result used as the
     * value of the future.
     */

    let (port, chan) = oneshot();

    let chan = Cell::new(chan);
    do task::spawn {
        let chan = chan.take();
        send_one(chan, blk());
    }

    return from_port(port);
}

#[cfg(test)]
mod test {
    use future::*;

    use std::cell::Cell;
    use std::comm::{oneshot, send_one};
    use std::task;

    #[test]
    fn test_from_value() {
        let mut f = from_value(~"snail");
        assert_eq!(f.get(), ~"snail");
    }

    #[test]
    fn test_from_port() {
        let (po, ch) = oneshot();
        send_one(ch, ~"whale");
        let mut f = from_port(po);
        assert_eq!(f.get(), ~"whale");
    }

    #[test]
    fn test_from_fn() {
        let mut f = from_fn(|| ~"brail");
        assert_eq!(f.get(), ~"brail");
    }

    #[test]
    fn test_interface_get() {
        let mut f = from_value(~"fail");
        assert_eq!(f.get(), ~"fail");
    }

    #[test]
    fn test_get_ref_method() {
        let mut f = from_value(22);
        assert_eq!(*f.get_ref(), 22);
    }

    #[test]
    fn test_spawn() {
        let mut f = spawn(|| ~"bale");
        assert_eq!(f.get(), ~"bale");
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_futurefail() {
        let mut f = spawn(|| fail!());
        let _x: ~str = f.get();
    }

    #[test]
    fn test_sendable_future() {
        let expected = "schlorf";
        let f = Cell::new(do spawn { expected });
        do task::spawn {
            let mut f = f.take();
            let actual = f.get();
            assert_eq!(actual, expected);
        }
    }
}
