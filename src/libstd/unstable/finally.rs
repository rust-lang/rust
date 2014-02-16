// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
The Finally trait provides a method, `finally` on
stack closures that emulates Java-style try/finally blocks.

Using the `finally` method is sometimes convenient, but the type rules
prohibit any shared, mutable state between the "try" case and the
"finally" case. For advanced cases, the `try_finally` function can
also be used. See that function for more details.

# Example

```
use std::unstable::finally::Finally;
# fn always_run_this() {}

(|| {
    // ...
}).finally(|| {
    always_run_this();
})
```
*/

use ops::Drop;

#[cfg(test)] use task::failing;

pub trait Finally<T> {
    fn finally(&self, dtor: ||) -> T;
}

impl<'a,T> Finally<T> for 'a || -> T {
    fn finally(&self, dtor: ||) -> T {
        try_finally(&mut (), (),
                    |_, _| (*self)(),
                    |_| dtor())
    }
}

impl<T> Finally<T> for fn() -> T {
    fn finally(&self, dtor: ||) -> T {
        try_finally(&mut (), (),
                    |_, _| (*self)(),
                    |_| dtor())
    }
}

/**
 * The most general form of the `finally` functions. The function
 * `try_fn` will be invoked first; whether or not it fails, the
 * function `finally_fn` will be invoked next. The two parameters
 * `mutate` and `drop` are used to thread state through the two
 * closures. `mutate` is used for any shared, mutable state that both
 * closures require access to; `drop` is used for any state that the
 * `try_fn` requires ownership of.
 *
 * **WARNING:** While shared, mutable state between the try and finally
 * function is often necessary, one must be very careful; the `try`
 * function could have failed at any point, so the values of the shared
 * state may be inconsistent.
 *
 * # Example
 *
 * ```
 * use std::unstable::finally::try_finally;
 *
 * struct State<'a> { buffer: &'a mut [u8], len: uint }
 * # let mut buf = [];
 * let mut state = State { buffer: buf, len: 0 };
 * try_finally(
 *     &mut state, (),
 *     |state, ()| {
 *         // use state.buffer, state.len
 *     },
 *     |state| {
 *         // use state.buffer, state.len to cleanup
 *     })
 * ```
 */
pub fn try_finally<T,U,R>(mutate: &mut T,
                          drop: U,
                          try_fn: |&mut T, U| -> R,
                          finally_fn: |&mut T|)
                          -> R {
    let f = Finallyalizer {
        mutate: mutate,
        dtor: finally_fn,
    };
    try_fn(&mut *f.mutate, drop)
}

struct Finallyalizer<'a,A> {
    mutate: &'a mut A,
    dtor: 'a |&mut A|
}

#[unsafe_destructor]
impl<'a,A> Drop for Finallyalizer<'a,A> {
    #[inline]
    fn drop(&mut self) {
        (self.dtor)(self.mutate);
    }
}

#[test]
fn test_success() {
    let mut i = 0;
    try_finally(
        &mut i, (),
        |i, ()| {
            *i = 10;
        },
        |i| {
            assert!(!failing());
            assert_eq!(*i, 10);
            *i = 20;
        });
    assert_eq!(i, 20);
}

#[test]
#[should_fail]
fn test_fail() {
    let mut i = 0;
    try_finally(
        &mut i, (),
        |i, ()| {
            *i = 10;
            fail!();
        },
        |i| {
            assert!(failing());
            assert_eq!(*i, 10);
        })
}

#[test]
fn test_retval() {
    let closure: || -> int = || 10;
    let i = closure.finally(|| { });
    assert_eq!(i, 10);
}

#[test]
fn test_compact() {
    fn do_some_fallible_work() {}
    fn but_always_run_this_function() { }
    do_some_fallible_work.finally(
        but_always_run_this_function);
}

