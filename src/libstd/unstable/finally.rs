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

# Example

 ```
(|| {
    ...
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

macro_rules! finally_fn {
    ($fnty:ty) => {
        impl<T> Finally<T> for $fnty {
            fn finally(&self, dtor: ||) -> T {
                let _d = Finallyalizer {
                    dtor: dtor
                };
                (*self)()
            }
        }
    }
}

impl<'a,T> Finally<T> for 'a || -> T {
    fn finally(&self, dtor: ||) -> T {
        let _d = Finallyalizer {
            dtor: dtor
        };

        (*self)()
    }
}

finally_fn!(extern "Rust" fn() -> T)

struct Finallyalizer<'a> {
    dtor: 'a ||
}

#[unsafe_destructor]
impl<'a> Drop for Finallyalizer<'a> {
    #[inline]
    fn drop(&mut self) {
        (self.dtor)();
    }
}

#[test]
fn test_success() {
    let mut i = 0;
    (|| {
        i = 10;
    }).finally(|| {
        assert!(!failing());
        assert_eq!(i, 10);
        i = 20;
    });
    assert_eq!(i, 20);
}

#[test]
#[should_fail]
fn test_fail() {
    let mut i = 0;
    (|| {
        i = 10;
        fail!();
    }).finally(|| {
        assert!(failing());
        assert_eq!(i, 10);
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

