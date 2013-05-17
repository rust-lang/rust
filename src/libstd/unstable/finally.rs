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

~~~
do || {
    ...
}.finally {
    alway_run_this();
}
~~~
*/

use ops::Drop;

#[cfg(test)] use task::{failing, spawn};

pub trait Finally<T> {
    fn finally(&self, dtor: &fn()) -> T;
}

impl<'self,T> Finally<T> for &'self fn() -> T {
    fn finally(&self, dtor: &fn()) -> T {
        let _d = Finallyalizer {
            dtor: dtor
        };

        (*self)()
    }
}

impl<T> Finally<T> for ~fn() -> T {
    fn finally(&self, dtor: &fn()) -> T {
        let _d = Finallyalizer {
            dtor: dtor
        };

        (*self)()
    }
}

impl<T> Finally<T> for @fn() -> T {
    fn finally(&self, dtor: &fn()) -> T {
        let _d = Finallyalizer {
            dtor: dtor
        };

        (*self)()
    }
}

struct Finallyalizer<'self> {
    dtor: &'self fn()
}

#[unsafe_destructor]
impl<'self> Drop for Finallyalizer<'self> {
    fn finalize(&self) {
        (self.dtor)();
    }
}

#[test]
fn test_success() {
    let mut i = 0;
    do (|| {
        i = 10;
    }).finally {
        assert!(!failing());
        assert_eq!(i, 10);
        i = 20;
    }
    assert_eq!(i, 20);
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
fn test_fail() {
    let mut i = 0;
    do (|| {
        i = 10;
        fail!();
    }).finally {
        assert!(failing());
        assert_eq!(i, 10);
    }
}

#[test]
fn test_retval() {
    let closure: &fn() -> int = || 10;
    let i = do closure.finally { };
    assert_eq!(i, 10);
}

#[test]
fn test_compact() {
    // FIXME #4727: Should be able to use a fn item instead
    // of a closure for do_some_fallible_work,
    // but it's a type error.
    let do_some_fallible_work: &fn() = || { };
    fn but_always_run_this_function() { }
    do_some_fallible_work.finally(
        but_always_run_this_function);
}

#[test]
fn test_owned() {
    fn spawn_with_finalizer(f: ~fn()) {
        do spawn { do f.finally { } }
    }
    let owned: ~fn() = || { };
    spawn_with_finalizer(owned);
}

#[test]
fn test_managed() {
    let i = @mut 10;
    let managed: @fn() -> int = || {
        let r = *i;
        *i += 10;
        r
    };
    assert_eq!(do managed.finally {}, 10);
    assert_eq!(*i, 20);
}