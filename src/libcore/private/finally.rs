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
use task::{spawn, failing};

pub trait Finally<T> {
    fn finally(&self, +dtor: &fn()) -> T;
}

impl<T> &fn() -> T: Finally<T> {
    // XXX: Should not require a mode here
    fn finally(&self, +dtor: &fn()) -> T {
        let _d = Finallyalizer {
            dtor: dtor
        };

        (*self)()
    }
}

struct Finallyalizer {
    dtor: &fn()
}

impl Finallyalizer: Drop {
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
        assert !failing();
        assert i == 10;
        i = 20;
    }
    assert i == 20;
}

#[test]
#[ignore(cfg(windows))]
#[should_fail]
fn test_fail() {
    let mut i = 0;
    do (|| {
        i = 10;
        fail;
    }).finally {
        assert failing();
        assert i == 10;
    }
}

#[test]
fn test_retval() {
    let i = do (fn&() -> int {
        10
    }).finally { };
    assert i == 10;
}

#[test]
fn test_compact() {
    // XXX Should be able to use a fn item instead
    // of a closure for do_some_fallible_work,
    // but it's a type error.
    let do_some_fallible_work: &fn() = || { };
    fn but_always_run_this_function() { }
    do_some_fallible_work.finally(
        but_always_run_this_function);
}