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

Condition handling

Conditions are a utility used to deal with handling error conditions. The syntax
of a condition handler strikes a resemblance to try/catch blocks in other
languages, but condition handlers are *not* a form of exception handling in the
same manner.

A condition is declared through the `condition!` macro provided by the compiler:

```rust
condition! {
    pub my_error: int -> ~str;
}
 ```

This macro declares an inner module called `my_error` with one static variable,
`cond` that is a static `Condition` instance. To help understand what the other
parameters are used for, an example usage of this condition would be:

```rust
do my_error::cond.trap(|raised_int| {

    // the condition `my_error` was raised on, and the value it raised is stored
    // in `raised_int`. This closure must return a `~str` type (as specified in
    // the declaration of the condition
    if raised_int == 3 { ~"three" } else { ~"oh well" }

}).inside {

    // The condition handler above is installed for the duration of this block.
    // That handler will override any previous handler, but the previous handler
    // is restored when this block returns (handlers nest)
    //
    // If any code from this block (or code from another block) raises on the
    // condition, then the above handler will be invoked (so long as there's no
    // other nested handler).

    println(my_error::cond.raise(3)); // prints "three"
    println(my_error::cond.raise(4)); // prints "oh well"

}
 ```

Condition handling is useful in cases where propagating errors is either to
cumbersome or just not necessary in the first place. It should also be noted,
though, that if there is not handler installed when a condition is raised, then
the task invokes `fail!()` and will terminate.

## More Info

Condition handlers as an error strategy is well explained in the [conditions
tutorial](http://static.rust-lang.org/doc/master/tutorial-conditions.html),
along with comparing and contrasting it with other error handling strategies.

*/

use local_data;
use prelude::*;
use unstable::raw::Closure;

#[doc(hidden)]
pub struct Handler<T, U> {
    priv handle: Closure,
    priv prev: Option<@Handler<T, U>>,
}

/// This struct represents the state of a condition handler. It contains a key
/// into TLS which holds the currently install handler, along with the name of
/// the condition (useful for debugging).
///
/// This struct should never be created directly, but rather only through the
/// `condition!` macro provided to all libraries using libstd.
pub struct Condition<T, U> {
    /// Name of the condition handler
    name: &'static str,
    /// TLS key used to insert/remove values in TLS.
    key: local_data::Key<@Handler<T, U>>
}

impl<T, U> Condition<T, U> {
    /// Creates an object which binds the specified handler. This will also save
    /// the current handler *on creation* such that when the `Trap` is consumed,
    /// it knows which handler to restore.
    ///
    /// # Example
    ///
    /// ```rust
    /// condition! { my_error: int -> int; }
    ///
    /// let trap = my_error::cond.trap(|error| error + 3);
    ///
    /// // use `trap`'s inside method to register the handler and then run a
    /// // block of code with the handler registered
    /// ```
    pub fn trap<'a>(&'a self, h: &'a fn(T) -> U) -> Trap<'a, T, U> {
        let h: Closure = unsafe { ::cast::transmute(h) };
        let prev = local_data::get(self.key, |k| k.map(|&x| *x));
        let h = @Handler { handle: h, prev: prev };
        Trap { cond: self, handler: h }
    }

    /// Raises on this condition, invoking any handler if one has been
    /// registered, or failing the current task otherwise.
    ///
    /// While a condition handler is being run, the condition will have no
    /// handler listed, so a task failure will occur if the condition is
    /// re-raised during the handler.
    ///
    /// # Arguments
    ///
    /// * t - The argument to pass along to the condition handler.
    ///
    /// # Return value
    ///
    /// If a handler is found, its return value is returned, otherwise this
    /// function will not return.
    pub fn raise(&self, t: T) -> U {
        let msg = fmt!("Unhandled condition: %s: %?", self.name, t);
        self.raise_default(t, || fail!(msg.clone()))
    }

    /// Performs the same functionality as `raise`, except that when no handler
    /// is found the `default` argument is called instead of failing the task.
    pub fn raise_default(&self, t: T, default: &fn() -> U) -> U {
        match local_data::pop(self.key) {
            None => {
                debug!("Condition.raise: found no handler");
                default()
            }
            Some(handler) => {
                debug!("Condition.raise: found handler");
                match handler.prev {
                    None => {}
                    Some(hp) => local_data::set(self.key, hp)
                }
                let handle : &fn(T) -> U = unsafe {
                    ::cast::transmute(handler.handle)
                };
                let u = handle(t);
                local_data::set(self.key, handler);
                u
            }
        }
    }
}

/// A `Trap` is created when the `trap` method is invoked on a `Condition`, and
/// it is used to actually bind a handler into the TLS slot reserved for this
/// condition.
///
/// Normally this object is not dealt with directly, but rather it's directly
/// used after being returned from `trap`
struct Trap<'self, T, U> {
    priv cond: &'self Condition<T, U>,
    priv handler: @Handler<T, U>
}

impl<'self, T, U> Trap<'self, T, U> {
    /// Execute a block of code with this trap handler's exception handler
    /// registered.
    ///
    /// # Example
    ///
    /// ```rust
    /// condition! { my_error: int -> int; }
    ///
    /// let result = do my_error::cond.trap(|error| error + 3).inside {
    ///     my_error::cond.raise(4)
    /// };
    /// assert_eq!(result, 7);
    /// ```
    pub fn inside<V>(&self, inner: &'self fn() -> V) -> V {
        let _g = Guard { cond: self.cond };
        debug!("Trap: pushing handler to TLS");
        local_data::set(self.cond.key, self.handler);
        inner()
    }
}

#[doc(hidden)]
struct Guard<'self, T, U> {
    priv cond: &'self Condition<T, U>
}

#[unsafe_destructor]
impl<'self, T, U> Drop for Guard<'self, T, U> {
    fn drop(&mut self) {
        debug!("Guard: popping handler from TLS");
        let curr = local_data::pop(self.cond.key);
        match curr {
            None => {}
            Some(h) => match h.prev {
                None => {}
                Some(hp) => local_data::set(self.cond.key, hp)
            }
        }
    }
}

#[cfg(test)]
mod test {
    condition! {
        sadness: int -> int;
    }

    fn trouble(i: int) {
        debug!("trouble: raising condition");
        let j = sadness::cond.raise(i);
        debug!("trouble: handler recovered with %d", j);
    }

    fn nested_trap_test_inner() {
        let mut inner_trapped = false;

        do sadness::cond.trap(|_j| {
            debug!("nested_trap_test_inner: in handler");
            inner_trapped = true;
            0
        }).inside {
            debug!("nested_trap_test_inner: in protected block");
            trouble(1);
        }

        assert!(inner_trapped);
    }

    #[test]
    fn nested_trap_test_outer() {
        let mut outer_trapped = false;

        do sadness::cond.trap(|_j| {
            debug!("nested_trap_test_outer: in handler");
            outer_trapped = true; 0
        }).inside {
            debug!("nested_guard_test_outer: in protected block");
            nested_trap_test_inner();
            trouble(1);
        }

        assert!(outer_trapped);
    }

    fn nested_reraise_trap_test_inner() {
        let mut inner_trapped = false;

        do sadness::cond.trap(|_j| {
            debug!("nested_reraise_trap_test_inner: in handler");
            inner_trapped = true;
            let i = 10;
            debug!("nested_reraise_trap_test_inner: handler re-raising");
            sadness::cond.raise(i)
        }).inside {
            debug!("nested_reraise_trap_test_inner: in protected block");
            trouble(1);
        }

        assert!(inner_trapped);
    }

    #[test]
    fn nested_reraise_trap_test_outer() {
        let mut outer_trapped = false;

        do sadness::cond.trap(|_j| {
            debug!("nested_reraise_trap_test_outer: in handler");
            outer_trapped = true; 0
        }).inside {
            debug!("nested_reraise_trap_test_outer: in protected block");
            nested_reraise_trap_test_inner();
        }

        assert!(outer_trapped);
    }

    #[test]
    fn test_default() {
        let mut trapped = false;

        do sadness::cond.trap(|j| {
            debug!("test_default: in handler");
            sadness::cond.raise_default(j, || { trapped=true; 5 })
        }).inside {
            debug!("test_default: in protected block");
            trouble(1);
        }

        assert!(trapped);
    }

    // Issue #6009
    mod m {
        condition! {
            // #6009, #8215: should this truly need a `pub` for access from n?
            pub sadness: int -> int;
        }

        mod n {
            use super::sadness;

            #[test]
            fn test_conditions_are_public() {
                let mut trapped = false;
                do sadness::cond.trap(|_| {
                    trapped = true;
                    0
                }).inside {
                    sadness::cond.raise(0);
                }
                assert!(trapped);
            }
        }
    }
}
