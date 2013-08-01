// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Condition handling */

#[allow(missing_doc)];

use local_data;
use prelude::*;

// helper for transmutation, shown below.
type RustClosure = (int, int);

pub struct Handler<T, U> {
    handle: RustClosure,
    prev: Option<@Handler<T, U>>,
}

pub struct Condition<T, U> {
    name: &'static str,
    key: local_data::Key<@Handler<T, U>>
}

impl<T, U> Condition<T, U> {
    pub fn trap<'a>(&'a self, h: &'a fn(T) -> U) -> Trap<'a, T, U> {
        unsafe {
            let p : *RustClosure = ::cast::transmute(&h);
            let prev = local_data::get(self.key, |k| k.map(|&x| *x));
            let h = @Handler { handle: *p, prev: prev };
            Trap { cond: self, handler: h }
        }
    }

    pub fn raise(&self, t: T) -> U {
        let msg = fmt!("Unhandled condition: %s: %?", self.name, t);
        self.raise_default(t, || fail!(msg.clone()))
    }

    pub fn raise_default(&self, t: T, default: &fn() -> U) -> U {
        unsafe {
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
                    let handle : &fn(T) -> U =
                        ::cast::transmute(handler.handle);
                    let u = handle(t);
                    local_data::set(self.key, handler);
                    u
                }
            }
        }
    }
}

struct Trap<'self, T, U> {
    cond: &'self Condition<T, U>,
    handler: @Handler<T, U>
}

impl<'self, T, U> Trap<'self, T, U> {
    pub fn inside<V>(&self, inner: &'self fn() -> V) -> V {
        let _g = Guard { cond: self.cond };
        debug!("Trap: pushing handler to TLS");
        local_data::set(self.cond.key, self.handler);
        inner()
    }
}

struct Guard<'self, T, U> {
    cond: &'self Condition<T, U>
}

#[unsafe_destructor]
impl<'self, T, U> Drop for Guard<'self, T, U> {
    fn drop(&self) {
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
            sadness: int -> int;
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
