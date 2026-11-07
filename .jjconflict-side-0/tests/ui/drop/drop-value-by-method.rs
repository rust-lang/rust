//! Regression test for <https://github.com/rust-lang/rust/issues/3220>.
//! Initially a pretty-printer bug, which omitted parentheses around `move`
//! keyword in lhs (`(move z).f()`), now this behaviour is untestable as
//! plain `move` outside a closure doesn't exist.
//!
//! Now appears to test that drop works correctly when value is moved by method.
//@ run-pass

#![allow(dead_code)]
#![allow(non_camel_case_types)]

struct thing { x: isize, }

impl Drop for thing {
    fn drop(&mut self) {}
}

fn thing() -> thing {
    thing {
        x: 0
    }
}

impl thing {
    pub fn f(self) {}
}

pub fn main() {
    let z = thing();
    (z).f();
}
