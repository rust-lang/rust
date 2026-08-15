//! Test that Debug::fmt is called exactly once during formatting.
//!
//! This is a regression test for PR https://github.com/rust-lang/rust/pull/10715

//@ run-pass
//@ needs-threads

use std::cell::Cell;
use std::{fmt, thread};

struct Foo(Cell<isize>);

impl fmt::Debug for Foo {
    fn fmt(&self, _fmt: &mut fmt::Formatter) -> fmt::Result {
        let Foo(ref f) = *self;
        assert_eq!(f.get(), 0);
        f.set(1);
        Ok(())
    }
}

pub fn main() {
    thread::spawn(move || {
        let mut f = Foo(Cell::new(0));
        println!("{:?}", f);
        let Foo(ref mut f) = f;
        assert_eq!(f.get(), 1);
    })
    .join()
    .ok()
    .unwrap();
}
