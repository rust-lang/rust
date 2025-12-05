// Test that adding an impl to a trait `Foo` DOES affect functions
// that only use `Bar` if they have methods in common.

//@ incremental
//@ compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_imports)]

fn main() { }

pub trait Foo: Sized {
    fn method(self) { }
}

pub trait Bar: Sized {
    fn method(self) { }
}

mod x {
    use crate::{Foo, Bar};

    #[rustc_if_this_changed]
    impl Foo for u32 { }

    impl Bar for char { }
}

mod y {
    use crate::{Foo, Bar};

    #[rustc_then_this_would_need(typeck)] //~ ERROR OK
    pub fn with_char() {
        char::method('a');
    }
}

mod z {
    use crate::y;

    #[rustc_then_this_would_need(typeck)] //~ ERROR no path
    pub fn z() {
        y::with_char();
    }
}
