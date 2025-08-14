// Test that when a trait impl changes, fns whose body uses that trait
// must also be recompiled.

//@ incremental
//@ compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(warnings)]

fn main() { }

pub trait Foo: Sized {
    type T;
    fn method(self) { }
}

mod x {
    use crate::Foo;

    #[rustc_if_this_changed]
    impl Foo for char { type T = char; }

    impl Foo for u32 { type T = u32; }
}

mod y {
    use crate::Foo;

    #[rustc_then_this_would_need(typeck)] //~ ERROR OK
    pub fn use_char_assoc() {
        // Careful here: in the representation, <char as Foo>::T gets
        // normalized away, so at a certain point we had no edge to
        // codegen.  (But now codegen just depends on typeck.)
        let x: <char as Foo>::T = 'a';
    }

    pub fn take_foo<T:Foo>(t: T) { }
}
