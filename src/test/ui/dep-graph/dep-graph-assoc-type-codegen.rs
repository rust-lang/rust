// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that when a trait impl changes, fns whose body uses that trait
// must also be recompiled.

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(warnings)]

fn main() { }

pub trait Foo: Sized {
    type T;
    fn method(self) { }
}

mod x {
    use Foo;

    #[rustc_if_this_changed]
    impl Foo for char { type T = char; }

    impl Foo for u32 { type T = u32; }
}

mod y {
    use Foo;

    #[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
    pub fn use_char_assoc() {
        // Careful here: in the representation, <char as Foo>::T gets
        // normalized away, so at a certain point we had no edge to
        // codegen.  (But now codegen just depends on typeck.)
        let x: <char as Foo>::T = 'a';
    }

    pub fn take_foo<T:Foo>(t: T) { }
}
