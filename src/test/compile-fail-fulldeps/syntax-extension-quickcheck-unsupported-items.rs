// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage1

#![feature(phase)]

extern crate quickcheck;
#[phase(syntax)] extern crate quickcheck_macros;

// Tests to make sure that `#[quickcheck]` will produce a compile error when
// applied to anything other than a function or static item.

#[quickcheck] //~ ERROR #[quickcheck] only supported on statics and functions
pub mod foo {
    pub fn bar() {}
}

#[quickcheck] //~ ERROR #[quickcheck] only supported on statics and functions
extern "C" {
    pub fn baz(name: *u8) -> u8;
}

#[quickcheck] //~ ERROR #[quickcheck] only supported on statics and functions
type Blah = [uint, ..8];

#[quickcheck] //~ ERROR #[quickcheck] only supported on statics and functions
enum Maybe<T> {
    Just(T),
    Nothing
}

#[quickcheck] //~ ERROR #[quickcheck] only supported on statics and functions
pub struct Stuff {
    a: Maybe<&'static int>,
    b: Option<u32>
}

#[quickcheck] //~ ERROR #[quickcheck] only supported on statics and functions
pub trait Thing<T> {
    fn do_it(&self) -> T;
}

#[quickcheck] //~ ERROR #[quickcheck] only supported on statics and functions
impl Thing<u32> for Stuff {
    fn do_it(&self) -> u32 {
        return self.b.unwrap_or(42);
    }
}

static CONSTANT: int = 2;

fn main () {
    let s = Stuff {a: Just(&CONSTANT), b: None};
    println!("{}", s.do_it());
}
