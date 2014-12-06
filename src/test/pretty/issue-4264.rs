// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-compare-only
// pretty-mode:typed
// pp-exact:issue-4264.pp

// #4264 fixed-length vector types

pub fn foo(_: [isize; 3]) {}

pub fn bar() {
    const FOO: usize = 5us - 4us;
    let _: [(); FOO] = [()];

    let _ : [(); 1us] = [()];

    let _ = &([1is,2,3]) as *const _ as *const [isize; 3us];

    format!("test");
}

pub type Foo = [isize; 3us];

pub struct Bar {
    pub x: [isize; 3us]
}

pub struct TupleBar([isize; 4us]);

pub enum Baz {
    BazVariant([isize; 5us])
}

pub fn id<T>(x: T) -> T { x }

pub fn use_id() {
    let _ = id::<[isize; 3us]>([1,2,3]);
}


fn main() {}
