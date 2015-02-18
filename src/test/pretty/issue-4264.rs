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

pub fn foo(_: [i32; 3]) {}

pub fn bar() {
    const FOO: usize = 5 - 4;
    let _: [(); FOO] = [()];

    let _ : [(); 1usize] = [()];

    let _ = &([1,2,3]) as *const _ as *const [i32; 3usize];

    format!("test");
}

pub type Foo = [i32; 3];

pub struct Bar {
    pub x: [i32; 3]
}

pub struct TupleBar([i32; 4]);

pub enum Baz {
    BazVariant([i32; 5])
}

pub fn id<T>(x: T) -> T { x }

pub fn use_id() {
    let _ = id::<[i32; 3]>([1,2,3]);
}


fn main() {}
