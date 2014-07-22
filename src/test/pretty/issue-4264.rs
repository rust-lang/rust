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

pub fn foo(_: [int, ..3]) {}

pub fn bar() {
    static FOO: uint = 5u - 4u;
    let _: [(), ..FOO] = [()];

    let _ : [(), ..1u] = [()];

    let _ = &([1i,2,3]) as *const _ as *const [int, ..3u];

    format!("test");
}

pub type Foo = [int, ..3u];

pub struct Bar {
    pub x: [int, ..3u]
}

pub struct TupleBar([int, ..4u]);

pub enum Baz {
    BazVariant([int, ..5u])
}

pub fn id<T>(x: T) -> T { x }

pub fn use_id() {
    let _ = id::<[int, ..3u]>([1,2,3]);
}


fn main() {}
