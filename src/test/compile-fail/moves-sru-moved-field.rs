// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


type Noncopyable = proc();

struct Foo {
    copied: int,
    moved: Box<int>,
    noncopyable: Noncopyable
}

fn test0(f: Foo, g: Noncopyable, h: Noncopyable) {
    // just copy implicitly copyable fields from `f`, no moves:
    let _b = Foo {moved: box 1, noncopyable: g, ..f};
    let _c = Foo {moved: box 2, noncopyable: h, ..f};
}

fn test1(f: Foo, g: Noncopyable, h: Noncopyable) {
    // copying move-by-default fields from `f`, so move:
    let _b = Foo {noncopyable: g, ..f};
    let _c = Foo {noncopyable: h, ..f}; //~ ERROR use of partially moved value: `f`
}

fn test2(f: Foo, g: Noncopyable) {
    // move non-copyable field
    let _b = Foo {copied: 22, moved: box 23, ..f};
    let _c = Foo {noncopyable: g, ..f}; //~ ERROR use of partially moved value: `f`
}

fn main() {}
