// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// run-pass

#![feature(const_fn_union)]

union Transmute<T: Copy, U: Copy> {
    t: T,
    u: U,
}

trait Bar {
    fn bar(&self) -> u32;
}

struct Foo {
    foo: u32,
    bar: bool,
}

impl Bar for Foo {
    fn bar(&self) -> u32 {
        self.foo
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        assert!(!self.bar);
        self.bar = true;
        println!("dropping Foo");
    }
}

#[derive(Copy, Clone)]
struct Fat<'a>(&'a Foo, &'static VTable);

struct VTable {
    drop: Option<for<'a> fn(&'a mut Foo)>,
    size: usize,
    align: usize,
    bar: for<'a> fn(&'a Foo) -> u32,
}

const FOO: &Bar = &Foo { foo: 128, bar: false };
const G: Fat = unsafe { Transmute { t: FOO }.u };
const F: Option<for<'a> fn(&'a mut Foo)> = G.1.drop;
const H: for<'a> fn(&'a Foo) -> u32 = G.1.bar;

fn main() {
    let mut foo = Foo { foo: 99, bar: false };
    (F.unwrap())(&mut foo);
    std::mem::forget(foo); // already ran the drop impl
    assert_eq!(H(&Foo { foo: 42, bar: false }), 42);
}
