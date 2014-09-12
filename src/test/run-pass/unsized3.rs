// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test structs with always-unsized fields.

use std::mem;
use std::raw;

struct Foo<T> {
    f: [T],
}

struct Bar {
    f1: uint,
    f2: [uint],
}

struct Baz {
    f1: uint,
    f2: str,
}

trait Tr {
    fn foo(&self) -> uint;
}

struct St {
    f: uint
}

impl Tr for St {
    fn foo(&self) -> uint {
        self.f
    }
}

struct Qux<'a> {
    f: Tr+'a
}

pub fn main() {
    let _: &Foo<f64>;
    let _: &Bar;
    let _: &Baz;

    let _: Box<Foo<i32>>;
    let _: Box<Bar>;
    let _: Box<Baz>;

    let _ = mem::size_of::<Box<Foo<u8>>>();
    let _ = mem::size_of::<Box<Bar>>();
    let _ = mem::size_of::<Box<Baz>>();

    unsafe {
        struct Foo_<T> {
            f: [T, ..3]
        }

        let data = box Foo_{f: [1i32, 2, 3] };
        let x: &Foo<i32> = mem::transmute(raw::Slice { len: 3, data: &*data });
        assert!(x.f.len() == 3);
        assert!(x.f[0] == 1);
        assert!(x.f[1] == 2);
        assert!(x.f[2] == 3);

        struct Baz_ {
            f1: uint,
            f2: [u8, ..5],
        }

        let data = box Baz_{ f1: 42, f2: ['a' as u8, 'b' as u8, 'c' as u8, 'd' as u8, 'e' as u8] };
        let x: &Baz = mem::transmute( raw::Slice { len: 5, data: &*data } );
        assert!(x.f1 == 42);
        let chs: Vec<char> = x.f2.chars().collect();
        assert!(chs.len() == 5);
        assert!(chs[0] == 'a');
        assert!(chs[1] == 'b');
        assert!(chs[2] == 'c');
        assert!(chs[3] == 'd');
        assert!(chs[4] == 'e');

        struct Qux_ {
            f: St
        }

        let obj: Box<St> = box St { f: 42 };
        let obj: &Tr = &*obj;
        let obj: raw::TraitObject = mem::transmute(&*obj);
        let data = box Qux_{ f: St { f: 234 } };
        let x: &Qux = mem::transmute(raw::TraitObject { vtable: obj.vtable,
                                                        data: mem::transmute(&*data) });
        assert!(x.f.foo() == 234);
    }
}
