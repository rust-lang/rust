// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo {
    Bar(u32),
    Baz,
    Quux(u64, u16)
}

static X: Foo = Baz;

pub fn main() {
    match X {
        Baz => {}
        _ => fail!()
    }
    match Y {
        Bar(s) => assert!(s == 2654435769),
        _ => fail!()
    }
    match Z {
        Quux(d,h) => {
            assert!((d == 0x123456789abcdef0));
            assert!((h == 0x1234));
        }
        _ => fail!()
    }
}

static Y: Foo = Bar(2654435769);
static Z: Foo = Quux(0x123456789abcdef0, 0x1234);
