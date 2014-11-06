// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo {
    Bar(int),
    Baz,
}

enum Other {
    Other1(Foo),
    Other2(Foo, Foo),
}

fn main() {
    match Foo::Baz {
        ::Foo::Bar(3) => panic!(),
        ::Foo::Bar(_) if false => panic!(),
        ::Foo::Bar(..) if false => panic!(),
        ::Foo::Bar(_n) => panic!(),
        ::Foo::Baz => {}
    }
    match Foo::Bar(3) {
        ::Foo::Bar(3) => {}
        ::Foo::Bar(_) if false => panic!(),
        ::Foo::Bar(..) if false => panic!(),
        ::Foo::Bar(_n) => panic!(),
        ::Foo::Baz => panic!(),
    }
    match Foo::Bar(4) {
        ::Foo::Bar(3) => panic!(),
        ::Foo::Bar(_) if false => panic!(),
        ::Foo::Bar(..) if false => panic!(),
        ::Foo::Bar(n) => assert_eq!(n, 4),
        ::Foo::Baz => panic!(),
    }

    match Other::Other1(Foo::Baz) {
        ::Other::Other1(::Foo::Baz) => {}
        ::Other::Other1(::Foo::Bar(_)) => {}
        ::Other::Other2(::Foo::Baz, ::Foo::Bar(_)) => {}
        ::Other::Other2(::Foo::Bar(..), ::Foo::Baz) => {}
        ::Other::Other2(..) => {}
    }
}
