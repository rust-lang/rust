// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(clippy::mem_discriminant_non_enum)]

use std::mem;

enum Foo {
    One(usize),
    Two(u8),
}

struct A(Foo);

fn main() {
    // bad
    mem::discriminant(&"hello");
    mem::discriminant(&&Some(2));
    mem::discriminant(&&None::<u8>);
    mem::discriminant(&&Foo::One(5));
    mem::discriminant(&&Foo::Two(5));
    mem::discriminant(&A(Foo::One(0)));

    let ro = &Some(3);
    let rro = &ro;
    mem::discriminant(&ro);
    mem::discriminant(rro);
    mem::discriminant(&rro);

    macro_rules! mem_discriminant_but_in_a_macro {
        ($param:expr) => {
            mem::discriminant($param)
        };
    }

    mem_discriminant_but_in_a_macro!(&rro);

    let rrrrro = &&&rro;
    mem::discriminant(&rrrrro);
    mem::discriminant(*rrrrro);

    // ok
    mem::discriminant(&Some(2));
    mem::discriminant(&None::<u8>);
    mem::discriminant(&Foo::One(5));
    mem::discriminant(&Foo::Two(5));
    mem::discriminant(ro);
    mem::discriminant(*rro);
    mem::discriminant(****rrrrro);
}
