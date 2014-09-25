// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Various examples of structs whose fields are not well-formed.

#![allow(dead_code)]

enum Ref1<'a, T> { //~ ERROR the parameter type `T` may not live long enough
    Ref1Variant1(&'a T)
}

enum Ref2<'a, T> { //~ ERROR the parameter type `T` may not live long enough
    Ref2Variant1,
    Ref2Variant2(int, &'a T),
}

enum RefOk<'a, T:'a> {
    RefOkVariant1(&'a T)
}

enum RefIndirect<'a, T> { //~ ERROR the parameter type `T` may not live long enough
    RefIndirectVariant1(int, RefOk<'a,T>)
}

enum RefDouble<'a, 'b, T> { //~ ERROR reference has a longer lifetime than the data
    RefDoubleVariant1(&'a &'b T)
}

fn main() { }
