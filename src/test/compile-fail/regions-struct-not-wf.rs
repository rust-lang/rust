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

// revisions:lexical nll

#![allow(dead_code)]
#![cfg_attr(nll, feature(nll))]

struct Ref<'a, T> {
    field: &'a T
        //[lexical]~^ ERROR the parameter type `T` may not live long enough
        //[nll]~^^ ERROR the parameter type `T` may not live long enough
}

struct RefOk<'a, T:'a> {
    field: &'a T
}

struct RefIndirect<'a, T> {
    field: RefOk<'a, T>
        //[lexical]~^ ERROR the parameter type `T` may not live long enough
        //[nll]~^^ ERROR the parameter type `T` may not live long enough
}

struct DoubleRef<'a, 'b, T> {
    field: &'a &'b T
        //[lexical]~^ ERROR reference has a longer lifetime than the data it references
        //[nll]~^^ ERROR reference has a longer lifetime than the data it references
}

fn main() { }
