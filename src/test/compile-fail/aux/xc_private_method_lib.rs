// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]

pub struct Struct {
    pub x: isize
}

impl Struct {
    fn static_meth_struct() -> Struct {
        Struct { x: 1 }
    }

    fn meth_struct(&self) -> isize {
        self.x
    }
}

pub enum Enum {
    Variant1(isize),
    Variant2(isize)
}

impl Enum {
    fn static_meth_enum() -> Enum {
        Enum::Variant2(10)
    }

    fn meth_enum(&self) -> isize {
        match *self {
            Enum::Variant1(x) |
            Enum::Variant2(x) => x
        }
    }
}
