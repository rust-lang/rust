// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_type_defaults)]

use std::ops::{Index};

trait Hierarchy {
    type Value;
    type ChildKey;
    type Children = Index<Self::ChildKey, Output=Hierarchy>;
    //~^ ERROR: the value of the associated type `ChildKey`
    //~^^ ERROR: the value of the associated type `Children`
    //~^^^ ERROR: the value of the associated type `Value`

    fn data(&self) -> Option<(Self::Value, Self::Children)>;
}

fn main() {}
