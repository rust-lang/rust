// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(immovable_types)]

use std::marker::{Move, Immovable};

fn mov<T: ?Move>(_: T) {}

pub fn main() {
    // Illegal: Moving out of children of observed values
    {
        let a = (Immovable, true);
        &a;
        mov(a.0); //~ ERROR cannot move value whose address is observed
    }

    // Illegal: Moving out of observed values
    {
        let a = (Immovable, true);
        &a;
        mov(a); //~ ERROR cannot move value whose address is observed
    }

    // Illegal: Moving out of values with observed children
    {
        let a = (Immovable, true);
        &a.0;
        mov(a); //~ ERROR cannot move value whose address is observed
    }

    // Illegal: Moving out of observed children
    {
        let a = (Immovable, true);
        &a.0;
        mov(a.0); //~ ERROR cannot move value whose address is observed
    }

    // Legal: Moving out of movable children of observed values
    {
        let a = (Immovable, true);
        &a;
        mov(a.1);
    }

    // Legal: Moving out of siblings of observed values
    {
        let a = (Immovable, Immovable);
        &a.0;
        mov(a.1);
    }

    // Legal: Moving out of value with an observed movable children
    {
        let a = (Immovable, true);
        &a.1;
        mov(a);
    }
}
