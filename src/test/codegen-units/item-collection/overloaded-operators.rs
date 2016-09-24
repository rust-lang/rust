// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager

#![deny(dead_code)]
#![crate_type="lib"]

use std::ops::{Index, IndexMut, Add, Deref};

pub struct Indexable {
    data: [u8; 3]
}

impl Index<usize> for Indexable {
    type Output = u8;

    //~ TRANS_ITEM fn overloaded_operators::{{impl}}[0]::index[0]
    fn index(&self, index: usize) -> &Self::Output {
        if index >= 3 {
            &self.data[0]
        } else {
            &self.data[index]
        }
    }
}

impl IndexMut<usize> for Indexable {
    //~ TRANS_ITEM fn overloaded_operators::{{impl}}[1]::index_mut[0]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= 3 {
            &mut self.data[0]
        } else {
            &mut self.data[index]
        }
    }
}


//~ TRANS_ITEM fn overloaded_operators::{{impl}}[4]::eq[0]
//~ TRANS_ITEM fn overloaded_operators::{{impl}}[4]::ne[0]
#[derive(PartialEq)]
pub struct Equatable(u32);


impl Add<u32> for Equatable {
    type Output = u32;

    //~ TRANS_ITEM fn overloaded_operators::{{impl}}[2]::add[0]
    fn add(self, rhs: u32) -> u32 {
        self.0 + rhs
    }
}

impl Deref for Equatable {
    type Target = u32;

    //~ TRANS_ITEM fn overloaded_operators::{{impl}}[3]::deref[0]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

//~ TRANS_ITEM drop-glue i8
