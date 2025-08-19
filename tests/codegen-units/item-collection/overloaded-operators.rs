//@ compile-flags:-Clink-dead-code

#![deny(dead_code)]
#![crate_type = "lib"]

use std::ops::{Add, Deref, Index, IndexMut};

pub struct Indexable {
    data: [u8; 3],
}

impl Index<usize> for Indexable {
    type Output = u8;

    //~ MONO_ITEM fn <Indexable as std::ops::Index<usize>>::index
    fn index(&self, index: usize) -> &Self::Output {
        if index >= 3 { &self.data[0] } else { &self.data[index] }
    }
}

impl IndexMut<usize> for Indexable {
    //~ MONO_ITEM fn <Indexable as std::ops::IndexMut<usize>>::index_mut
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= 3 { &mut self.data[0] } else { &mut self.data[index] }
    }
}

//~ MONO_ITEM fn <Equatable as std::cmp::PartialEq>::eq
//~ MONO_ITEM fn <Equatable as std::cmp::PartialEq>::ne
#[derive(PartialEq)]
pub struct Equatable(u32);

impl Add<u32> for Equatable {
    type Output = u32;

    //~ MONO_ITEM fn <Equatable as std::ops::Add<u32>>::add
    fn add(self, rhs: u32) -> u32 {
        self.0 + rhs
    }
}

impl Deref for Equatable {
    type Target = u32;

    //~ MONO_ITEM fn <Equatable as std::ops::Deref>::deref
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
