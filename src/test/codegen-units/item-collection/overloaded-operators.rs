// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![crate_type="lib"]

use std::ops::{Index, IndexMut, Add, Deref};

pub struct Indexable {
    data: [u8; 3]
}

impl Index<usize> for Indexable {
    type Output = u8;

    //~ MONO_ITEM fn overloaded_operators::{{impl}}[0]::index[0]
    fn index(&self, index: usize) -> &Self::Output {
        if index >= 3 {
            &self.data[0]
        } else {
            &self.data[index]
        }
    }
}

impl IndexMut<usize> for Indexable {
    //~ MONO_ITEM fn overloaded_operators::{{impl}}[1]::index_mut[0]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= 3 {
            &mut self.data[0]
        } else {
            &mut self.data[index]
        }
    }
}


//~ MONO_ITEM fn overloaded_operators::{{impl}}[4]::eq[0]
//~ MONO_ITEM fn overloaded_operators::{{impl}}[4]::ne[0]
#[derive(PartialEq)]
pub struct Equatable(u32);


impl Add<u32> for Equatable {
    type Output = u32;

    //~ MONO_ITEM fn overloaded_operators::{{impl}}[2]::add[0]
    fn add(self, rhs: u32) -> u32 {
        self.0 + rhs
    }
}

impl Deref for Equatable {
    type Target = u32;

    //~ MONO_ITEM fn overloaded_operators::{{impl}}[3]::deref[0]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
