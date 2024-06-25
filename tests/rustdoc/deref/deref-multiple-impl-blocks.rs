#![crate_name="foo"]

use std::ops::{Deref, DerefMut};

//@ has foo/struct.Vec.html
//@ count - '//h2[@id="deref-methods-Slice"]' 1
//@ count - '//div[@id="deref-methods-Slice-1"]' 1
//@ count - '//div[@id="deref-methods-Slice-1"][@class="impl-items"]' 1
//@ count - '//div[@id="deref-methods-Slice-1"]/div[@class="impl-items"]' 0
pub struct Vec;

pub struct Slice;

impl Deref for Vec {
    type Target = Slice;
    fn deref(&self) -> &Slice {
        &Slice
    }
}

impl DerefMut for Vec {
    fn deref_mut(&mut self) -> &mut Slice {
        &mut Slice
    }
}

impl Slice {
    pub fn sort_floats(&mut self) {
        todo!();
    }
}

impl Slice {
    pub fn sort(&mut self) {
        todo!();
    }
}

impl Slice {
    pub fn len(&self) {
        todo!();
    }
}
