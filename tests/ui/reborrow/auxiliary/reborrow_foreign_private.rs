#![allow(dead_code)]

use std::marker::PhantomData;

#[derive(Clone, Copy)]
pub struct ForeignRef<'a> {
    value: &'a i32,
}

// SAFETY invariant: the pointer is valid as `&'a i32`.
#[derive(Clone, Copy)]
pub struct ForeignPtrRef<'a>((*const i32, PhantomData<&'a ()>));

impl<'a> ForeignPtrRef<'a> {
    pub fn new(r: &'a i32) -> Self {
        ForeignPtrRef((r, PhantomData))
    }

    pub fn to_ref(self) -> &'a i32 {
        unsafe { &*self.0.0 }
    }
}
