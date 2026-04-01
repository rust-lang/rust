// Regression test for https://github.com/rust-lang/rust/issues/56445#issuecomment-629426939
//@ check-pass

#![crate_type = "lib"]

use std::marker::PhantomData;

pub struct S<'a> {
    pub m1: PhantomData<&'a u8>,
    pub m2: [u8; S::size()],
}

impl<'a> S<'a>
{
    pub const fn size() -> usize { 1 }

    pub fn new() -> Self
    {
        Self
        {
            m1: PhantomData,
            m2: [0; Self::size()],
        }
    }
}
