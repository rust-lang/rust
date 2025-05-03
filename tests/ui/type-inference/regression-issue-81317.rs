// Regression test for #81317: type can no longer be infered as of 1.49
//@ check-fail

use std::ops::BitXor;

pub struct S;

pub trait P {
    type I: Into<u64> + Into<S>;
}

pub fn decrypt_portion<T: P>(index: T::I) {
    let iv = S ^ index.into();
    //~^ ERROR type annotations needed
    &iv.to_bytes_be();
}

impl S {
    fn to_bytes_be(&self) -> &[u8] {
        &[]
    }
}

impl BitXor for S {
    type Output = S;

    fn bitxor(self, _rhs: Self) -> Self::Output {
        S
    }
}

impl<'a> BitXor<&'a S> for S {
    type Output = S;

    fn bitxor(self, _rhs: &'a S) -> Self::Output {
        S
    }
}

fn main() {}
