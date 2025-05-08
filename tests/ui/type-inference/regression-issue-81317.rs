// Regression test for #81317: type can no longer be infered as of 1.49
//
// The problem is that the xor operator and the index.into() call
// each have two candidate impls that could apply
// { S as BitXor<S>, S as BitXor<&'a S> } for xor and
// { T::I as Into<u64>, T::I as Into<S> } for index.into()
// previously inference was able to infer that the only valid combination was
// S as BitXor<S> and T::I as Into<S>
//
// after rust-lang/rust#73905 this is no longer infered
//
// the error message could be better e.g.
// when iv is unused or has an an explicitly specified type S
// there is currently the following help message
//
// error[E0284]: type annotations needed
//   --> src/main.rs:13:24
//    |
// 44 |     let iv = S ^ index.into();
//    |                -       ^^^^
//    |                |
//    |                type must be known at this point
//    |
//    = note: cannot satisfy `<S as BitXor<_>>::Output == _`
// help: try using a fully qualified path to specify the expected types
//    |
// 44 -     let iv = S ^ index.into();
// 44 +     let iv = S ^ <<T as P>::I as Into<T>>::into(index);
//
// this is better as it's actually sufficent to fix the problem,
// while just specifying the type of iv as currently suggested is insufficent
//
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
