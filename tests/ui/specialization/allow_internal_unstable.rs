// check-pass
// test for #119950
// compile-flags: --crate-type lib

#![allow(internal_features)]
#![feature(allow_internal_unstable)]

#[allow_internal_unstable(min_specialization)]
macro_rules! test {
    () => {
        struct T<U>(U);
        trait Tr {}
        impl<U> Tr for T<U> {}
        impl Tr for T<u8> {}
    }
}

test! {}
