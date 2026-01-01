//@ compile-flags: -Copt-level=3 --crate-type=rlib
//@ build-pass

// A regression test for #149081. The environment of `size` and `align`
// currently means that the item bound of`T::Assoc` doesn't hold. This can
// result in normalization failures and ICE during MIR optimizations.
//
// This will no longer be an issue once #149283 is implemented.

pub fn align<T: WithAssoc<Assoc = U>, U>() -> usize {
    std::mem::align_of::<Wrapper<T>>()
}

pub fn size<T: WithAssoc<Assoc = U>, U>() -> usize {
    std::mem::size_of::<Wrapper<T>>()
}

pub struct Wrapper<T: WithAssoc> {
    assoc2: <T::Assoc as WithAssoc>::Assoc,
    value: T,
}

pub trait WithAssoc {
    type Assoc: WithAssoc;
}
