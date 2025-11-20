// Regression test for #149081. Check that an attempt to evaluate size_of / align_of intrinsics
// during an optimization, when type parameter is not monomorphic enough, doesn't lead to a
// compilation failure.
//
//@compile-flags: --crate-type=lib -O
//@build-pass

pub fn size_align_of<T: WithAssoc<Assoc = U>, U>() -> (usize, usize) {
    let a = const { std::mem::size_of::<Wrapper<T>>() };
    let b = const { std::mem::align_of::<Wrapper<T>>() };
    (a, b)
}

pub struct Wrapper<T: WithAssoc> {
    pub assoc2: <T::Assoc as WithAssoc>::Assoc,
    pub value: T,
}

pub trait WithAssoc {
    type Assoc: WithAssoc;
}
