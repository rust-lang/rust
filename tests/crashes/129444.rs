//@ known-bug: rust-lang/rust#129444

//@ compile-flags: -Znext-solver=coherence

trait Trait {
    type Assoc;
}

struct W<T: Trait>(*mut T);
impl<T: ?Trait> Trait for W<W<W<T>>> {}

trait NoOverlap {}
impl<T: Trait<W<T>>> NoOverlap for T {}

impl<T: Trait<Assoc = u32>> NoOverlap for W<T> {}
