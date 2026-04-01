// Computing the ambiguity causes for the overlap ended up
// causing an exponential blowup when recursing into the normalization
// goals for `<Box<?t> as RecursiveSuper>::Assoc`. This test
// takes multiple minutes when doing so and less than a second
// otherwise.

//@ compile-flags: -Znext-solver=coherence

trait RecursiveSuper:
    Super<
        A0 = Self::Assoc,
        A1 = Self::Assoc,
        A2 = Self::Assoc,
        A3 = Self::Assoc,
        A4 = Self::Assoc,
        A5 = Self::Assoc,
        A6 = Self::Assoc,
        A7 = Self::Assoc,
        A8 = Self::Assoc,
        A9 = Self::Assoc,
        A10 = Self::Assoc,
        A11 = Self::Assoc,
        A12 = Self::Assoc,
        A13 = Self::Assoc,
        A14 = Self::Assoc,
        A15 = Self::Assoc,
    >
{
    type Assoc;
}

trait Super {
    type A0;
    type A1;
    type A2;
    type A3;
    type A4;
    type A5;
    type A6;
    type A7;
    type A8;
    type A9;
    type A10;
    type A11;
    type A12;
    type A13;
    type A14;
    type A15;
}

trait Overlap {}
impl<T: RecursiveSuper> Overlap for T {}
impl<T> Overlap for Box<T> {}
//~^ ERROR conflicting implementations of trait `Overlap` for type `Box<_>`

fn main() {}
