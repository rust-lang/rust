// Regression test for the original ICE in issue #163320.
//@ compile-flags: -Znext-solver=globally --crate-type=lib
//@ check-fail

trait HasStorage {
    type Storage;
}

struct Bits;

trait TryFromBits {
    type Bits;
}

trait ReadEndian: TryFromBits {}

struct Zeros<const N: u32>;

impl<const N: u32> TryFromBits for Zeros<N> {
    type Bits = Bits;
}

impl<const N: u32> ReadEndian for Zeros<N>
where
    <Self::Bits as HasStorage>::Storage::Bits: Sized,
    //~^ ERROR ambiguous associated type
{}
