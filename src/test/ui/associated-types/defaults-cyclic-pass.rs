// check-pass

#![feature(associated_type_defaults)]

// Having a cycle in assoc. type defaults is okay, as long as there's no impl
// that retains it.
trait Tr {
    type A = Self::B;
    type B = Self::A;
}

// An impl has to break the cycle to be accepted.
impl Tr for u8 {
    type A = u8;
}

fn main() {}
