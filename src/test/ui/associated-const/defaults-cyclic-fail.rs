// build-fail
//~^ ERROR cycle detected when normalizing `<() as Tr>::A`

// Cyclic assoc. const defaults don't error unless *used*
trait Tr {
    const A: u8 = Self::B;

    const B: u8 = Self::A;
}

// This impl is *allowed* unless its assoc. consts are used
impl Tr for () {}

fn main() {
    // This triggers the cycle error
    assert_eq!(<() as Tr>::A, 0);
}
