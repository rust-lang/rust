//@ run-pass

// Cyclic assoc. const defaults don't error unless *used*
trait Tr {
    const A: u8 = Self::B;
    const B: u8 = Self::A;
}

// This impl is *allowed* unless its assoc. consts are used, matching the
// behavior without defaults.
impl Tr for () {}

// Overriding either constant breaks the cycle
impl Tr for u8 {
    const A: u8 = 42;
}

impl Tr for u16 {
    const B: u8 = 0;
}

impl Tr for u32 {
    const A: u8 = 100;
    const B: u8 = 123;
}

fn main() {
    assert_eq!(<u8 as Tr>::A, 42);
    assert_eq!(<u8 as Tr>::B, 42);

    assert_eq!(<u16 as Tr>::A, 0);
    assert_eq!(<u16 as Tr>::B, 0);

    assert_eq!(<u32 as Tr>::A, 100);
    assert_eq!(<u32 as Tr>::B, 123);
}
