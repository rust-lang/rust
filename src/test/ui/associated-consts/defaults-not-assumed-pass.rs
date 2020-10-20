// run-pass

trait Tr {
    const A: u8 = 255;

    // This should not be a constant evaluation error (overflow). The value of
    // `Self::A` must not be assumed to hold inside the trait.
    const B: u8 = Self::A + 1;
}

// An impl that doesn't override any constant will NOT cause a const eval error
// just because it's defined, but only if the bad constant is used anywhere.
// This matches the behavior without defaults.
impl Tr for () {}

// An impl that overrides either constant with a suitable value will be fine.
impl Tr for u8 {
    const A: u8 = 254;
}

impl Tr for u16 {
    const B: u8 = 0;
}

impl Tr for u32 {
    const A: u8 = 254;
    const B: u8 = 0;
}

fn main() {
    assert_eq!(<() as Tr>::A, 255);
    //assert_eq!(<() as Tr>::B, 0);  // using this is an error

    assert_eq!(<u8 as Tr>::A, 254);
    assert_eq!(<u8 as Tr>::B, 255);

    assert_eq!(<u16 as Tr>::A, 255);
    assert_eq!(<u16 as Tr>::B, 0);

    assert_eq!(<u32 as Tr>::A, 254);
    assert_eq!(<u32 as Tr>::B, 0);
}
