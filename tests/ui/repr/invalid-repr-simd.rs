// Regression test for the ICEs described in #83505 & #121097.

#![crate_type="lib"]

#[repr(simd)]
//~^ ERROR: attribute should be applied to a struct [E0517]
//~| ERROR: unsupported representation for zero-variant enum [E0084]
enum Es {}
static CLs: Es;
//~^ ERROR: free static item without body

#[repr(simd)]
//~^ ERROR: attribute should be applied to a struct [E0517]
enum Aligned {
    Zero = 0,
    One = 1,
}

pub fn tou8(al: Aligned) -> u8 {
    al as u8
}
