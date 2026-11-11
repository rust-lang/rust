//@ compile-flags: --crate-type=lib

// Test diagnostic output for E0117 when implementing foreign traits
// with defaulted parameters (like `PartialEq` and `Add`) on foreign types.
// addresses https://github.com/rust-lang/rust/issues/160648

use std::ops::Add;

// Case 1: Foreign trait, foreign type in Self position, defaulted Rhs (PartialEq)
impl PartialEq for Option<u32> {
    //~^ ERROR only traits defined in the current crate can be implemented for types defined outside of the crate
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

// Case 2: Foreign trait, foreign primitive in Self position, defaulted Rhs (Add)
impl Add for u32 {
    //~^ ERROR only traits defined in the current crate can be implemented for primitive types
    type Output = u32;
    fn add(self, _rhs: u32) -> u32 {
        self
    }
}

// Case 3: Foreign trait with explicit foreign Rhs type on a foreign Self type
impl PartialEq<String> for Option<u32> {
    //~^ ERROR only traits defined in the current crate can be implemented for types defined outside of the crate
    fn eq(&self, _other: &String) -> bool {
        false
    }
}

// Case 4: Foreign trait with explicit foreign array Rhs type on a foreign Self type
// (control case: Array already used the foreign-trait label before this fix,
// via the pre-existing `Slice`/`Array`/`Tuple` arms. included here so that
// behavior is also pinned down as a regression test.)
impl PartialEq<[i32; 3]> for Option<u32> {
    //~^ ERROR only traits defined in the current crate can be implemented for types defined outside of the crate
    fn eq(&self, _other: &[i32; 3]) -> bool {
        false
    }
}
