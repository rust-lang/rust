//! This test verifies that `std::num::TryFromIntError` correctly implements `PartialEq`,
//! allowing `Result<T, TryFromIntError>` values to be compared for equality using `==`.
//! It specifically checks a successful numeric conversion scenario where the `Result::Ok`
//! variant is compared, ensuring that the comparison yields the expected boolean result.

//@ run-pass

#![allow(unused_must_use)] // Allow ignoring the result of the comparison for the test's purpose

use std::convert::TryFrom;
use std::num::TryFromIntError;

fn main() {
    let x: u32 = 125;
    // Attempt to convert u32 to u8, which should succeed as 125 fits in u8.
    let y: Result<u8, TryFromIntError> = u8::try_from(x);
    // Verify that the Result can be correctly compared with an Ok value.
    y == Ok(125);
}
