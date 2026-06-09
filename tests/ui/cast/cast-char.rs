#![deny(overflowing_literals)]

fn main() {
    // Valid cases - should suggest char literal

    // u8 range (0-255)
    const VALID_U8_1: char = 0x41 as char; // 'A'
    const VALID_U8_2: char = 0xFF as char; // maximum u8
    const VALID_U8_3: char = 0x00 as char; // minimum u8

    // Valid Unicode in lower range [0x0, 0xD7FF]
    const VALID_LOW_1: char = 0x1000 as char; // 4096
    //~^ ERROR: only `u8` can be cast into `char`
    const VALID_LOW_2: char = 0xD7FF as char; // last valid in lower range
    //~^ ERROR: only `u8` can be cast into `char`
    const VALID_LOW_3: char = 0x0500 as char; // cyrillic range
    //~^ ERROR: only `u8` can be cast into `char`

    // Valid Unicode in upper range [0xE000, 0x10FFFF]
    const VALID_HIGH_1: char = 0xE000 as char; // first valid in upper range
    //~^ ERROR only `u8` can be cast into `char`
    const VALID_HIGH_2: char = 0x1F888 as char; // 129160 - example from issue
    //~^ ERROR only `u8` can be cast into `char`
    const VALID_HIGH_3: char = 0x10FFFF as char; // maximum valid Unicode
    //~^ ERROR only `u8` can be cast into `char`
    const VALID_HIGH_4: char = 0xFFFD as char; // replacement character
    //~^ ERROR only `u8` can be cast into `char`
    const VALID_HIGH_5: char = 0x1F600 as char; // emoji
    //~^ ERROR only `u8` can be cast into `char`

    // Invalid cases - should show InvalidCharCast

    // Surrogate range [0xD800, 0xDFFF] - reserved for UTF-16
    const INVALID_SURROGATE_1: char = 0xD800 as char; // first surrogate
    //~^ ERROR: surrogate values are not valid
    const INVALID_SURROGATE_2: char = 0xDFFF as char; // last surrogate
    //~^ ERROR: surrogate values are not valid
    const INVALID_SURROGATE_3: char = 0xDB00 as char; // middle of surrogate range
    //~^ ERROR: surrogate values are not valid

    // Too large values (> 0x10FFFF)
    const INVALID_TOO_BIG_1: char = 0x110000 as char; // one more than maximum
    //~^ ERROR: value exceeds maximum `char` value
    const INVALID_TOO_BIG_2: char = 0xEF8888 as char; // example from issue
    //~^ ERROR: value exceeds maximum `char` value
    const INVALID_TOO_BIG_3: char = 0x1FFFFF as char; // much larger
    //~^ ERROR: value exceeds maximum `char` value
    const INVALID_TOO_BIG_4: char = 0xFFFFFF as char; // 24-bit maximum
    //~^ ERROR: value exceeds maximum `char` value

    // Boundary cases
    const BOUNDARY_1: char = 0xD7FE as char; // valid, before surrogate
    //~^ ERROR only `u8` can be cast into `char`
    const BOUNDARY_2: char = 0xE001 as char; // valid, after surrogate
    //~^ ERROR only `u8` can be cast into `char`
    const BOUNDARY_3: char = 0x10FFFE as char; // valid, near maximum
    //~^ ERROR only `u8` can be cast into `char`
}
