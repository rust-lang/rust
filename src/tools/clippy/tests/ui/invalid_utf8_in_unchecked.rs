#![warn(clippy::invalid_utf8_in_unchecked)]

fn main() {
    // Valid
    unsafe {
        std::str::from_utf8_unchecked(&[99, 108, 105, 112, 112, 121]);
        std::str::from_utf8_unchecked(&[b'c', b'l', b'i', b'p', b'p', b'y']);
        std::str::from_utf8_unchecked(b"clippy");

        let x = 0xA0;
        std::str::from_utf8_unchecked(&[0xC0, x]);
    }

    // Invalid
    unsafe {
        std::str::from_utf8_unchecked(&[99, 108, 130, 105, 112, 112, 121]);
        std::str::from_utf8_unchecked(&[b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        std::str::from_utf8_unchecked(b"cl\x82ippy");
    }
}
