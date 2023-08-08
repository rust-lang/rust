// check-pass

#![feature(concat_bytes)]
#![warn(invalid_from_utf8_unchecked)]
#![warn(invalid_from_utf8)]

pub fn from_utf8_unchecked_mut() {
    // Valid
    unsafe {
        std::str::from_utf8_unchecked_mut(&mut [99, 108, 105, 112, 112, 121]);
        std::str::from_utf8_unchecked_mut(&mut [b'c', b'l', b'i', b'p', b'p', b'y']);

        let x = 0xA0;
        std::str::from_utf8_unchecked_mut(&mut [0xC0, x]);
    }

    // Invalid
    unsafe {
        std::str::from_utf8_unchecked_mut(&mut [99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `std::str::from_utf8_unchecked_mut`
        std::str::from_utf8_unchecked_mut(&mut [b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `std::str::from_utf8_unchecked_mut`
    }
}

pub fn from_utf8_unchecked() {
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
        //~^ WARN calls to `std::str::from_utf8_unchecked`
        std::str::from_utf8_unchecked(&[b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `std::str::from_utf8_unchecked`
        std::str::from_utf8_unchecked(b"cl\x82ippy");
        //~^ WARN calls to `std::str::from_utf8_unchecked`
        std::str::from_utf8_unchecked(concat_bytes!(b"cl", b"\x82ippy"));
        //~^ WARN calls to `std::str::from_utf8_unchecked`
    }
}

pub fn from_utf8_mut() {
    // Valid
    {
        std::str::from_utf8_mut(&mut [99, 108, 105, 112, 112, 121]);
        std::str::from_utf8_mut(&mut [b'c', b'l', b'i', b'p', b'p', b'y']);

        let x = 0xa0;
        std::str::from_utf8_mut(&mut [0xc0, x]);
    }

    // Invalid
    {
        std::str::from_utf8_mut(&mut [99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `std::str::from_utf8_mut`
        std::str::from_utf8_mut(&mut [b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `std::str::from_utf8_mut`
    }
}

pub fn from_utf8() {
    // Valid
    {
        std::str::from_utf8(&[99, 108, 105, 112, 112, 121]);
        std::str::from_utf8(&[b'c', b'l', b'i', b'p', b'p', b'y']);
        std::str::from_utf8(b"clippy");

        let x = 0xA0;
        std::str::from_utf8(&[0xC0, x]);
    }

    // Invalid
    {
        std::str::from_utf8(&[99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `std::str::from_utf8`
        std::str::from_utf8(&[b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `std::str::from_utf8`
        std::str::from_utf8(b"cl\x82ippy");
        //~^ WARN calls to `std::str::from_utf8`
        std::str::from_utf8(concat_bytes!(b"cl", b"\x82ippy"));
        //~^ WARN calls to `std::str::from_utf8`
    }
}

fn main() {}
