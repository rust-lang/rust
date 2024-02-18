//@ check-pass

#![feature(inline_const)]
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

pub fn from_utf8_with_indirections() {
    let mut a = [99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8_mut(&mut a);
    //~^ WARN calls to `std::str::from_utf8_mut`
    let mut b = &mut a;
    let mut c = b;
    std::str::from_utf8_mut(c);
    //~^ WARN calls to `std::str::from_utf8_mut`
    let mut c = &[99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8(c);
    //~^ WARN calls to `std::str::from_utf8`
    const INVALID_1: [u8; 7] = [99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8(&INVALID_1);
    //~^ WARN calls to `std::str::from_utf8`
    static INVALID_2: [u8; 7] = [99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8(&INVALID_2);
    //~^ WARN calls to `std::str::from_utf8`
    const INVALID_3: &'static [u8; 7] = &[99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8(INVALID_3);
    //~^ WARN calls to `std::str::from_utf8`
    const INVALID_4: &'static [u8; 7] = { &[99, 108, 130, 105, 112, 112, 121] };
    std::str::from_utf8(INVALID_4);
    //~^ WARN calls to `std::str::from_utf8`
}

fn main() {}
