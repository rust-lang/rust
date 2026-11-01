//@ check-pass

#![feature(concat_bytes)]
#![warn(invalid_from_utf8_unchecked)]
#![warn(invalid_from_utf8)]

pub fn from_utf8_unchecked_mut() {
    // Valid
    unsafe {
        std::str::from_utf8_unchecked_mut(&mut [99, 108, 105, 112, 112, 121]);
        str::from_utf8_unchecked_mut(&mut [99, 108, 105, 112, 112, 121]);
        std::str::from_utf8_unchecked_mut(&mut [b'c', b'l', b'i', b'p', b'p', b'y']);
        str::from_utf8_unchecked_mut(&mut [b'c', b'l', b'i', b'p', b'p', b'y']);

        let x = 0xA0;
        std::str::from_utf8_unchecked_mut(&mut [0xC0, x]);
    }

    // Invalid
    unsafe {
        std::str::from_utf8_unchecked_mut(&mut [99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `std::str::from_utf8_unchecked_mut`
        str::from_utf8_unchecked_mut(&mut [99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `str::from_utf8_unchecked_mut`
        std::str::from_utf8_unchecked_mut(&mut [b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `std::str::from_utf8_unchecked_mut`
        str::from_utf8_unchecked_mut(&mut [b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `str::from_utf8_unchecked_mut`
    }
}

pub fn from_utf8_unchecked() {
    // Valid
    unsafe {
        std::str::from_utf8_unchecked(&[99, 108, 105, 112, 112, 121]);
        str::from_utf8_unchecked(&[99, 108, 105, 112, 112, 121]);
        std::str::from_utf8_unchecked(&[b'c', b'l', b'i', b'p', b'p', b'y']);
        str::from_utf8_unchecked(&[b'c', b'l', b'i', b'p', b'p', b'y']);
        std::str::from_utf8_unchecked(b"clippy");
        str::from_utf8_unchecked(b"clippy");

        let x = 0xA0;
        std::str::from_utf8_unchecked(&[0xC0, x]);
        str::from_utf8_unchecked(&[0xC0, x]);
    }

    // Invalid
    unsafe {
        std::str::from_utf8_unchecked(&[99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `std::str::from_utf8_unchecked`
        str::from_utf8_unchecked(&[99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `str::from_utf8_unchecked`
        std::str::from_utf8_unchecked(&[b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `std::str::from_utf8_unchecked`
        str::from_utf8_unchecked(&[b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `str::from_utf8_unchecked`
        std::str::from_utf8_unchecked(b"cl\x82ippy");
        //~^ WARN calls to `std::str::from_utf8_unchecked`
        str::from_utf8_unchecked(b"cl\x82ippy");
        //~^ WARN calls to `str::from_utf8_unchecked`
        std::str::from_utf8_unchecked(concat_bytes!(b"cl", b"\x82ippy"));
        //~^ WARN calls to `std::str::from_utf8_unchecked`
        str::from_utf8_unchecked(concat_bytes!(b"cl", b"\x82ippy"));
        //~^ WARN calls to `str::from_utf8_unchecked`
    }
}

pub fn from_utf8_mut() {
    // Valid
    {
        std::str::from_utf8_mut(&mut [99, 108, 105, 112, 112, 121]);
        str::from_utf8_mut(&mut [99, 108, 105, 112, 112, 121]);
        std::str::from_utf8_mut(&mut [b'c', b'l', b'i', b'p', b'p', b'y']);
        str::from_utf8_mut(&mut [b'c', b'l', b'i', b'p', b'p', b'y']);

        let x = 0xa0;
        std::str::from_utf8_mut(&mut [0xc0, x]);
        str::from_utf8_mut(&mut [0xc0, x]);
    }

    // Invalid
    {
        std::str::from_utf8_mut(&mut [99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `std::str::from_utf8_mut`
        str::from_utf8_mut(&mut [99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `str::from_utf8_mut`
        std::str::from_utf8_mut(&mut [b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `std::str::from_utf8_mut`
        str::from_utf8_mut(&mut [b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `str::from_utf8_mut`
    }
}

pub fn from_utf8() {
    // Valid
    {
        std::str::from_utf8(&[99, 108, 105, 112, 112, 121]);
        str::from_utf8(&[99, 108, 105, 112, 112, 121]);
        std::str::from_utf8(&[b'c', b'l', b'i', b'p', b'p', b'y']);
        str::from_utf8(&[b'c', b'l', b'i', b'p', b'p', b'y']);
        std::str::from_utf8(b"clippy");
        str::from_utf8(b"clippy");

        let x = 0xA0;
        std::str::from_utf8(&[0xC0, x]);
        str::from_utf8(&[0xC0, x]);
    }

    // Invalid
    {
        std::str::from_utf8(&[99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `std::str::from_utf8`
        str::from_utf8(&[99, 108, 130, 105, 112, 112, 121]);
        //~^ WARN calls to `str::from_utf8`
        std::str::from_utf8(&[b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `std::str::from_utf8`
        str::from_utf8(&[b'c', b'l', b'\x82', b'i', b'p', b'p', b'y']);
        //~^ WARN calls to `str::from_utf8`
        std::str::from_utf8(b"cl\x82ippy");
        //~^ WARN calls to `std::str::from_utf8`
        str::from_utf8(b"cl\x82ippy");
        //~^ WARN calls to `str::from_utf8`
        std::str::from_utf8(concat_bytes!(b"cl", b"\x82ippy"));
        //~^ WARN calls to `std::str::from_utf8`
        str::from_utf8(concat_bytes!(b"cl", b"\x82ippy"));
        //~^ WARN calls to `str::from_utf8`
    }
}

pub fn from_utf8_with_indirections() {
    // NOTE: We used to lint on the patterns below, but due to the
    // binding being mutable it could be changed between the
    // declaration and the call and that would have created a
    // false-positive, so until we can reliably avoid those false
    // postive we don't lint on them. Example of FP below.
    //
    // let mut a = [99, 108, 130, 105, 112, 112, 121];
    // std::str::from_utf8_mut(&mut a);
    // str::from_utf8_mut(&mut a);
    // let mut b = &mut a;
    // let mut c = b;
    // std::str::from_utf8_mut(c);
    // str::from_utf8_mut(c);

    let c = &[99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8(c);
    //~^ WARN calls to `std::str::from_utf8`
    str::from_utf8(c);
    //~^ WARN calls to `str::from_utf8`
    const INVALID_1: [u8; 7] = [99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8(&INVALID_1);
    //~^ WARN calls to `std::str::from_utf8`
    str::from_utf8(&INVALID_1);
    //~^ WARN calls to `str::from_utf8`
    static INVALID_2: [u8; 7] = [99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8(&INVALID_2);
    //~^ WARN calls to `std::str::from_utf8`
    str::from_utf8(&INVALID_2);
    //~^ WARN calls to `str::from_utf8`
    const INVALID_3: &'static [u8; 7] = &[99, 108, 130, 105, 112, 112, 121];
    std::str::from_utf8(INVALID_3);
    //~^ WARN calls to `std::str::from_utf8`
    str::from_utf8(INVALID_3);
    //~^ WARN calls to `str::from_utf8`
    const INVALID_4: &'static [u8; 7] = { &[99, 108, 130, 105, 112, 112, 121] };
    std::str::from_utf8(INVALID_4);
    //~^ WARN calls to `std::str::from_utf8`
    str::from_utf8(INVALID_4);
    //~^ WARN calls to `str::from_utf8`

    let mut a = [99, 108, 130, 105, 112, 112, 121]; // invalid
    loop {
        a = [99, 108, 130, 105, 112, 112, 121]; // still invalid, but too complex
        break;
    }
    std::str::from_utf8_mut(&mut a);

    let mut a = [99, 108, 130, 105, 112, 112]; // invalid
    loop {
        a = *b"clippy"; // valid
        break;
    }
    std::str::from_utf8_mut(&mut a);
}

fn main() {}
