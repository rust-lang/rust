//@ check-pass

#![no_std]
#![crate_type = "lib"]
#![warn(invalid_from_utf8_unchecked)]
#![warn(invalid_from_utf8)]

pub fn checked() {
    let _ = core::str::from_utf8(b"cl\x82ippy");
    //~^ WARN calls to `core::str::from_utf8`
    let _ = core::str::from_utf8_mut(&mut [b'c', b'l', b'\x82']);
    //~^ WARN calls to `core::str::from_utf8_mut`
}

pub fn unchecked() {
    unsafe {
        let _ = core::str::from_utf8_unchecked(b"cl\x82ippy");
        //~^ WARN calls to `core::str::from_utf8_unchecked`
        let _ = core::str::from_utf8_unchecked_mut(&mut [b'c', b'l', b'\x82']);
        //~^ WARN calls to `core::str::from_utf8_unchecked_mut`
    }
}
