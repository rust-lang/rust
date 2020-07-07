#![allow(invalid_value)]

union MyUninit {
    init: (),
    uninit: char,
}

fn main() {
    let _b = unsafe { MyUninit { init: () }.uninit }; //~ ERROR encountered uninitialized bytes, but expected a valid unicode scalar value (in `0..=0x10FFFF` but not in `0xD800..=0xDFFF`)
}
