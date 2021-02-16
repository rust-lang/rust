// ignore-test FIXME (Miri issue #1711)
#![allow(invalid_value)]

union MyUninit {
    init: (),
    uninit: fn(),
}

fn main() {
    let _b = unsafe { MyUninit { init: () }.uninit }; //~ ERROR encountered uninitialized bytes, but expected a function pointer
}
