#![allow(invalid_value)]

union MyUninit {
    init: (),
    uninit: fn(),
}

fn main() {
    let _b = unsafe { MyUninit { init: () }.uninit }; //~ ERROR encountered uninitialized bytes
}
