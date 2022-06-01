#![allow(invalid_value)]

union MyUninit {
    init: (),
    uninit: bool,
}

fn main() {
    let _b = unsafe { MyUninit { init: () }.uninit }; //~ ERROR encountered uninitialized bytes, but expected a boolean
}
