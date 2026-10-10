#![allow(invalid_value)]

union MyUninit {
    init: (),
    uninit: [bool; 1],
}

fn main() {
    let _b = unsafe { MyUninit { init: () }.uninit }; //~ ERROR: constructing invalid value
}
