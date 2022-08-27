#![allow(invalid_value)]

union MyUninit {
    init: (),
    uninit: char,
}

fn main() {
    let _b = unsafe { MyUninit { init: () }.uninit }; //~ ERROR: uninitialized
}
