#![allow(invalid_value)]

union MyUninit {
    init: (),
    uninit: [char; 1],
}

fn main() {
    let _b = unsafe { MyUninit { init: () }.uninit }; //~ ERROR: constructing invalid value
}
