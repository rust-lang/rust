//@ only-x86_64
//@ revisions: apple other
//@[apple] only-apple
//@[other] ignore-apple

// Apple targets extend up to 32 bits for both arguments and returns, other targets only extend
// arguments.

#![crate_type = "lib"]
#![feature(rustc_attrs)]

#[rustc_abi(debug)]
pub extern "sysv64" fn i8(x: i8) -> i8 {
    //~^ ERROR fn_abi_of(i8)
    x
}

#[rustc_abi(debug)]
pub extern "sysv64" fn u8(x: u8) -> u8 {
    //~^ ERROR fn_abi_of(u8)
    x
}

#[rustc_abi(debug)]
pub extern "sysv64" fn i16(x: i16) -> i16 {
    //~^ ERROR fn_abi_of(i16)
    x
}

#[rustc_abi(debug)]
pub extern "sysv64" fn u16(x: u16) -> u16 {
    //~^ ERROR fn_abi_of(u16)
    x
}

#[rustc_abi(debug)]
pub extern "sysv64" fn i32(x: i32) -> i32 {
    //~^ ERROR fn_abi_of(i32)
    x
}

#[rustc_abi(debug)]
pub extern "sysv64" fn u32(x: u32) -> u32 {
    //~^ ERROR fn_abi_of(u32)
    x
}
