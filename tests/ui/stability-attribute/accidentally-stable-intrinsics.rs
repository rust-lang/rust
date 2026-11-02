//@ run-rustfix
#![crate_type = "lib"]
#![allow(unnecessary_transmutes, unused_imports)]

extern crate core;

use std::intrinsics::transmute as _;
//~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
//~| WARN previously accepted
use core::intrinsics::copy as _;
//~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
//~| WARN previously accepted
use std::intrinsics::copy_nonoverlapping as _;
//~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
//~| WARN previously accepted
use core::intrinsics::write_bytes as _;
//~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
//~| WARN previously accepted

use core::intrinsics::{
    copy as _,
    //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
    //~| WARN previously accepted
    copy_nonoverlapping as _,
    //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
    //~| WARN previously accepted
    write_bytes as _,
    //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
    //~| WARN previously accepted
};

pub fn what() {
    unsafe {
        let value = 42_u8;
        let mut dst = 0;
        let _ = std::intrinsics::transmute::<u8, i8>(value);
        //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
        //~| WARN previously accepted
        core::intrinsics::copy(&value, &mut dst, 1);
        //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
        //~| WARN previously accepted
        core::intrinsics::copy_nonoverlapping(&value, &mut dst, 1);
        //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
        //~| WARN previously accepted
        std::intrinsics::write_bytes(&mut dst, value, 1)
        //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
        //~| WARN previously accepted
    }
}
