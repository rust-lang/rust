//@ run-rustfix
#![crate_type = "lib"]
#![allow(unnecessary_transmutes, unused_imports)]
#![deny(deprecated)]

extern crate core;

use std::intrinsics::transmute as _;
//~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
use core::intrinsics::copy as _;
//~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
use std::intrinsics::copy_nonoverlapping as _;
//~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
use core::intrinsics::write_bytes as _;
//~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`

use core::intrinsics::{
    copy as _,
    //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
    copy_nonoverlapping as _,
    //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
    write_bytes as _,
    //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
};

pub fn what() {
    unsafe {
        let value = 42_u8;
        let mut dst = 0;
        let _ = std::intrinsics::transmute::<u8, i8>(value);
        //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
        core::intrinsics::copy(&value, &mut dst, 1);
        //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
        core::intrinsics::copy_nonoverlapping(&value, &mut dst, 1);
        //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
        std::intrinsics::write_bytes(&mut dst, value, 1)
        //~^ ERROR use of deprecated import through accidentally stabilized module `intrinsics`
    }
}
