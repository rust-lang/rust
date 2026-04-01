#![feature(staged_api)]
#![feature(const_eval_select)]
#![feature(core_intrinsics)]
#![stable(since = "1.0", feature = "ui_test")]

use std::intrinsics::const_eval_select;

fn log() {
    println!("HEY HEY HEY")
}

const fn nothing(){}

#[stable(since = "1.0", feature = "hey")]
#[rustc_const_stable(since = "1.0", feature = "const_hey")]
pub const fn hey() {
    const_eval_select((), nothing, log);
    //~^ ERROR cannot use `#[feature(const_eval_select)]`
}

fn main() {}
