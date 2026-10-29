#![warn(clippy::mem_replace_option_with_some)]
#![no_std]

use core::mem;

#[clippy::msrv = "1.31"]
fn it_works() {
    let mut an_option = Some(0);
    let replaced = mem::replace(&mut an_option, Some(1));
    //~^ mem_replace_option_with_some

    let mut an_option = &mut Some(0);
    let replaced = mem::replace(an_option, Some(1));
    //~^ mem_replace_option_with_some

    let (mut opt1, mut opt2) = (Some(0), Some(0));
    let b = true;
    let replaced = mem::replace(if b { &mut opt1 } else { &mut opt2 }, Some(1));
    //~^ mem_replace_option_with_some
}

#[clippy::msrv = "1.30"]
fn bad_msrv() {
    let mut an_option = Some(0);
    let replaced = mem::replace(&mut an_option, Some(1));
}
