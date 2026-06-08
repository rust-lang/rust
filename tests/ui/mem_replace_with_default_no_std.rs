#![warn(clippy::mem_replace_with_default)]
#![no_std]

use core::mem;

fn it_works() {
    let mut refstr = "hello";
    let _ = mem::replace(&mut refstr, "");
    //~^ mem_replace_with_default

    let mut slice: &[i32] = &[1, 2, 3];
    let _ = mem::replace(&mut slice, &[]);
    //~^ mem_replace_with_default
}
