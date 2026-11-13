#![warn(clippy::mem_replace_option_with_none)]
#![no_std]

use core::mem;

fn it_works() {
    let mut an_option = Some(1);
    let _ = mem::replace(&mut an_option, None);
    //~^ mem_replace_option_with_none
    let an_option = &mut Some(1);
    let _ = mem::replace(an_option, None);
    //~^ mem_replace_option_with_none
}
