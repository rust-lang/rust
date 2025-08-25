#![allow(integer_to_ptr_transmutes)]

use std::mem::transmute;

#[cfg(target_pointer_width = "32")]
type TwoPtrs = i64;
#[cfg(target_pointer_width = "64")]
type TwoPtrs = i128;

fn main() {
    for &my_bool in &[true, false] {
        let mask = -(my_bool as TwoPtrs); // false -> 0, true -> -1 aka !0
        // This is branchless code to select one or the other pointer.
        // However, it drops provenance when transmuting to TwoPtrs, so this is UB.
        let val = unsafe {
            transmute::<_, &str>(
                //~^ ERROR: constructing invalid value: encountered a dangling reference
                !mask & transmute::<_, TwoPtrs>("false !")
                    | mask & transmute::<_, TwoPtrs>("true !"),
            )
        };
        println!("{}", val);
    }
}
