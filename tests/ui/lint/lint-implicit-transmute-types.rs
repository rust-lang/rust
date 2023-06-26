// Test the implicit_transmute_types lint.

#![deny(implicit_transmute_types)]

use std::mem::transmute;

fn main() {
    unsafe {
        let _: i32 = transmute(123u32);
        //~^ ERROR: `transmute` called without explicit type parameters

        let _: i32 = transmute::<u32, i32>(123u32);
        let _: i32 = transmute::<_, _>(123u32);
    }
}
