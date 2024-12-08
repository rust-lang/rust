// Check that we forbid nested statics in `thread_local` statics.

#![feature(thread_local)]

#[thread_local]
static mut FOO: &u32 = {
    //~^ ERROR: does not support implicit nested statics
    // Prevent promotion (that would trigger on `&42` as an expression)
    let x = 42;
    &{ x }
};

fn main() {}
