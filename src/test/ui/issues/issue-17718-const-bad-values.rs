const C1: &'static mut [usize] = &mut [];
//~^ ERROR: references in constants may only refer to immutable values

static mut S: usize = 3;
const C2: &'static mut usize = unsafe { &mut S };
//~^ ERROR: constants cannot refer to statics
//~| ERROR: references in constants may only refer to immutable values

fn main() {}
