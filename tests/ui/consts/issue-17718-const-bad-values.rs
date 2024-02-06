const C1: &'static mut [usize] = &mut [];
//~^ ERROR: mutable references are not allowed

static mut S: usize = 3;
const C2: &'static mut usize = unsafe { &mut S };
//~^ ERROR: referencing statics in constants
//~| ERROR: referencing statics in constants
//~| WARN mutable reference of mutable static is discouraged [static_mut_ref]

fn main() {}
