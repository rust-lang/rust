const C1: &'static mut [usize] = &mut [];
//~^ ERROR: mutable references are not allowed

static mut S: usize = 3;
const C2: &'static mut usize = unsafe { &mut S };
//~^ WARN creating a mutable reference to mutable static is discouraged [static_mut_refs]
//~^^ ERROR: constants cannot refer to statics
//~| ERROR: constants cannot refer to statics

fn main() {}
