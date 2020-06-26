// Checks that immutable static items can't have mutable slices

static TEST: &'static mut [isize] = &mut [];
//~^ ERROR mutable references are not allowed in statics

pub fn main() { }
