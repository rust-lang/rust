// Checks that immutable static items can't have mutable slices

static TEST: &'static mut [isize] = &mut [];
//~^ ERROR references in statics may only refer to immutable values

pub fn main() { }
