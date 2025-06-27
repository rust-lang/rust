// Checks that immutable static items can't have mutable slices

static TEST: &'static mut [isize] = &mut [];
//~^ ERROR mutable borrows of temporaries

pub fn main() { }
