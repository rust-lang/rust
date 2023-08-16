//@error-in-other-file:can't find crate for `fake_crate`

extern crate fake_crate as extra;

fn main() { }
