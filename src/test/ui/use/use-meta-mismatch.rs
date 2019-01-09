// error-pattern:can't find crate for `extra`

extern crate fake_crate as extra;

fn main() { }
