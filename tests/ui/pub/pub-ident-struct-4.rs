//@ run-rustfix

#[allow(dead_code)]
pub T(String);
//~^ ERROR missing `struct` for struct definition

fn main() {}
