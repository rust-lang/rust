#[crate_type = "dylib"];
extern mod m1;

pub fn m2() { m1::m1() }
