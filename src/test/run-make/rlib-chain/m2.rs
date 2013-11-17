#[crate_type = "rlib"];
extern mod m1;

pub fn m2() { m1::m1() }
