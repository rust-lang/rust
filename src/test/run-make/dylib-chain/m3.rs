#[crate_type = "dylib"];
extern mod m2;

pub fn m3() { m2::m2() }
