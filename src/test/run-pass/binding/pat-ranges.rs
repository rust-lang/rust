// run-pass
// Parsing of range patterns

const NUM1: i32 = 10;

mod m {
    pub const NUM2: i32 = 16;
}

fn main() {
    if let NUM1 ... m::NUM2 = 10 {} else { panic!() }
    if let ::NUM1 ... ::m::NUM2 = 11 {} else { panic!() }
    if let -13 ... -10 = 12 { panic!() } else {}
}
