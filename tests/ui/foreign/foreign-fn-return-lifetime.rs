extern "C" {
    pub fn g(_: &u8) -> &u8; // OK
    pub fn f() -> &u8; //~ ERROR missing lifetime specifier
}

fn main() {}
