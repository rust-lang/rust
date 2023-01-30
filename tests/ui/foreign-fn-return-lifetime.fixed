// run-rustfix

extern "C" {
    pub fn g(_: &u8) -> &u8; // OK
    pub fn f() -> &'static u8; //~ ERROR missing lifetime specifier
}

fn main() {}
