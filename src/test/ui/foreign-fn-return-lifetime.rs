extern "C" {
    fn g(_: &u8) -> &u8; // OK
    fn f() -> &u8; //~ ERROR missing lifetime specifier
}

fn main() {}
