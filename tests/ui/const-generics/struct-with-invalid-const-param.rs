// Checks that a const param cannot be stored in a struct.

struct S<const C: u8>(C); //~ ERROR expected type, found const parameter

fn main() {}
