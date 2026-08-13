// Checks that a const param cannot be stored in a struct.

struct S<const C: u8>(C); //~ ERROR cannot find type `C` in this scope

fn main() {}
