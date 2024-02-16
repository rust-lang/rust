//@ run-rustfix

extern crate std;
fn main() {}
//~^^ ERROR the name `std` is defined multiple times [E0259]
