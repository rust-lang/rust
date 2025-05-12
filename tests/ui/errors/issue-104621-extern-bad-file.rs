//@ compile-flags: --extern foo={{src-base}}/errors/issue-104621-extern-bad-file.rs
//@ only-linux

extern crate foo;
//~^ ERROR extern location for foo is of an unknown type
//~| ERROR file name should be lib*.rlib or lib*.so
//~| ERROR can't find crate for `foo` [E0463]
fn main() {}
