// Unspecified kind should fail with an error

//@ compile-flags: -l =mylib

fn main() {}

//~? ERROR unknown library kind ``
