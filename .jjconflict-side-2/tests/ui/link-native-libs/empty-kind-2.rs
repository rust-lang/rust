// Unspecified kind should fail with an error

//@ compile-flags: -l :+bundle=mylib

fn main() {}

//~? ERROR unknown library kind ``
