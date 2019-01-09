// compile-pass

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO NOT warn when lifetime name is used only
// once in a fn return type -- using `'_` is not legal there,
// as it must refer back to an argument.
//
// (Normally, using `'static` would be preferred, but there are
// times when that is not what you want.)
//
// run-pass

fn b<'a>() -> &'a u32 { // OK: used only in return type
    &22
}

fn main() { }
