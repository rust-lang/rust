//@ edition: 2024
//@ compile-flags: -Zthreads=0 --crate-type lib

#![allow(dead_code)]

fn f<const N: u8>(_: impl Sized) {
    f::<{ async || {} }>(());
    //~^ ERROR mismatched types
}

//@ normalize-stderr: "found `\{async closure@[^`]*\}`" -> "found `{async closure@...}`"
