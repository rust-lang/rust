//@ edition: 2024
//@ compile-flags: -Zthreads=0 --crate-type lib

#![allow(dead_code)]

fn f<T, const N: u8>() {
    f::<u8, { async || {} }>();
    //~^ ERROR mismatched types
}

//@ normalize-stderr: "found `\{async closure@[^`]*\}`" -> "found `{async closure@...}`"
