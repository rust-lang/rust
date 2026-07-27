//@ edition: 2024
//@ compile-flags: -Zthreads=0 --crate-type lib

#![allow(dead_code)]

fn with_option<const O: u32>() {
    with_option::<{ async || {} }>();
    //~^ ERROR mismatched types
}

//@ normalize-stderr: "found `\{async closure@[^`]*\}`" -> "found `{async closure@...}`"
