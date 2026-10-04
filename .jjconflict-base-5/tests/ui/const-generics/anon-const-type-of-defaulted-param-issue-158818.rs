//@ edition: 2024
//@ compile-flags: -Zthreads=0 --crate-type lib

#![allow(dead_code)]
#![allow(invalid_type_param_default)]

fn f<const N: u8, T = ()>() {
    f::<{ async || {} }>();
    //~^ ERROR mismatched types
}

//@ normalize-stderr: "found `\{async closure@[^`]*\}`" -> "found `{async closure@...}`"
