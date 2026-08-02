//@ edition: 2024
//@ compile-flags: -Zthreads=0 --crate-type lib

#![allow(dead_code)]

fn f<'a, const N: u8>()
where
    'a: 'static,
{
    f::<{ async || {} }>();
    //~^ ERROR mismatched types
}

//@ normalize-stderr: "found `\{async closure@[^`]*\}`" -> "found `{async closure@...}`"
