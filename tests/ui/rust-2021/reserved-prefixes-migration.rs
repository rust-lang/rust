//@ check-pass
//@ run-rustfix
//@ edition:2018

#![warn(rust_2021_prefixes_incompatible_syntax)]

macro_rules! m2 {
    ($a:tt $b:tt) => {};
}

macro_rules! m3 {
    ($a:tt $b:tt $c:tt) => {};
}

fn main() {
    m2!(z"hey");
    //~^ WARNING prefix `z` is unknown [rust_2021_prefixes_incompatible_syntax]
    //~| WARNING hard error in Rust 2021
    m2!(prefix"hey");
    //~^ WARNING prefix `prefix` is unknown [rust_2021_prefixes_incompatible_syntax]
    //~| WARNING hard error in Rust 2021
    m3!(hey#123);
    //~^ WARNING prefix `hey` is unknown [rust_2021_prefixes_incompatible_syntax]
    //~| WARNING hard error in Rust 2021
    m3!(hey#hey);
    //~^ WARNING prefix `hey` is unknown [rust_2021_prefixes_incompatible_syntax]
    //~| WARNING hard error in Rust 2021
}

macro_rules! quote {
    (# name = # kind # value) => {};
}

quote! {
    #name = #kind#value
    //~^ WARNING prefix `kind` is unknown [rust_2021_prefixes_incompatible_syntax]
    //~| WARNING hard error in Rust 2021
}
