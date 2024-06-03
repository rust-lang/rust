//@ edition:2021
// ignore-tidy-linelength

#![warn(rust_2024_guarded_string_incompatible_syntax)]

macro_rules! demo2 {
    ( $a:tt $b:tt ) => { println!("two tokens") };
}

macro_rules! demo3 {
    ( $a:tt $b:tt $c:tt ) => { println!("three tokens") };
}

macro_rules! demo4 {
    ( $a:tt $b:tt $c:tt $d:tt ) => { println!("four tokens") };
}

macro_rules! demon {
    ( $($n:tt)* ) => { println!("unknown number of tokens") };
}

fn main() {
    // Non-ascii identifiers
    demo2!(Ñ"foo");
    //~^ ERROR prefix `Ñ` is unknown
    demo4!(Ñ#""#);
    //~^ ERROR prefix `Ñ` is unknown
    //~| WARNING parsed as a guarded string in Rust 2024 [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    demo3!(🙃#"");
    //~^ ERROR identifiers cannot contain emoji
    //~| WARNING parsed as a guarded string in Rust 2024 [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024

    // More than 255 hashes
    demon!(####################################################################################################################################################################################################################################################################"foo");
    //~^ WARNING parsed as a guarded string in Rust 2024 [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
}
