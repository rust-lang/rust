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

macro_rules! demo5 {
    ( $a:tt $b:tt $c:tt $d:tt $e:tt ) => { println!("five tokens") };
}

macro_rules! demo7 {
    ( $a:tt $b:tt $c:tt $d:tt $e:tt $f:tt $g:tt ) => { println!("seven tokens") };
}


fn main() {
    demo3!(## "foo");
    //~^ WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    demo4!(### "foo");
    //~^ WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
   demo4!(## "foo"#);
    //~^ WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    demo7!(### "foo"###);
    //~^ WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024

    demo5!(###"foo"#);
    //~^ WARNING parsed as a guarded string literal in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a guarded string literal in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a guarded string literal in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    demo5!(#"foo"###);
    //~^ WARNING parsed as a guarded string literal in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    demo4!("foo"###);
    //~^ WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    //~| WARNING parsed as a reserved token in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024

    // Non-ascii identifiers
    demo2!(Ñ"foo");
    //~^ ERROR prefix `Ñ` is unknown
    demo4!(Ñ#""#);
    //~^ ERROR prefix `Ñ` is unknown
    //~| WARNING parsed as a guarded string literal in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
    demo3!(🙃#"");
    //~^ ERROR identifiers cannot contain emoji
    //~| WARNING parsed as a guarded string literal in Rust 2024 and onward [rust_2024_guarded_string_incompatible_syntax]
    //~| WARNING hard error in Rust 2024
}
