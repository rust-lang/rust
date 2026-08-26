// Prefixes including `c` as used by C string literals are only reserved in Rust 2021 and onward.
// Exercise what happens pre Rust 2021 with C string literal "lookalikes".

//@ check-pass
//@ edition: 2015..2021

#![warn(rust_2021_prefixes_incompatible_syntax)]

fn main() {
    // Make sure that pre Rust 2021 editions we continue to parse the snippet
    // `c"hello"` as an identifier followed by a (normal) string literal and
    // allow the code below to compile.
    //
    // issue: <https://github.com/rust-lang/rust/issues/113235>

    // Moreover, make sure we emit the relevant edition migration lint with an appropriate
    // diagnostic (for a period of time we used to incorrectly state prefix `c` was unknown and
    // that the token sequence would unconditionally lead to a hard error in the next edition).

    macro_rules! parse {
        (c $e:expr) => {
            $e
        };
    }

    let _: &'static str = parse!(c"hello");
    //~^ WARNING prefix `c` is unknown
    //~| WARNING hard error in Rust 2021

    macro_rules! indifferent {
        ($e:expr) => {};
        (c $e:expr) => {};
    }

    indifferent!(c"...");
    //~^ WARNING prefix `c` is unknown
    //~| WARNING hard error in Rust 2021
}
