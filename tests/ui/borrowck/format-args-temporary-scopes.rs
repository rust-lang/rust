//! Test for #145784 as it relates to format arguments: arguments to macros such as `println!`
//! should obey normal temporary scoping rules.
//@ revisions: e2021 e2024
//@ [e2021] check-pass
//@ [e2021] edition: 2021
//@ [e2024] edition: 2024

fn temp() {}

fn main() {
    // In Rust 2024, block tail expressions are temporary scopes, so the result of `temp()` is
    // dropped after evaluating `&temp()`.
    println!("{:?}", { &temp() });
    //[e2024]~^ ERROR: temporary value dropped while borrowed [E0716]

    // In Rust 1.89, `format_args!` extended the lifetime of all extending expressions in its
    // arguments when provided with two or more arguments. This caused the result of `temp()` to
    // outlive the result of the block, making this compile.
    println!("{:?}{:?}", { &temp() }, ());
    //[e2024]~^ ERROR: temporary value dropped while borrowed [E0716]
}
