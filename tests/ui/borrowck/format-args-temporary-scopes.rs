//! Test for #145784 as it relates to format arguments: arguments to macros such as `println!`
//! should obey normal temporary scoping rules.
//@ revisions: e2021 e2024
//@ [e2021] edition: 2021
//@ [e2024] edition: 2024

fn temp() {}

fn main() {
    // In Rust 2024, block tail expressions are temporary scopes, so the result of `temp()` is
    // dropped after evaluating `&temp()`.
    println!("{:?}", { &temp() });
    //[e2024]~^ ERROR: temporary value dropped while borrowed [E0716]

    // Arguments to function calls aren't extending expressions, so `temp()` is dropped at the end
    // of the block in Rust 2024.
    println!("{:?}", { std::convert::identity(&temp()) });
    //[e2024]~^ ERROR: temporary value dropped while borrowed [E0716]

    // In Rust 1.89, `format_args!` extended the lifetime of all extending expressions in its
    // arguments when provided with two or more arguments. This caused the result of `temp()` to
    // outlive the result of the block, making this compile.
    println!("{:?}{:?}", { &temp() }, ());
    //[e2024]~^ ERROR: temporary value dropped while borrowed [E0716]

    println!("{:?}{:?}", { std::convert::identity(&temp()) }, ());
    //[e2024]~^ ERROR: temporary value dropped while borrowed [E0716]

    // In real-world projects, this typically appeared in `if` expressions with a `&str` in one
    // branch and a reference to a `String` temporary in the other. Since the consequent and `else`
    // blocks of `if` expressions are temporary scopes in all editions, this affects Rust 2021 and
    // earlier as well.
    println!("{:?}{:?}", (), if true { &format!("") } else { "" });
    //~^ ERROR: temporary value dropped while borrowed [E0716]

    println!("{:?}{:?}", (), if true { std::convert::identity(&format!("")) } else { "" });
    //~^ ERROR: temporary value dropped while borrowed [E0716]

    // This has likewise occurred with `match`, affecting all editions.
    println!("{:?}{:?}", (), match true { true => &"" as &dyn std::fmt::Debug, false => &temp() });
    //~^ ERROR: temporary value dropped while borrowed [E0716]
}
