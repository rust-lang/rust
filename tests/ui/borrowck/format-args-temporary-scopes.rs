//! Test for #145784 as it relates to format arguments: arguments to macros such as `println!`
//! should obey normal temporary scoping rules.
//@ revisions: e2021 e2024
//@ [e2021] edition: 2021
//@ [e2024] edition: 2024

fn temp() {}

fn main() {
    // In Rust 2024, block tail expressions are temporary scopes, but temporary lifetime extension
    // rules apply: `&temp()` here is an extending borrow expression, so `temp()`'s lifetime is
    // extended past the block.
    println!("{:?}", { &temp() });

    // Arguments to function calls aren't extending expressions, so `temp()` is dropped at the end
    // of the block in Rust 2024.
    println!("{:?}", { std::convert::identity(&temp()) });
    //[e2024]~^ ERROR: temporary value dropped while borrowed [E0716]

    // In Rust 1.89, `format_args!` had different lifetime extension behavior dependent on how many
    // formatting arguments it had (#145880), so let's test that too.
    println!("{:?}{:?}", { &temp() }, ());

    println!("{:?}{:?}", { std::convert::identity(&temp()) }, ());
    //[e2024]~^ ERROR: temporary value dropped while borrowed [E0716]

    // In real-world projects, this typically appeared in `if` expressions with a `&str` in one
    // branch and a reference to a `String` temporary in the other. Since the consequent and `else`
    // blocks of `if` expressions are temporary scopes in all editions, this affects Rust 2021 and
    // earlier as well.
    println!("{:?}{:?}", (), if true { &format!("") } else { "" });

    println!("{:?}{:?}", (), if true { std::convert::identity(&format!("")) } else { "" });
    //~^ ERROR: temporary value dropped while borrowed [E0716]

    // This has likewise occurred with `match`, affecting all editions.
    println!("{:?}{:?}", (), match true { true => &"" as &dyn std::fmt::Debug, false => &temp() });
}
