//@ compile-flags:-D bogus
//@ check-pass

//@ error-pattern:requested on the command line with `-D bogus`
//@ error-pattern:`#[warn(unknown_lints)]` on by default

fn main() {}

//~? WARN unknown lint: `bogus`
//~? WARN unknown lint: `bogus`
//~? WARN unknown lint: `bogus`
