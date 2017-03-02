- Feature Name: main_reexport
- Start Date: 2015-08-19
- RFC PR: https://github.com/rust-lang/rfcs/pull/1260
- Rust Issue: https://github.com/rust-lang/rust/issues/28937

# Summary

Allow a re-export of a function as entry point `main`.

# Motivation

Functions and re-exports of functions usually behave the same way, but they do
not for the program entry point `main`. This RFC aims to fix this inconsistency.

The above mentioned inconsistency means that e.g. you currently cannot use a
library's exported function as your main function.

Example:

    pub mod foo {
        pub fn bar() {
            println!("Hello world!");
        }
    }
    use foo::bar as main;

Example 2:

    extern crate main_functions;
    pub use main_functions::rmdir as main;

See also https://github.com/rust-lang/rust/issues/27640 for the corresponding
issue discussion.

The `#[main]` attribute can also be used to change the entry point of the
generated binary. This is largely irrelevant for this RFC as this RFC tries to
fix an inconsistency with re-exports and directly defined functions.
Nevertheless, it can be pointed out that the `#[main]` attribute does not cover
all the above-mentioned use cases.

# Detailed design

Use the symbol `main` at the top-level of a crate that is compiled as a program
(`--crate-type=bin`) – instead of explicitly only accepting directly-defined
functions, also allow (possibly non-`pub`) re-exports.

# Drawbacks

None.

# Alternatives

None.

# Unresolved questions

None.
