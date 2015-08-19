- Feature Name: main_reexport
- Start Date: 2015-08-19
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Allow a re-export of a function as entry point `main`.

# Motivation

Functions and re-exports of functions usually behave the same way, but they do
not for the program entry point `main`. This RFC aims to fix this inconsistency.

The above mentioned inconsistency means that e.g. you currently cannot use a
library's exported function as your main function.

# Detailed design

Use the symbol `main` at the top-level of a crate that is compiled as a program
(`--crate-type=bin`) – instead of explicitly only accepting directly-defined
functions, also allow re-exports.

# Drawbacks

None.

# Alternatives

None.

# Unresolved questions

None.
