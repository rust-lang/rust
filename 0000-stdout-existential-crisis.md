- Feature Name: stdout_existential_crisis
- Start Date: 2015-03-25
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

When calling `println!` it currently causes a panic if `stdout` does not exist. Change this to ignore this specific error and simply void the output.

# Motivation

On linux `stdout` almost always exists, so when people write games and turn off the terminal there is still an `stdout` that they write to. Then when getting the code to run on Windows, when the console is disabled, suddenly `stdout` doesn't exist and `println!` panicks. This behavior difference is frustrating to developers trying to move to Windows.

There is also precedent with C and C++. On both Linux and Windows, if `stdout` is closed or doesn't exist, neither platform will error when printing to the console.

# Detailed design

Change the internal implementation of `println!` `print!` `panic!` and `assert!` to not `panic!` when `stdout` or `stderr` doesn't exist. When getting `stdout` or `stderr` through the `std::io` methods, those versions should continue to return an error if `stdout` or `stderr` doesn't exist.

# Drawbacks

Hides an error from the user which we may want to expose and may lead to people missing panicks occuring in threads.

# Alternatives

* Make `println!` `print!` `panic!` `assert!` return errors that the user has to handle.
* Continue with the status quo and panic if `stdout` or `stderr` doesn't exist.

# Unresolved questions

* Should `std::io::stdout` return `Err` or `None` when there is no `stdout` instead of unconditionally returning `Stdout`?
