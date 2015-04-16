- Feature Name: stdout_existential_crisis
- Start Date: 2015-03-25
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

When calling `println!` it currently causes a panic if `stdout` does not exist. Change this to ignore this specific error and simply void the output.

# Motivation

On Linux `stdout` almost always exists, so when people write games and turn off the terminal there is still an `stdout` that they write to. Then when getting the code to run on Windows, when the console is disabled, suddenly `stdout` doesn't exist and `println!` panicks. This behavior difference is frustrating to developers trying to move to Windows.

There is also precedent with C and C++. On both Linux and Windows, if `stdout` is closed or doesn't exist, neither platform will error when attempting to print to the console.

# Detailed design

When using any of the convenience macros that write to either `stdout` or `stderr`, such as `println!` `print!` `panic!` and `assert!`, change the implementation to ignore the specific error of `stdout` or `stderr` not existing. The behavior of all other errors will be unaffected. This can be implemented by redirecting `stdout` and `stderr` to `std::io::sink` if the original handles do not exist.

Update the methods `std::io::stdin` `std::io::stdout` and `std::io::stderr` (and any raw versions of these) to return a `Result`. If their respective handles do not exist, then return `Err`.

# Drawbacks

* Hides an error from the user which we may want to expose and may lead to people missing panicks occuring in threads.
* Some languages, such as Ruby and Python, do throw an exception when stdout is missing.

# Alternatives

* Make `println!` `print!` `panic!` `assert!` return errors that the user has to handle. This would lose a large part of the convenience of these macros.
* Continue with the status quo and panic if `stdout` or `stderr` doesn't exist.
* For `std::io::stdin` `std::io::stdout` and `std::io::stderr`, make them return the equivalent of `std::io::empty` or `std::io::sink` if their respective handles don't exist. This leaves people unable to explicitly handle the case of them not existing, but has the advantage of not breaking stable signatures.
** Or they could simply error upon attempting to write to/read from the handles.

# Unresolved questions

* Which is better? Breaking the signatures of those three methods in `std::io`, making them silently redirect to `empty`/`sink`, or erroring upon attempting to write to/read from the handle?
