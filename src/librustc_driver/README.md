The `driver` crate is effectively the "main" function for the rust
compiler.  It orchestrates the compilation process and "knits together"
the code from the other crates within rustc. This crate itself does
not contain any of the "main logic" of the compiler (though it does
have some code related to pretty printing or other minor compiler
options).

For more information about how the driver works, see the [rustc dev guide].

[rustc dev guide]: https://rustc-dev-guide.rust-lang.org/rustc-driver.html
