NB: This crate is part of the Rust compiler. For an overview of the
compiler as a whole, see
[the README.md file found in `librustc`](../librustc/README.md).

The `driver` crate is effectively the "main" function for the rust
compiler.  It orchestrates the compilation process and "knits together"
the code from the other crates within rustc. This crate itself does
not contain any of the "main logic" of the compiler (though it does
have some code related to pretty printing or other minor compiler
options).


