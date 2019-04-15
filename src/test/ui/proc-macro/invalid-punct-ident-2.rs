// aux-build:invalid-punct-ident.rs

// FIXME https://github.com/rust-lang/rust/issues/59998
// normalize-stderr-windows "thread.*panicked.*proc_macro_server.rs.*\n" -> ""
// normalize-stderr-windows "note:.*RUST_BACKTRACE=1.*\n" -> ""

#[macro_use]
extern crate invalid_punct_ident;

invalid_ident!(); //~ ERROR proc macro panicked
