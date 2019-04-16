// aux-build:invalid-punct-ident.rs

// FIXME https://github.com/rust-lang/rust/issues/59998
// normalize-stderr-test "thread.*panicked.*proc_macro_server.rs.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""

#[macro_use]
extern crate invalid_punct_ident;

invalid_punct!(); //~ ERROR proc macro panicked
