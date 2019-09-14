// aux-build:invalid-punct-ident.rs

// FIXME https://github.com/rust-lang/rust/issues/59998
// normalize-stderr-test "thread.*panicked.*proc_macro_server.rs.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""
// normalize-stderr-test "\nerror: internal compiler error.*\n\n" -> ""
// normalize-stderr-test "note:.*unexpectedly panicked.*\n\n" -> ""
// normalize-stderr-test "note: we would appreciate a bug report.*\n\n" -> ""
// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*\n\n" -> ""

#[macro_use]
extern crate invalid_punct_ident;

invalid_raw_ident!(); //~ ERROR proc macro panicked
