// aux-build:invalid-punct-ident.rs
// rustc-env:RUST_BACKTRACE=0

// FIXME https://github.com/rust-lang/rust/issues/59998
// normalize-stderr-test "thread.*panicked.*proc_macro_server.rs.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""
// normalize-stderr-test "\nerror: internal compiler error.*\n\n" -> ""
// normalize-stderr-test "note:.*unexpectedly panicked.*\n\n" -> ""
// normalize-stderr-test "note: we would appreciate a bug report.*\n\n" -> ""
// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*\n\n" -> ""
// normalize-stderr-test "query stack during panic:\n" -> ""
// normalize-stderr-test "we're just showing a limited slice of the query stack\n" -> ""
// normalize-stderr-test "end of query stack\n" -> ""

#[macro_use]
extern crate invalid_punct_ident;

invalid_punct!(); //~ ERROR proc macro panicked

fn main() {}
