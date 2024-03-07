//! This test ICEs because the `repr(packed)` attributes
//! end up on the `Dealigned` struct's attribute list, but the
//! derive didn't see that.

//@known-bug: #120873
//@ failure-status: 101
//@ normalize-stderr-test "note: .*\n\n" -> ""
//@ normalize-stderr-test "thread 'rustc' panicked.*\n" -> ""
//@ normalize-stderr-test "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@ rustc-env:RUST_BACKTRACE=0

#[repr(packed)]
struct Dealigned<T>(u8, T);

#[derive(PartialEq)]
#[repr(C)]
struct Dealigned<T>(u8, T);

fn main() {}
