// known-bug
// failure-status: 101
// compile-flags: --edition=2021 --crate-type=lib
// rustc-env:RUST_BACKTRACE=0

// normalize-stderr-test "thread 'rustc' panicked.*" -> "thread 'rustc' panicked"
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""
// normalize-stderr-test "\nerror: internal compiler error.*\n\n" -> ""
// normalize-stderr-test "note:.*unexpectedly panicked.*\n\n" -> ""
// normalize-stderr-test "note: we would appreciate a bug report.*\n\n" -> ""
// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*\n\n" -> ""
// normalize-stderr-test "#.*\n" -> ""
// normalize-stderr-test ".*delayed.*\n" -> ""

// tracked in https://github.com/rust-lang/rust/issues/96572

#![feature(type_alias_impl_trait)]

fn main() {
    type T = impl Copy;
    let foo: T = (1u32, 2u32);
    let (a, b): (u32, u32) = foo;
}
