// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

// Regression test for #62894, shouldn't crash.
// error-pattern: this file contains an unclosed delimiter
// error-pattern: expected one of `(`, `[`, or `{`, found keyword `fn`

fn f() { assert_eq!(f(), (), assert_eq!(assert_eq!

fn main() {}
