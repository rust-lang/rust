// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

// error-pattern: cannot find a built-in macro with name `line`

#![feature(rustc_attrs)]

#[rustc_builtin_macro]
macro_rules! unknown { () => () } //~ ERROR cannot find a built-in macro with name `unknown`

#[rustc_builtin_macro]
macro_rules! line { () => () }

fn main() {
    line!();
    std::prelude::v1::line!();
}
