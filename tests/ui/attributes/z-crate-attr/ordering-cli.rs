// Multiple `-Zcrate-attr` flags apply in command-line order, so the later `deny` wins.
//@ compile-flags: -Zcrate-attr=warn(unused) -Zcrate-attr=deny(unused)

fn foo() {} //~ ERROR function `foo` is never used

fn main() {}
