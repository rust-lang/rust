//@ check-pass
//@ ignore-windows
//@ compile-flags: -Cremark=foo --error-format=human --color=always

// FIXME(#61117): Respect debuginfo-level-tests, do not force debuginfo-level=0
//@ compile-flags: -Cdebuginfo=0

fn main() {}
