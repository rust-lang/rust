// aux-build:macro_crate_test.rs
// ignore-stage1

// Issue #15750: a macro that internally parses its input and then
// uses `quote_expr!` to rearrange it should be hygiene-preserving.

#![feature(plugin)]
#![plugin(macro_crate_test)]

fn main() {
    let x = 3;
    assert_eq!(3, identity!(x));
    assert_eq!(6, identity!(x+x));
    let x = 4;
    assert_eq!(4, identity!(x));
}
