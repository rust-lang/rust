#![feature(plugin)]
#![plugin(clippy)]

#![deny(inline_always)]

#[inline(always)] //~ERROR you have declared `#[inline(always)]` on `test_attr_lint`.
fn test_attr_lint() {
    assert!(true)
}

#[inline(always)]
fn false_positive_expr() {
    unreachable!()
}

#[inline(always)]
fn false_positive_stmt() {
    unreachable!();
}

#[inline(always)]
fn empty_and_false_positive_stmt() {
    ;
    unreachable!();
}


fn main() {
    test_attr_lint();
    if false { false_positive_expr() }
    if false { false_positive_stmt() }
    if false { empty_and_false_positive_stmt() }
}
