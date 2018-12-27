// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

use std::rc::Rc;

fn lub_short<'a, T>(_: &[&'a T], _: &[&'a T]) {}

// The two arguments are a subtype of their LUB, after coercion.
fn long_and_short<'a, T>(xs: &[&'static T; 1], ys: &[&'a T; 1]) {
    lub_short(xs, ys);
}

// The argument coerces to a subtype of the return type.
fn long_to_short<'a, 'b, T>(xs: &'b [&'static T; 1]) -> &'b [&'a T] {
    xs
}

// Rc<T> is covariant over T just like &T.
fn long_to_short_rc<'a, T>(xs: Rc<[&'static T; 1]>) -> Rc<[&'a T]> {
    xs
}

// LUB-coercion (if-else/match/array) coerces `xs: &'b [&'static T: N]`
// to a subtype of the LUB of `xs` and `ys` (i.e., `&'b [&'a T]`),
// regardless of the order they appear (in if-else/match/array).
fn long_and_short_lub1<'a, 'b, T>(xs: &'b [&'static T; 1], ys: &'b [&'a T]) {
    let _order1 = [xs, ys];
    let _order2 = [ys, xs];
}

// LUB-coercion should also have the exact same effect when `&'b [&'a T; N]`
// needs to be coerced, i.e., the resulting type is not &'b [&'static T], but
// rather the `&'b [&'a T]` LUB.
fn long_and_short_lub2<'a, 'b, T>(xs: &'b [&'static T], ys: &'b [&'a T; 1]) {
    let _order1 = [xs, ys];
    let _order2 = [ys, xs];
}

fn main() {}
