//@ check-pass
//@ run-rustfix

#![warn(unused_parens, unused_braces)]

fn foo<const N: usize>() {}

#[rustfmt::skip]
fn main() {
    //~v WARN unnecessary parentheses around `if` condition
    if (
        true // yay!
    ) {}

    if (/* a */ true) {}
    //~^ WARN unnecessary parentheses around `if` condition

    while (false /* b */) {}
    //~^ WARN unnecessary parentheses around `while` condition

    //~v WARN unnecessary parentheses around `if` condition
    if (
        true
    ) {}

    let _ = ( 1 );
    //~^ WARN unnecessary parentheses around assigned value

    let _: (/* c */ u8) = 1;
    //~^ WARN unnecessary parentheses around type

    let (/* d */ _x) = 1;
    //~^ WARN unnecessary parentheses around pattern

    foo::<{ /* e */ 7 }>();
    //~^ WARN unnecessary braces around const expression
}
