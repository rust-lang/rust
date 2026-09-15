//! Regression test for https://github.com/rust-lang/rust/issues/147877.
//! Recover shorthand struct literals in expression contexts without changing statement parsing.
//@ run-rustfix

fn consume(_: u32) {}

fn main() {
    condition();
    call_argument();
    while_expression();
    adjacent_expressions(false);
}

fn condition() {
    let foo = Foo { x: 3 };
    let x = 3;
    let _ = if foo == Foo { x } { x } else { 0 };
    //~^ ERROR struct literals are not allowed here
}

fn call_argument() {
    let foo = Foo { x: 3 };
    let x = 3;
    consume(if &foo == &Foo { x } { x } else { 0 });
    //~^ ERROR struct literals are not allowed here
}

fn while_expression() {
    let foo = Foo { x: 3 };
    let x = 3;
    let _ = while foo == Foo { x } {};
    //~^ ERROR struct literals are not allowed here
}

fn adjacent_expressions(cond: bool) {
    let unit = ();
    if cond { unit }
    { println!() }
}

#[derive(Eq, PartialEq)]
struct Foo {
    x: u32,
}
