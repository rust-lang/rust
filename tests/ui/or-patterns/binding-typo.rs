// Issue #51976
//@ run-rustfix
#![allow(unused_variables)] // allowed so we don't get overlapping suggestions
enum Lol {
    Foo,
    Bar,
}

fn foo(x: (Lol, Lol)) {
    use Lol::*;
    match &x {
        (Foo, Bar) | (Ban, Foo) => {}
        //~^ ERROR: variable `Ban` is not bound in all patterns
        //~| HELP: you might have meant to use the similarly named previously used binding `Bar`
        //~| NOTE: pattern doesn't bind `Ban`
        //~| NOTE: variable not in all patterns
        _ => {}
    }
    match &x {
        (Foo, _) | (Ban, Foo) => {}
        //~^ ERROR: variable `Ban` is not bound in all patterns
        //~| HELP: you might have meant to use the similarly named unit variant `Bar`
        //~| NOTE: pattern doesn't bind `Ban`
        //~| NOTE: variable not in all patterns
        _ => {}
    }
}

fn main() {
    use Lol::*;
    foo((Foo, Bar));
}
