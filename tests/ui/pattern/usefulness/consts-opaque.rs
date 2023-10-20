// This file tests the exhaustiveness algorithm on opaque constants. Most of the examples give
// unnecessary warnings because const_to_pat.rs converts a constant pattern to a wildcard when the
// constant is not allowed as a pattern. This is an edge case so we may not care to fix it.
// See also https://github.com/rust-lang/rust/issues/78057

#![deny(unreachable_patterns)]

#[derive(PartialEq)]
struct Foo(i32);
impl Eq for Foo {}
const FOO: Foo = Foo(42);
const FOO_REF: &Foo = &Foo(42);
const FOO_REF_REF: &&Foo = &&Foo(42);

#[derive(PartialEq)]
struct Bar;
impl Eq for Bar {}
const BAR: Bar = Bar;

#[derive(PartialEq)]
enum Baz {
    Baz1,
    Baz2,
}
impl Eq for Baz {}
const BAZ: Baz = Baz::Baz1;

#[rustfmt::skip]
fn main() {
    match FOO {
        FOO => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => {}
    }

    match FOO_REF {
        FOO_REF => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        Foo(_) => {}
    }

    // This used to cause an ICE (https://github.com/rust-lang/rust/issues/78071)
    match FOO_REF_REF {
        FOO_REF_REF => {}
        //~^ WARNING must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARNING this was previously accepted by the compiler but is being phased out
        Foo(_) => {}
    }

    match BAR {
        Bar => {}
        BAR => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => {}
    }

    match BAR {
        BAR => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        Bar => {}
        _ => {}
    }

    match BAR {
        BAR => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        BAR => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => {}
    }

    match BAZ {
        BAZ => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        Baz::Baz1 => {}
        _ => {}
    }

    match BAZ {
        Baz::Baz1 => {}
        BAZ => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => {}
    }

    match BAZ {
        BAZ => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        Baz::Baz2 => {}
        _ => {}
    }

    type Quux = fn(usize, usize) -> usize;
    fn quux(a: usize, b: usize) -> usize { a + b }
    const QUUX: Quux = quux;

    match QUUX {
        QUUX => {} //~WARN behave unpredictably
        //~| previously accepted
        QUUX => {} //~WARN behave unpredictably
        //~| previously accepted
        _ => {}
    }

    #[derive(PartialEq, Eq)]
    struct Wrap<T>(T);
    const WRAPQUUX: Wrap<Quux> = Wrap(quux);

    match WRAPQUUX {
        WRAPQUUX => {} //~WARN behave unpredictably
        //~| previously accepted
        WRAPQUUX => {} //~WARN behave unpredictably
        //~| previously accepted
        Wrap(_) => {}
    }

    match WRAPQUUX {
        Wrap(_) => {}
        WRAPQUUX => {} //~WARN behave unpredictably
        //~| previously accepted
    }

    match WRAPQUUX {
        Wrap(_) => {}
    }

    match WRAPQUUX {
        //~^ ERROR: non-exhaustive patterns: `Wrap(_)` not covered
        WRAPQUUX => {} //~WARN behave unpredictably
        //~| previously accepted
    }

    #[derive(PartialEq, Eq)]
    enum WhoKnows<T> {
        Yay(T),
        Nope,
    };
    const WHOKNOWSQUUX: WhoKnows<Quux> = WhoKnows::Yay(quux);

    match WHOKNOWSQUUX {
        WHOKNOWSQUUX => {} //~WARN behave unpredictably
        //~| previously accepted
        WhoKnows::Yay(_) => {}
        WHOKNOWSQUUX => {} //~WARN behave unpredictably
        //~| previously accepted
        WhoKnows::Nope => {}
    }
}
