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
        _ => {}
    }

    match FOO_REF {
        FOO_REF => {}
        Foo(_) => {}
    }

    // This used to cause an ICE (https://github.com/rust-lang/rust/issues/78071)
    match FOO_REF_REF {
        FOO_REF_REF => {}
        Foo(_) => {}
    }

    match BAR {
        Bar => {}
        BAR => {}
        //~^ ERROR unreachable pattern
        _ => {}
        //~^ ERROR unreachable pattern
    }

    match BAR {
        BAR => {}
        Bar => {}
        //~^ ERROR unreachable pattern
        _ => {}
        //~^ ERROR unreachable pattern
    }

    match BAR {
        BAR => {}
        BAR => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
        _ => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
    }

    match BAZ {
        BAZ => {}
        Baz::Baz1 => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
        _ => {}
    }

    match BAZ {
        Baz::Baz1 => {}
        BAZ => {}
        //~^ ERROR unreachable pattern
        _ => {}
    }

    match BAZ {
        BAZ => {}
        Baz::Baz2 => {}
        _ => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
    }

    type Quux = fn(usize, usize) -> usize;
    fn quux(a: usize, b: usize) -> usize { a + b }
    const QUUX: Quux = quux;

    match QUUX {
        QUUX => {} //~ERROR behave unpredictably
        QUUX => {} //~ERROR behave unpredictably
        _ => {}
    }

    #[derive(PartialEq, Eq)]
    struct Wrap<T>(T);
    const WRAPQUUX: Wrap<Quux> = Wrap(quux);

    match WRAPQUUX {
        WRAPQUUX => {} //~ERROR behave unpredictably
        WRAPQUUX => {} //~ERROR behave unpredictably
        Wrap(_) => {}
    }

    match WRAPQUUX {
        Wrap(_) => {}
        WRAPQUUX => {} //~ERROR behave unpredictably
    }

    match WRAPQUUX {
        Wrap(_) => {}
    }

    match WRAPQUUX {
        WRAPQUUX => {} //~ERROR behave unpredictably
    }

    #[derive(PartialEq, Eq)]
    enum WhoKnows<T> {
        Yay(T),
        Nope,
    };
    const WHOKNOWSQUUX: WhoKnows<Quux> = WhoKnows::Yay(quux);

    match WHOKNOWSQUUX {
        WHOKNOWSQUUX => {} //~ERROR behave unpredictably
        WhoKnows::Yay(_) => {}
        WHOKNOWSQUUX => {} //~ERROR behave unpredictably
        WhoKnows::Nope => {}
    }
}
