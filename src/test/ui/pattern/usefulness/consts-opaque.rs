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
    Baz2
}
impl Eq for Baz {}
const BAZ: Baz = Baz::Baz1;

fn main() {
    match FOO {
        FOO => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
    }

    match FOO_REF {
        FOO_REF => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        Foo(_) => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
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
        BAR => {} // should not be emitting unreachable warning
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        //~| ERROR unreachable pattern
        _ => {}
        //~^ ERROR unreachable pattern
    }

    match BAR {
        BAR => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        Bar => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
        _ => {}
        //~^ ERROR unreachable pattern
    }

    match BAR {
        BAR => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        BAR => {} // should not be emitting unreachable warning
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        //~| ERROR unreachable pattern
        _ => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
    }

    match BAZ {
        BAZ => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        Baz::Baz1 => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
        _ => {}
        //~^ ERROR unreachable pattern
    }

    match BAZ {
        Baz::Baz1 => {}
        BAZ => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => {}
        //~^ ERROR unreachable pattern
    }

    match BAZ {
        BAZ => {}
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        Baz::Baz2 => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
        _ => {} // should not be emitting unreachable warning
        //~^ ERROR unreachable pattern
    }

    type Quux = fn(usize, usize) -> usize;
    fn quux(a: usize, b: usize) -> usize { a + b }
    const QUUX: Quux = quux;

    match QUUX {
        QUUX => {}
        QUUX => {}
        _ => {}
    }

    #[derive(PartialEq, Eq)]
    struct Wrap<T>(T);
    const WRAPQUUX: Wrap<Quux> = Wrap(quux);

    match WRAPQUUX {
        WRAPQUUX => {}
        WRAPQUUX => {}
        Wrap(_) => {}
    }

    match WRAPQUUX {
        Wrap(_) => {}
        WRAPQUUX => {} // detected unreachable because we do inspect the `Wrap` layer
        //~^ ERROR unreachable pattern
    }

    #[derive(PartialEq, Eq)]
    enum WhoKnows<T> {
        Yay(T),
        Nope,
    };
    const WHOKNOWSQUUX: WhoKnows<Quux> = WhoKnows::Yay(quux);

    match WHOKNOWSQUUX {
        WHOKNOWSQUUX => {}
        WhoKnows::Yay(_) => {}
        WHOKNOWSQUUX => {} // detected unreachable because we do inspect the `WhoKnows` layer
        //~^ ERROR unreachable pattern
        WhoKnows::Nope => {}
    }
}
