// Tests that a suggestion is issued if the user wrote a colon instead of
// a path separator in a match arm.

mod qux {
    pub enum Foo {
        Bar,
        Baz,
    }
}

use qux::Foo;

fn f() -> Foo { Foo::Bar }

fn g1() {
    match f() {
        Foo:Bar => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
    match f() {
        qux::Foo:Bar => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
    match f() {
        qux:Foo::Baz => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
    match f() {
        qux: Foo::Baz if true => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
    if let Foo:Bar = f() { //~ WARN: irrefutable `if let` pattern
    //~^ ERROR: expected one of
    //~| HELP: maybe write a path separator here
    //~| HELP: consider replacing the `if let` with a `let`
    }
}

fn g1_neg() {
    match f() {
        ref qux: Foo::Baz => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
}

fn g2_neg() {
    match f() {
        mut qux: Foo::Baz => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
}

fn main() {
    let myfoo = Foo::Bar;
    match myfoo {
        Foo::Bar => {}
        Foo:Bar::Baz => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
    }
    match myfoo {
        Foo::Bar => {}
        Foo:Bar => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
    }
}
