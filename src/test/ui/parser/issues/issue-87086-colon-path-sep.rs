// Tests that a suggestion is issued if the user wrote a colon instead of
// a path separator in a match arm.

enum Foo {
    Bar,
    Baz,
}

fn f() -> Foo { Foo::Bar }

fn g1() {
    match f() {
        Foo:Bar => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
    match f() {
        Foo::Bar:Baz => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
    match f() {
        Foo:Bar::Baz => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
    match f() {
        Foo: Bar::Baz if true => {}
        //~^ ERROR: expected one of
        //~| HELP: maybe write a path separator here
        _ => {}
    }
    if let Bar:Baz = f() {
    //~^ ERROR: expected one of
    //~| HELP: maybe write a path separator here
    }
}

fn g1_neg() {
    match f() {
        ref Foo: Bar::Baz => {}
        //~^ ERROR: expected one of
        _ => {}
    }
}

fn g2_neg() {
    match f() {
        mut Foo: Bar::Baz => {}
        //~^ ERROR: expected one of
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
}
