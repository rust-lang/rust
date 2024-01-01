// revisions: or1 or2 or3 or4 or5
// [or1] build-pass
// [or2] build-pass
// [or5] build-pass

#![allow(unreachable_patterns)]






fn foo() {
    let x = "foo";
    match x {
        x @ ((("h" | "ho" | "yo" | ("dude" | "w")) | "no" | "nop") | ("hey" | "gg")) |
        x @ ("black" | "pink") |
        x @ ("red" | "blue") => {
        }
        _ => (),
    }
}

fn bar() {
    let x = "foo";
    match x {
        x @ ("foo" | "bar") |
        (x @ "red" | (x @ "blue" | x @ "red")) => {
        }
        _ => (),
    }
}

#[cfg(or3)]
fn zot() {
    let x = "foo";
    match x {
        x @ ((("h" | "ho" | "yo" | ("dude" | "w")) | () | "nop") | ("hey" | "gg")) |
        //[or3]~^ ERROR mismatched types
        x @ ("black" | "pink") |
        x @ ("red" | "blue") => {
        }
        _ => (),
    }
}


#[cfg(or4)]
fn hey() {
    let x = "foo";
    match x {
        x @ ("foo" | "bar") |
        (x @ "red" | (x @ "blue" |  "red")) => {
        //[or4]~^ variable `x` is not bound in all patterns
        }
        _ => (),
    }
}

fn don() {
    enum Foo {
        A,
        B,
        C,
    }

    match Foo::A {
        | _foo @ (Foo::A | Foo::B) => {}
        Foo::C => {}
    };
}

fn main(){}
