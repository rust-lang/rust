// Issue #51976
#![deny(unused_variables)] //~ NOTE: the lint level is defined here
enum Lol {
    Foo,
    Bar,
}
const Bat: () = ();
const Battery: () = ();
struct Bay;

fn foo(x: (Lol, Lol)) {
    use Lol::*;
    match &x {
        (Foo, Bar) | (Ban, Foo) => {}
        //~^ ERROR: variable `Ban` is not bound in all patterns
        //~| HELP: you might have meant to use the similarly named previously used binding `Bar`
        //~| NOTE: pattern doesn't bind `Ban`
        //~| NOTE: variable not in all patterns
        //~| ERROR: variable `Ban` is assigned to, but never used
        //~| NOTE: consider using `_Ban` instead
        //~| HELP: you might have meant to pattern match on the similarly named
        _ => {}
    }
    match &x {
        (Foo, _) | (Ban, Foo) => {}
        //~^ ERROR: variable `Ban` is not bound in all patterns
        //~| HELP: you might have meant to use the similarly named unit variant `Bar`
        //~| NOTE: pattern doesn't bind `Ban`
        //~| NOTE: variable not in all patterns
        //~| ERROR: variable `Ban` is assigned to, but never used
        //~| NOTE: consider using `_Ban` instead
        //~| HELP: you might have meant to pattern match on the similarly named
        _ => {}
    }
    match Some(42) {
        Some(_) => {}
        Non => {}
        //~^ ERROR: unused variable: `Non`
        //~| HELP: if this is intentional, prefix it with an underscore
        //~| HELP: you might have meant to pattern match on the similarly named
    }
    match Some(42) {
        Some(_) => {}
        Non | None => {}
        //~^ ERROR: unused variable: `Non`
        //~| HELP: if this is intentional, prefix it with an underscore
        //~| ERROR: variable `Non` is not bound in all patterns [E0408]
        //~| NOTE: pattern doesn't bind `Non`
        //~| NOTE: variable not in all patterns
        //~| HELP: you might have meant to use the similarly named unit variant `None`
        //~| HELP: you might have meant to pattern match on the similarly named
    }
    match Some(42) {
        Non | Some(_) => {}
        //~^ ERROR: unused variable: `Non`
        //~| HELP: if this is intentional, prefix it with an underscore
        //~| ERROR: variable `Non` is not bound in all patterns [E0408]
        //~| NOTE: pattern doesn't bind `Non`
        //~| NOTE: variable not in all patterns
        //~| HELP: you might have meant to use the similarly named unit variant `None`
        //~| HELP: you might have meant to pattern match on the similarly named
    }
}
fn bar(x: (Lol, Lol)) {
    use Lol::*;
    use ::Bat;
    use ::Bay;
    match &x {
        (Foo, _) | (Ban, Foo) => {}
        //~^ ERROR: variable `Ban` is not bound in all patterns
        //~| HELP: you might have meant to use the similarly named unit variant `Bar`
        //~| HELP: you might have meant to use the similarly named unit struct `Bay`
        //~| HELP: you might have meant to use the similarly named constant `Bat`
        //~| NOTE: pattern doesn't bind `Ban`
        //~| NOTE: variable not in all patterns
        //~| ERROR: variable `Ban` is assigned to, but never used
        //~| NOTE: consider using `_Ban` instead
        //~| HELP: you might have meant to pattern match on the similarly named
        _ => {}
    }
}
fn baz(x: (Lol, Lol)) {
    use Lol::*;
    use Bat;
    match &x {
        (Foo, _) | (Ban, Foo) => {}
        //~^ ERROR: variable `Ban` is not bound in all patterns
        //~| HELP: you might have meant to use the similarly named unit variant `Bar`
        //~| HELP: you might have meant to use the similarly named constant `Bat`
        //~| NOTE: pattern doesn't bind `Ban`
        //~| NOTE: variable not in all patterns
        //~| ERROR: variable `Ban` is assigned to, but never used
        //~| NOTE: consider using `_Ban` instead
        //~| HELP: you might have meant to pattern match on the similarly named
        _ => {}
    }
    match &x {
        (Ban, _) => {}
        //~^ ERROR: unused variable: `Ban`
        //~| HELP: if this is intentional, prefix it with an underscore
        //~| HELP: you might have meant to pattern match on the similarly named
    }
    match Bay {
        Ban => {}
        //~^ ERROR: unused variable: `Ban`
        //~| HELP: if this is intentional, prefix it with an underscore
        //~| HELP: you might have meant to pattern match on the similarly named
    }
    match () {
        Batery => {}
        //~^ ERROR: unused variable: `Batery`
        //~| HELP: if this is intentional, prefix it with an underscore
        //~| HELP: you might have meant to pattern match on the similarly named constant
    }
}

fn main() {
    use Lol::*;
    foo((Foo, Bar));
    bar((Foo, Bar));
    baz((Foo, Bar));
}
