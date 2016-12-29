#![feature(plugin)]
#![plugin(clippy)]

#![deny(short_circuit_statement)]

fn main() {
    f() && g();
    //~^ ERROR boolean short circuit operator
    //~| HELP replace it with
    //~| SUGGESTION if f() { g(); }
    f() || g();
    //~^ ERROR boolean short circuit operator
    //~| HELP replace it with
    //~| SUGGESTION if !f() { g(); }
    1 == 2 || g();
    //~^ ERROR boolean short circuit operator
    //~| HELP replace it with
    //~| SUGGESTION if !(1 == 2) { g(); }
}

fn f() -> bool {
    true
}

fn g() -> bool {
    false
}
