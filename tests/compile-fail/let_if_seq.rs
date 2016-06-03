#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused_variables, unused_assignments, similar_names, blacklisted_name)]
#![deny(useless_let_if_seq)]

fn f() -> bool { true }

fn issue975() -> String {
    let mut udn = "dummy".to_string();
    if udn.starts_with("uuid:") {
        udn = String::from(&udn[5..]);
    }
    udn
}

fn early_return() -> u8 {
    // FIXME: we could extend the lint to include such cases:
    let foo;

    if f() {
        return 42;
    } else {
        foo = 0;
    }

    foo
}

fn main() {
    early_return();
    issue975();

    let mut foo = 0;
    //~^ ERROR `if _ { .. } else { .. }` is an expression
    //~| HELP more idiomatic
    //~| SUGGESTION let <mut> foo = if f() { 42 } else { 0 };
    if f() {
        foo = 42;
    }

    let mut bar = 0;
    //~^ ERROR `if _ { .. } else { .. }` is an expression
    //~| HELP more idiomatic
    //~| SUGGESTION let <mut> bar = if f() { ..; 42 } else { ..; 0 };
    if f() {
        f();
        bar = 42;
    }
    else {
        f();
    }

    let quz;
    //~^ ERROR `if _ { .. } else { .. }` is an expression
    //~| HELP more idiomatic
    //~| SUGGESTION let quz = if f() { 42 } else { 0 };

    if f() {
        quz = 42;
    } else {
        quz = 0;
    }

    // `toto` is used several times
    let mut toto;

    if f() {
        toto = 42;
    } else {
        for i in &[1, 2] {
            toto = *i;
        }

        toto = 2;
    }

    // baz needs to be mut
    let mut baz = 0;
    //~^ ERROR `if _ { .. } else { .. }` is an expression
    //~| HELP more idiomatic
    //~| SUGGESTION let <mut> baz = if f() { 42 } else { 0 };
    if f() {
        baz = 42;
    }

    baz = 1337;
}
