#![feature(custom_inner_attributes)]
#![rustfmt::skip]
// Test the "defined here" and "not covered" diagnostic hints.
// We also make sure that references are peeled off from the scrutinee type
// so that the diagnostics work better with default binding modes.

#[derive(Clone)]
enum E {
    //~^ NOTE `E` defined here
    //~| NOTE `E` defined here
    //~| NOTE `E` defined here
    //~| NOTE
    //~| NOTE
    //~| NOTE
    //~| NOTE
    //~| NOTE
    //~| NOTE
    A,
    B,
    //~^ NOTE  not covered
    //~| NOTE  not covered
    //~| NOTE  not covered
    //~| NOTE  not covered
    //~| NOTE  not covered
    //~| NOTE  not covered
    C
    //~^ NOTE not covered
    //~| NOTE not covered
    //~| NOTE not covered
    //~| NOTE not covered
    //~| NOTE not covered
    //~| NOTE not covered
}

fn by_val(e: E) {
    let e1 = e.clone();
    match e1 { //~ ERROR non-exhaustive patterns: `E::B` and `E::C` not covered
        //~^ NOTE patterns `E::B` and `E::C` not covered
        //~| NOTE the matched value is of type `E`
        E::A => {}
    }

    let E::A = e;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE patterns `E::B` and `E::C` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with
    //~| NOTE for more information, visit https://doc.rust-lang.org/book/ch19-02-refutability.html
    //~| NOTE the matched value is of type `E`
}

fn by_ref_once(e: &E) {
    match e {
    //~^ ERROR non-exhaustive patterns
    //~| NOTE patterns `&E::B` and `&E::C` not covered
    //~| NOTE the matched value is of type `&E`
        E::A => {}
    }

    let E::A = e;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE patterns `&E::B` and `&E::C` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with
    //~| NOTE for more information, visit https://doc.rust-lang.org/book/ch19-02-refutability.html
    //~| NOTE the matched value is of type `&E`
}

fn by_ref_thrice(e: & &mut &E) {
    match e {
    //~^ ERROR non-exhaustive patterns
    //~| NOTE patterns `&&mut &E::B` and `&&mut &E::C` not covered
    //~| NOTE the matched value is of type `&&mut &E`
        E::A => {}
    }

    let E::A = e;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE patterns `&&mut &E::B` and `&&mut &E::C` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with
    //~| NOTE for more information, visit https://doc.rust-lang.org/book/ch19-02-refutability.html
    //~| NOTE the matched value is of type `&&mut &E`
}

enum Opt {
    //~^ NOTE `Opt` defined here
    //~| NOTE
    //~| NOTE
    Some(u8),
    None,
    //~^ NOTE not covered
    //~| NOTE not covered
}

fn ref_pat(e: Opt) {
    match e {
        //~^ ERROR non-exhaustive patterns
        //~| NOTE pattern `Opt::None` not covered
        //~| NOTE the matched value is of type `Opt`
        Opt::Some(ref _x) => {}
    }

    let Opt::Some(ref _x) = e;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE the matched value is of type `Opt`
    //~| NOTE pattern `Opt::None` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with
    //~| NOTE for more information, visit https://doc.rust-lang.org/book/ch19-02-refutability.html
}

fn main() {}
