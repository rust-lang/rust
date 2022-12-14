// Test the "defined here" and "not covered" diagnostic hints.
// We also make sure that references are peeled off from the scrutinee type
// so that the diagnostics work better with default binding modes.

#[derive(Clone)]
enum E {
    //~^ NOTE
    //~| NOTE
    //~| NOTE
    //~| NOTE
    //~| NOTE
    //~| NOTE
    A,
    B,
    //~^ NOTE `E` defined here
    //~| NOTE `E` defined here
    //~| NOTE `E` defined here
    //~| NOTE `E` defined here
    //~| NOTE `E` defined here
    //~| NOTE `E` defined here
    //~| NOTE  not covered
    //~| NOTE  not covered
    //~| NOTE  not covered
    //~| NOTE  not covered
    //~| NOTE  not covered
    //~| NOTE  not covered
    C
    //~^ not covered
    //~| not covered
    //~| not covered
    //~| not covered
    //~| not covered
    //~| not covered
}

fn by_val(e: E) {
    let e1 = e.clone();
    match e1 { //~ ERROR non-exhaustive patterns: `E::B` and `E::C` not covered
        //~^ NOTE patterns `E::B` and `E::C` not covered
        //~| NOTE the matched value is of type `E`
        E::A => {}
    }

    let E::A = e; //~ ERROR refutable pattern in local binding: `E::B` and `E::C` not covered
    //~^ NOTE patterns `E::B` and `E::C` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with
    //~| NOTE for more information, visit https://doc.rust-lang.org/book/ch18-02-refutability.html
    //~| NOTE the matched value is of type `E`
}

fn by_ref_once(e: &E) {
    match e { //~ ERROR non-exhaustive patterns: `&E::B` and `&E::C` not covered
    //~^ NOTE patterns `&E::B` and `&E::C` not covered
    //~| NOTE the matched value is of type `&E`
        E::A => {}
    }

    let E::A = e; //~ ERROR refutable pattern in local binding: `&E::B` and `&E::C` not covered
    //~^ NOTE patterns `&E::B` and `&E::C` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with
    //~| NOTE for more information, visit https://doc.rust-lang.org/book/ch18-02-refutability.html
    //~| NOTE the matched value is of type `&E`
}

fn by_ref_thrice(e: & &mut &E) {
    match e { //~ ERROR non-exhaustive patterns: `&&mut &E::B` and `&&mut &E::C` not covered
    //~^ NOTE patterns `&&mut &E::B` and `&&mut &E::C` not covered
    //~| NOTE the matched value is of type `&&mut &E`
        E::A => {}
    }

    let E::A = e;
    //~^ ERROR refutable pattern in local binding: `&&mut &E::B` and `&&mut &E::C` not covered
    //~| NOTE patterns `&&mut &E::B` and `&&mut &E::C` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with
    //~| NOTE for more information, visit https://doc.rust-lang.org/book/ch18-02-refutability.html
    //~| NOTE the matched value is of type `&&mut &E`
}

enum Opt {
    //~^ NOTE
    //~| NOTE
    Some(u8),
    None,
    //~^ NOTE `Opt` defined here
    //~| NOTE `Opt` defined here
    //~| NOTE not covered
    //~| NOTE not covered
}

fn ref_pat(e: Opt) {
    match e {//~ ERROR non-exhaustive patterns: `Opt::None` not covered
        //~^ NOTE pattern `Opt::None` not covered
        //~| NOTE the matched value is of type `Opt`
        Opt::Some(ref _x) => {}
    }

    let Opt::Some(ref _x) = e; //~ ERROR refutable pattern in local binding: `Opt::None` not covered
    //~^ NOTE the matched value is of type `Opt`
    //~| NOTE pattern `Opt::None` not covered
    //~| NOTE `let` bindings require an "irrefutable pattern", like a `struct` or an `enum` with
    //~| NOTE for more information, visit https://doc.rust-lang.org/book/ch18-02-refutability.html
}

fn main() {}
