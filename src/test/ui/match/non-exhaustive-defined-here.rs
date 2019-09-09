// Test the "defined here" and "not covered" diagnostic hints.
// We also make sure that references are peeled off from the scrutinee type
// so that the diagnostics work better with default binding modes.

#[derive(Clone)]
enum E {
//~^ `E` defined here
//~| `E` defined here
//~| `E` defined here
//~| `E` defined here
//~| `E` defined here
//~| `E` defined here
    A,
    B,
    //~^ not covered
    //~| not covered
    //~| not covered
    //~| not covered
    //~| not covered
    //~| not covered
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
    match e1 { //~ ERROR non-exhaustive patterns: `B` and `C` not covered
        E::A => {}
    }

    let E::A = e; //~ ERROR refutable pattern in local binding: `B` and `C` not covered
}

fn by_ref_once(e: &E) {
    match e { //~ ERROR non-exhaustive patterns: `&B` and `&C` not covered
        E::A => {}
    }

    let E::A = e; //~ ERROR refutable pattern in local binding: `&B` and `&C` not covered
}

fn by_ref_thrice(e: & &mut &E) {
    match e { //~ ERROR non-exhaustive patterns: `&&mut &B` and `&&mut &C` not covered
        E::A => {}
    }

    let E::A = e;
    //~^ ERROR refutable pattern in local binding: `&&mut &B` and `&&mut &C` not covered
}

enum Opt {
//~^ `Opt` defined here
//~| `Opt` defined here
    Some(u8),
    None,
    //~^ not covered
}

fn ref_pat(e: Opt) {
    match e {//~ ERROR non-exhaustive patterns: `None` not covered
        Opt::Some(ref _x) => {}
    }

    let Opt::Some(ref _x) = e; //~ ERROR refutable pattern in local binding: `None` not covered
}

fn main() {}
