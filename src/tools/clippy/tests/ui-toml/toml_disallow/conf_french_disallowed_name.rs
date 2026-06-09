#![allow(dead_code)]
#![allow(clippy::single_match)]
#![allow(unused_variables)]
#![warn(clippy::disallowed_names)]

fn test(toto: ()) {}
//~^ disallowed_names

fn main() {
    let toto = 42;
    //~^ disallowed_names
    let tata = 42;
    //~^ disallowed_names
    let titi = 42;
    //~^ disallowed_names

    let tatab = 42;
    let tatatataic = 42;

    match (42, Some(1337), Some(0)) {
        (toto, Some(tata), titi @ Some(_)) => (),
        //~^ disallowed_names
        //~| disallowed_names
        //~| disallowed_names
        _ => (),
    }
}
