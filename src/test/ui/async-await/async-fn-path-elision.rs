// edition:2018
#![deny(elided_lifetimes_in_paths)]

struct HasLifetime<'a>(&'a bool);

async fn error(lt: HasLifetime) { //~ ERROR hidden lifetime parameters in types are deprecated
    if *lt.0 {}
}

fn also_error(lt: HasLifetime) { //~ ERROR hidden lifetime parameters in types are deprecated
    if *lt.0 {}
}

fn main() {}
