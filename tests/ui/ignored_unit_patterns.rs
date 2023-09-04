#![warn(clippy::ignored_unit_patterns)]
#![allow(clippy::redundant_pattern_matching, clippy::single_match)]

fn foo() -> Result<(), ()> {
    unimplemented!()
}

fn main() {
    match foo() {
        Ok(_) => {},  //~ ERROR: matching over `()` is more explicit
        Err(_) => {}, //~ ERROR: matching over `()` is more explicit
    }
    if let Ok(_) = foo() {}
    //~^ ERROR: matching over `()` is more explicit
    let _ = foo().map_err(|_| todo!());
    //~^ ERROR: matching over `()` is more explicit
}
