//@ check-pass
//@ edition:2018
#![warn(semicolon_in_expressions_from_macros)]

macro_rules! inner {
    () => {
        true; //~  WARN trailing semicolon in macro
              //~| WARN this was previously accepted
    };
}

macro_rules! outer {
    () => {{
        let _ = inner!();
    }};
}

fn main() {
    outer!()
}
