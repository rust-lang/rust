#![warn(clippy::ignored_unit_patterns)]
#![allow(clippy::redundant_pattern_matching, clippy::single_match)]

fn foo() -> Result<(), ()> {
    unimplemented!()
}

fn main() {
    match foo() {
        Ok(_) => {},
        Err(_) => {},
    }
    if let Ok(_) = foo() {}
    let _ = foo().map_err(|_| todo!());
}
