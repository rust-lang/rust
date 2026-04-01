// https://github.com/rust-lang/rust/issues/51632
#![allow(dead_code)]

fn missing_discourses() -> Result<isize, ()> {
    Ok(1)
}

fn forbidden_narratives() -> Result<isize, ()> {
    missing_discourses()?
    //~^ ERROR: `?` operator has incompatible types
}

fn main() {}
