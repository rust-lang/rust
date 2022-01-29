#![allow(dead_code)]

fn missing_discourses() -> Result<isize, ()> {
    Ok(1)
}

fn forbidden_narratives() -> Result<isize, ()> {
    missing_discourses()?
    //~^ ERROR: `?` operator has incompatible types
}

fn main() {}
