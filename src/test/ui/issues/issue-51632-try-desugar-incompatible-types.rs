// run-rustfix

#![allow(dead_code)]

fn missing_discourses() -> Result<isize, ()> {
    Ok(1)
}

fn forbidden_narratives() -> Result<isize, ()> {
    missing_discourses()?
    //~^ ERROR try expression alternatives have incompatible types
    //~| HELP try wrapping with a success variant
}

fn main() {}
