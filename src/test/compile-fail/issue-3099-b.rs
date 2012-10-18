#[legacy_exports]
mod a {}

#[legacy_exports]
mod a {} //~ ERROR duplicate definition of type a

fn main() {}
