#[legacy_exports]
mod a {}

#[legacy_exports]
mod a {} //~ ERROR Duplicate definition of module a

fn main() {}
