mod lib;
//~^ WARN found module declaration for lib.rs
//~| ERROR file not found for module `lib`
mod main;
//~^ WARN found module declaration for main.rs
//~| ERROR file not found for module `main`

fn main() {}
