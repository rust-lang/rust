// Ensure we reject `#![crate_name = ""]`.

#![crate_name = ""] //~ ERROR crate name must not be empty

fn main() {}
