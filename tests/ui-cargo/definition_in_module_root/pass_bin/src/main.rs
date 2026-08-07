#![warn(clippy::definition_in_module_root)]

// Definitions in main.rs are allowed — the lint only applies to mod.rs.
pub struct Foo;
pub fn bar() {}
pub const BAZ: u32 = 1;

fn main() {}
