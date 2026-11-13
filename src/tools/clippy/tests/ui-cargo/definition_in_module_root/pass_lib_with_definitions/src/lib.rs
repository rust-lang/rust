#![warn(clippy::definition_in_module_root)]

// Definitions in lib.rs are allowed — the lint only applies to mod.rs.
pub struct Foo;
pub enum Bar {
    A,
    B,
}
pub fn baz() {}
pub const QUUX: u32 = 1;
pub trait Trait1 {}
pub type Alias<T> = std::result::Result<T, String>;
