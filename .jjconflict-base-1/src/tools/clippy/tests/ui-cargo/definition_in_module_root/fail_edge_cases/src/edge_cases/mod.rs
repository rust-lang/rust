#![warn(clippy::definition_in_module_root)]

// Case g: extern crate is a declaration → no fire.
extern crate alloc;

// Case b: re-export of named item from sibling → no fire.
pub use super::other::Bar;

// Case c: glob re-export → no fire.
pub use super::other::*;

// Case j (top half): item right above `mod foo;` → fires.
pub struct AboveMod;

// Case j: nested file module declaration → no fire (declaration only).
mod sub;

// Case a: derive — struct fires once, derived impls do NOT fire (from_expansion).
#[derive(Clone, Debug)]
pub struct Foo;

// Case d: generics — struct fires + impl block fires.
pub struct Generic<T>(T);
impl<T> Generic<T> {
    pub fn new(t: T) -> Self {
        Self(t)
    }
}

// Case e: async fn fires as 'function'.
pub async fn frob() {}

// Case f: const fn fires as 'function'.
pub const fn cnst() -> u32 {
    0
}

// Case h: static fires as 'static'.
pub static FOO: u32 = 1;

// Case k: non-exported macro fires as 'macro'.
#[allow(unused_macros)]
macro_rules! local_macro {
    () => {};
}

// Case l: trait with default method body fires once (not its inner items).
pub trait WithDefault {
    fn default_method(&self) -> u32 {
        42
    }
}

// Case i: inline module — children should NOT fire.
#[cfg(test)]
mod tests {
    pub struct InsideInline;
    pub fn t() {}
}
