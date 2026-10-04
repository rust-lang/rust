#![feature(decl_macro)]

// Glob import in macro namespace
mod inner { pub macro mac() {} }
pub use inner::*;

// Macro-expanded single import in macro namespace
macro_rules! define_mac {
    () => { pub macro mac() {} }
}
define_mac!();
