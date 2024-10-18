use rustc_span::Symbol;
use rustc_span::symbol::sym;

use crate::attr::{self, AttributeExt};

#[derive(Debug)]
pub enum EntryPointType {
    /// This function is not an entrypoint.
    None,
    /// This is a function called `main` at the root level.
    /// ```
    /// fn main() {}
    /// ```
    MainNamed,
    /// This is a function with the `#[rustc_main]` attribute.
    /// Used by the testing harness to create the test entrypoint.
    /// ```ignore (clashes with test entrypoint)
    /// #[rustc_main]
    /// fn main() {}
    /// ```
    RustcMainAttr,
    /// This is a function with the `#[start]` attribute.
    /// ```ignore (clashes with test entrypoint)
    /// #[start]
    /// fn main() {}
    /// ```
    Start,
    /// This function is **not** an entrypoint but simply named `main` (not at the root).
    /// This is only used for diagnostics.
    /// ```
    /// #[allow(dead_code)]
    /// mod meow {
    ///     fn main() {}
    /// }
    /// ```
    OtherMain,
}

pub fn entry_point_type(
    attrs: &[impl AttributeExt],
    at_root: bool,
    name: Option<Symbol>,
) -> EntryPointType {
    if attr::contains_name(attrs, sym::start) {
        EntryPointType::Start
    } else if attr::contains_name(attrs, sym::rustc_main) {
        EntryPointType::RustcMainAttr
    } else if let Some(name) = name
        && name == sym::main
    {
        if at_root {
            // This is a top-level function so it can be `main`.
            EntryPointType::MainNamed
        } else {
            EntryPointType::OtherMain
        }
    } else {
        EntryPointType::None
    }
}
