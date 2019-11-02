//! Definitions shared by macros / syntax extensions and e.g. librustc.

use crate::ast::Attribute;
use syntax_pos::symbol::sym;

pub mod allocator;

bitflags::bitflags! {
    /// Built-in derives that need some extra tracking beyond the usual macro functionality.
    #[derive(Default)]
    pub struct SpecialDerives: u8 {
        const PARTIAL_EQ = 1 << 0;
        const EQ         = 1 << 1;
        const COPY       = 1 << 2;
    }
}

pub fn is_proc_macro_attr(attr: &Attribute) -> bool {
    [sym::proc_macro, sym::proc_macro_attribute, sym::proc_macro_derive]
        .iter().any(|kind| attr.check_name(*kind))
}
