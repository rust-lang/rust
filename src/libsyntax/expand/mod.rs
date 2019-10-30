//! Definitions shared by macros / syntax extensions and e.g. librustc.

use crate::ast::Attribute;
use syntax_pos::symbol::sym;

pub mod allocator;

pub fn is_proc_macro_attr(attr: &Attribute) -> bool {
    [sym::proc_macro, sym::proc_macro_attribute, sym::proc_macro_derive]
        .iter().any(|kind| attr.check_name(*kind))
}
