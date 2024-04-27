use crate::{attr, Attribute};
use rustc_span::symbol::sym;
use rustc_span::Symbol;

#[derive(Debug)]
pub enum EntryPointType {
    None,
    MainNamed,
    RustcMainAttr,
    Start,
    OtherMain, // Not an entry point, but some other function named main
}

pub fn entry_point_type(
    attrs: &[Attribute],
    at_root: bool,
    name: Option<Symbol>,
) -> EntryPointType {
    if attr::contains_name(attrs, sym::start) {
        EntryPointType::Start
    } else if attr::contains_name(attrs, sym::rustc_main) {
        EntryPointType::RustcMainAttr
    } else {
        if let Some(name) = name
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
}
