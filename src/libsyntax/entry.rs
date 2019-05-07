use crate::attr;
use crate::ast::{Item, ItemKind};
use crate::symbol::sym;

pub enum EntryPointType {
    None,
    MainNamed,
    MainAttr,
    Start,
    OtherMain, // Not an entry point, but some other function named main
}

// Beware, this is duplicated in librustc/middle/entry.rs, make sure to keep
// them in sync.
pub fn entry_point_type(item: &Item, depth: usize) -> EntryPointType {
    match item.node {
        ItemKind::Fn(..) => {
            if attr::contains_name(&item.attrs, sym::start) {
                EntryPointType::Start
            } else if attr::contains_name(&item.attrs, sym::main) {
                EntryPointType::MainAttr
            } else if item.ident.name == sym::main {
                if depth == 1 {
                    // This is a top-level function so can be 'main'
                    EntryPointType::MainNamed
                } else {
                    EntryPointType::OtherMain
                }
            } else {
                EntryPointType::None
            }
        }
        _ => EntryPointType::None,
    }
}
