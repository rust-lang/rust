use attr;
use ast::{Item, ItemKind};

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
            if attr::contains_name(&item.attrs, "start") {
                EntryPointType::Start
            } else if attr::contains_name(&item.attrs, "main") {
                EntryPointType::MainAttr
            } else if item.ident.name == "main" {
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
