// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use attr;
use ast::{Item, ItemFn};

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
        ItemFn(..) => {
            if attr::contains_name(&item.attrs, "start") {
                EntryPointType::Start
            } else if attr::contains_name(&item.attrs, "main") {
                EntryPointType::MainAttr
            } else if item.ident.name.as_str() == "main" {
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
