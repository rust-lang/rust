// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc_data_structures::fx::FxHashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};

use CrateCtxt;

/// Enforce that we do not have two items in an impl with the same name.
pub fn enforce_impl_items_are_distinct<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                                 impl_item_ids: &[hir::ImplItemId])
{
    let tcx = ccx.tcx;
    let mut seen_type_items = FxHashMap();
    let mut seen_value_items = FxHashMap();
    for &impl_item_id in impl_item_ids {
        let impl_item = tcx.map.impl_item(impl_item_id);
        let seen_items = match impl_item.node {
            hir::ImplItemKind::Type(_) => &mut seen_type_items,
            _                    => &mut seen_value_items,
        };
        match seen_items.entry(impl_item.name) {
            Occupied(entry) => {
                let mut err = struct_span_err!(tcx.sess, impl_item.span, E0201,
                                               "duplicate definitions with name `{}`:",
                                               impl_item.name);
                err.span_label(*entry.get(),
                               &format!("previous definition of `{}` here",
                                        impl_item.name));
                err.span_label(impl_item.span, &format!("duplicate definition"));
                err.emit();
            }
            Vacant(entry) => {
                entry.insert(impl_item.span);
            }
        }
    }
}
