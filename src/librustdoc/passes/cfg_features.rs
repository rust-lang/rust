// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

use rustc::util::nodemap::FxHashSet;
use syntax::ast::{LitKind, MetaItem, MetaItemKind};
use syntax::symbol::InternedString;

use clean::{self, Item, AttributesExt, ConfigFeatureMap};
use fold::{self, DocFolder as Underscore};
use plugins;

pub fn cfg_features(krate: clean::Crate) -> plugins::PluginResult {
    let mut folder = DocFolder::new();
    let mut krate = folder.fold_crate(krate);
    krate.cfg_feature_map = folder.map;
    krate
}

struct DocFolder {
    map: ConfigFeatureMap,
    current: FxHashSet<InternedString>,
}

impl DocFolder {
    fn new() -> DocFolder {
        let map = ConfigFeatureMap::default();
        let set = FxHashSet::default();
        DocFolder {
            map: map,
            current: set,
        }
    }
}

impl fold::DocFolder for DocFolder {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let previous = self.current.clone();
        let introduced = {
            get_cfg_features_from_item(&item)
                .filter(|s| !self.current.contains(s))
                .collect::<FxHashSet<_>>()
        };
        self.current.extend(introduced.iter().cloned());
        let item = self.fold_item_recur(item);
        let current = mem::replace(&mut self.current, previous);
        if let Some(ref item) = item {
            if item.def_id.index.as_u32() != 0 {
                self.map.set_all(item.def_id, current);
                self.map.set_introduced(item.def_id, introduced);
            }
        }
        item
    }
}

fn get_cfg_features_from_item<'a>(item: &'a Item) -> impl Iterator<Item=InternedString> {
    let mut cfg_features = Vec::default();
    for attr in item.attrs.lists("cfg") {
        let attr = attr.meta_item().unwrap();
        get_cfg_features_from_attr(attr, &mut cfg_features);
    }
    cfg_features.into_iter()
}

/// Parses features out of the inner part of a cfg attr,
/// i.e. the `feature = "blah"` part of `#[cfg(feature = "blah")]`
///
/// Only handles simple cases and (recursively) `all` cases, anything using
/// `any` is ignored as it's unlikely to happen and give a useful result in
/// normal usage.
fn get_cfg_features_from_attr<'a>(attr: &'a MetaItem, cfg_features: &mut Vec<InternedString>) {
    match attr.node {
        MetaItemKind::List(ref attrs) if attr.name == "all" => {
            for attr in attrs.iter().filter_map(|a| a.meta_item()) {
                get_cfg_features_from_attr(attr, cfg_features);
            }
        }
        MetaItemKind::NameValue(ref value) if attr.name == "feature" => {
            if let LitKind::Str(ref value, _) = value.node {
                cfg_features.push(value.as_str());
            }
        }
        _ => (),
    }
}
