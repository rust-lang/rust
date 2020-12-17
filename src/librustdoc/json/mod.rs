//! Rustdoc's JSON backend
//!
//! This module contains the logic for rendering a crate as JSON rather than the normal static HTML
//! output. See [the RFC](https://github.com/rust-lang/rfcs/pull/2963) and the [`types`] module
//! docs for usage and details.

mod conversions;
pub mod types;

use std::cell::RefCell;
use std::fs::File;
use std::path::PathBuf;
use std::rc::Rc;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_session::Session;
use rustc_span::edition::Edition;

use crate::clean;
use crate::config::{RenderInfo, RenderOptions};
use crate::error::Error;
use crate::formats::cache::Cache;
use crate::formats::FormatRenderer;
use crate::html::render::cache::ExternalLocation;

#[derive(Clone)]
crate struct JsonRenderer {
    sess: Lrc<Session>,
    /// A mapping of IDs that contains all local items for this crate which gets output as a top
    /// level field of the JSON blob.
    index: Rc<RefCell<FxHashMap<types::Id, types::Item>>>,
    /// The directory where the blob will be written to.
    out_path: PathBuf,
}

impl JsonRenderer {
    fn get_trait_implementors(
        &mut self,
        id: rustc_span::def_id::DefId,
        cache: &Cache,
    ) -> Vec<types::Id> {
        cache
            .implementors
            .get(&id)
            .map(|implementors| {
                implementors
                    .iter()
                    .map(|i| {
                        let item = &i.impl_item;
                        self.item(item.clone(), cache).unwrap();
                        item.def_id.into()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_impls(&mut self, id: rustc_span::def_id::DefId, cache: &Cache) -> Vec<types::Id> {
        cache
            .impls
            .get(&id)
            .map(|impls| {
                impls
                    .iter()
                    .filter_map(|i| {
                        let item = &i.impl_item;
                        if item.def_id.is_local() {
                            self.item(item.clone(), cache).unwrap();
                            Some(item.def_id.into())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_trait_items(&mut self, cache: &Cache) -> Vec<(types::Id, types::Item)> {
        cache
            .traits
            .iter()
            .filter_map(|(&id, trait_item)| {
                // only need to synthesize items for external traits
                if !id.is_local() {
                    trait_item.items.clone().into_iter().for_each(|i| self.item(i, cache).unwrap());
                    Some((
                        id.into(),
                        types::Item {
                            id: id.into(),
                            crate_id: id.krate.as_u32(),
                            name: cache
                                .paths
                                .get(&id)
                                .unwrap_or_else(|| {
                                    cache
                                        .external_paths
                                        .get(&id)
                                        .expect("Trait should either be in local or external paths")
                                })
                                .0
                                .last()
                                .map(Clone::clone),
                            visibility: types::Visibility::Public,
                            kind: types::ItemKind::Trait,
                            inner: types::ItemEnum::TraitItem(trait_item.clone().into()),
                            source: None,
                            docs: Default::default(),
                            links: Default::default(),
                            attrs: Default::default(),
                            deprecation: Default::default(),
                        },
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl FormatRenderer for JsonRenderer {
    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        _render_info: RenderInfo,
        _edition: Edition,
        _cache: &mut Cache,
        sess: Lrc<Session>,
    ) -> Result<(Self, clean::Crate), Error> {
        debug!("Initializing json renderer");
        Ok((
            JsonRenderer {
                sess,
                index: Rc::new(RefCell::new(FxHashMap::default())),
                out_path: options.output,
            },
            krate,
        ))
    }

    /// Inserts an item into the index. This should be used rather than directly calling insert on
    /// the hashmap because certain items (traits and types) need to have their mappings for trait
    /// implementations filled out before they're inserted.
    fn item(&mut self, item: clean::Item, cache: &Cache) -> Result<(), Error> {
        // Flatten items that recursively store other items
        item.kind.inner_items().for_each(|i| self.item(i.clone(), cache).unwrap());

        let id = item.def_id;
        if let Some(mut new_item) = self.convert_item(item) {
            if let types::ItemEnum::TraitItem(ref mut t) = new_item.inner {
                t.implementors = self.get_trait_implementors(id, cache)
            } else if let types::ItemEnum::StructItem(ref mut s) = new_item.inner {
                s.impls = self.get_impls(id, cache)
            } else if let types::ItemEnum::EnumItem(ref mut e) = new_item.inner {
                e.impls = self.get_impls(id, cache)
            }
            let removed = self.index.borrow_mut().insert(id.into(), new_item.clone());
            // FIXME(adotinthevoid): Currently, the index is duplicated. This is a sanity check
            // to make sure the items are unique.
            if let Some(old_item) = removed {
                assert_eq!(old_item, new_item);
            }
        }

        Ok(())
    }

    fn mod_item_in(
        &mut self,
        item: &clean::Item,
        _module_name: &str,
        cache: &Cache,
    ) -> Result<(), Error> {
        use clean::types::ItemKind::*;
        if let ModuleItem(m) = &item.kind {
            for item in &m.items {
                match &item.kind {
                    // These don't have names so they don't get added to the output by default
                    ImportItem(_) => self.item(item.clone(), cache).unwrap(),
                    ExternCrateItem(_, _) => self.item(item.clone(), cache).unwrap(),
                    ImplItem(i) => {
                        i.items.iter().for_each(|i| self.item(i.clone(), cache).unwrap())
                    }
                    _ => {}
                }
            }
        }
        self.item(item.clone(), cache).unwrap();
        Ok(())
    }

    fn mod_item_out(&mut self, _item_name: &str) -> Result<(), Error> {
        Ok(())
    }

    fn after_krate(&mut self, krate: &clean::Crate, cache: &Cache) -> Result<(), Error> {
        debug!("Done with crate");
        let mut index = (*self.index).clone().into_inner();
        index.extend(self.get_trait_items(cache));
        let output = types::Crate {
            root: types::Id(String::from("0:0")),
            crate_version: krate.version.clone(),
            includes_private: cache.document_private,
            index,
            paths: cache
                .paths
                .clone()
                .into_iter()
                .chain(cache.external_paths.clone().into_iter())
                .map(|(k, (path, kind))| {
                    (
                        k.into(),
                        types::ItemSummary { crate_id: k.krate.as_u32(), path, kind: kind.into() },
                    )
                })
                .collect(),
            external_crates: cache
                .extern_locations
                .iter()
                .map(|(k, v)| {
                    (
                        k.as_u32(),
                        types::ExternalCrate {
                            name: v.0.to_string(),
                            html_root_url: match &v.2 {
                                ExternalLocation::Remote(s) => Some(s.clone()),
                                _ => None,
                            },
                        },
                    )
                })
                .collect(),
            format_version: 1,
        };
        let mut p = self.out_path.clone();
        p.push(output.index.get(&output.root).unwrap().name.clone().unwrap());
        p.set_extension("json");
        let file = File::create(&p).map_err(|error| Error { error: error.to_string(), file: p })?;
        serde_json::ser::to_writer(&file, &output).unwrap();
        Ok(())
    }

    fn after_run(&mut self, _diag: &rustc_errors::Handler) -> Result<(), Error> {
        Ok(())
    }
}
