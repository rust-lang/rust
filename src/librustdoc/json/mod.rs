//! Rustdoc's JSON backend
//!
//! This module contains the logic for rendering a crate as JSON rather than the normal static HTML
//! output. See [the RFC](https://github.com/rust-lang/rfcs/pull/2963) and the [`types`] module
//! docs for usage and details.

mod conversions;
mod import_finder;

use std::cell::RefCell;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::rc::Rc;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::DefId;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::def_id::LOCAL_CRATE;

use rustdoc_json_types as types;

use crate::clean::types::{ExternalCrate, ExternalLocation};
use crate::clean::ItemKind;
use crate::config::RenderOptions;
use crate::docfs::PathError;
use crate::error::Error;
use crate::formats::cache::Cache;
use crate::formats::FormatRenderer;
use crate::json::conversions::{from_item_id, from_item_id_with_name, IntoWithTcx};
use crate::{clean, try_err};

#[derive(Clone)]
pub(crate) struct JsonRenderer<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// A mapping of IDs that contains all local items for this crate which gets output as a top
    /// level field of the JSON blob.
    index: Rc<RefCell<FxHashMap<types::Id, types::Item>>>,
    /// The directory where the blob will be written to.
    out_path: PathBuf,
    cache: Rc<Cache>,
    imported_items: FxHashSet<DefId>,
}

impl<'tcx> JsonRenderer<'tcx> {
    fn sess(&self) -> &'tcx Session {
        self.tcx.sess
    }

    fn get_trait_implementors(&mut self, id: DefId) -> Vec<types::Id> {
        Rc::clone(&self.cache)
            .implementors
            .get(&id)
            .map(|implementors| {
                implementors
                    .iter()
                    .map(|i| {
                        let item = &i.impl_item;
                        self.item(item.clone()).unwrap();
                        from_item_id_with_name(item.item_id, self.tcx, item.name)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_impls(&mut self, id: DefId) -> Vec<types::Id> {
        Rc::clone(&self.cache)
            .impls
            .get(&id)
            .map(|impls| {
                impls
                    .iter()
                    .filter_map(|i| {
                        let item = &i.impl_item;

                        // HACK(hkmatsumoto): For impls of primitive types, we index them
                        // regardless of whether they're local. This is because users can
                        // document primitive items in an arbitrary crate by using
                        // `doc(primitive)`.
                        let mut is_primitive_impl = false;
                        if let clean::types::ItemKind::ImplItem(ref impl_) = *item.kind {
                            if impl_.trait_.is_none() {
                                if let clean::types::Type::Primitive(_) = impl_.for_ {
                                    is_primitive_impl = true;
                                }
                            }
                        }

                        if item.item_id.is_local() || is_primitive_impl {
                            self.item(item.clone()).unwrap();
                            Some(from_item_id_with_name(item.item_id, self.tcx, item.name))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_trait_items(&mut self) -> Vec<(types::Id, types::Item)> {
        Rc::clone(&self.cache)
            .traits
            .iter()
            .filter_map(|(&id, trait_item)| {
                // only need to synthesize items for external traits
                if !id.is_local() {
                    let trait_item = &trait_item.trait_;
                    for item in &trait_item.items {
                        self.item(item.clone()).unwrap();
                    }
                    let item_id = from_item_id(id.into(), self.tcx);
                    Some((
                        item_id.clone(),
                        types::Item {
                            id: item_id,
                            crate_id: id.krate.as_u32(),
                            name: self
                                .cache
                                .paths
                                .get(&id)
                                .unwrap_or_else(|| {
                                    self.cache
                                        .external_paths
                                        .get(&id)
                                        .expect("Trait should either be in local or external paths")
                                })
                                .0
                                .last()
                                .map(|s| s.to_string()),
                            visibility: types::Visibility::Public,
                            inner: types::ItemEnum::Trait(trait_item.clone().into_tcx(self.tcx)),
                            span: None,
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

impl<'tcx> FormatRenderer<'tcx> for JsonRenderer<'tcx> {
    fn descr() -> &'static str {
        "json"
    }

    const RUN_ON_MODULE: bool = false;

    fn init(
        krate: clean::Crate,
        options: RenderOptions,
        cache: Cache,
        tcx: TyCtxt<'tcx>,
    ) -> Result<(Self, clean::Crate), Error> {
        debug!("Initializing json renderer");

        let (krate, imported_items) = import_finder::get_imports(krate);

        Ok((
            JsonRenderer {
                tcx,
                index: Rc::new(RefCell::new(FxHashMap::default())),
                out_path: options.output,
                cache: Rc::new(cache),
                imported_items,
            },
            krate,
        ))
    }

    fn make_child_renderer(&self) -> Self {
        self.clone()
    }

    /// Inserts an item into the index. This should be used rather than directly calling insert on
    /// the hashmap because certain items (traits and types) need to have their mappings for trait
    /// implementations filled out before they're inserted.
    fn item(&mut self, item: clean::Item) -> Result<(), Error> {
        trace!("rendering {} {:?}", item.type_(), item.name);

        // Flatten items that recursively store other items. We include orphaned items from
        // stripped modules and etc that are otherwise reachable.
        if let ItemKind::StrippedItem(inner) = &*item.kind {
            inner.inner_items().for_each(|i| self.item(i.clone()).unwrap());
        }

        // Flatten items that recursively store other items
        item.kind.inner_items().for_each(|i| self.item(i.clone()).unwrap());

        let name = item.name;
        let item_id = item.item_id;
        if let Some(mut new_item) = self.convert_item(item) {
            let can_be_ignored = match new_item.inner {
                types::ItemEnum::Trait(ref mut t) => {
                    t.implementations = self.get_trait_implementors(item_id.expect_def_id());
                    false
                }
                types::ItemEnum::Struct(ref mut s) => {
                    s.impls = self.get_impls(item_id.expect_def_id());
                    false
                }
                types::ItemEnum::Enum(ref mut e) => {
                    e.impls = self.get_impls(item_id.expect_def_id());
                    false
                }
                types::ItemEnum::Union(ref mut u) => {
                    u.impls = self.get_impls(item_id.expect_def_id());
                    false
                }

                types::ItemEnum::Method(_)
                | types::ItemEnum::Module(_)
                | types::ItemEnum::AssocConst { .. }
                | types::ItemEnum::AssocType { .. }
                | types::ItemEnum::PrimitiveType(_) => true,
                types::ItemEnum::ExternCrate { .. }
                | types::ItemEnum::Import(_)
                | types::ItemEnum::StructField(_)
                | types::ItemEnum::Variant(_)
                | types::ItemEnum::Function(_)
                | types::ItemEnum::TraitAlias(_)
                | types::ItemEnum::Impl(_)
                | types::ItemEnum::Typedef(_)
                | types::ItemEnum::OpaqueTy(_)
                | types::ItemEnum::Constant(_)
                | types::ItemEnum::Static(_)
                | types::ItemEnum::ForeignType
                | types::ItemEnum::Macro(_)
                | types::ItemEnum::ProcMacro(_) => false,
            };
            let removed = self
                .index
                .borrow_mut()
                .insert(from_item_id_with_name(item_id, self.tcx, name), new_item.clone());

            // FIXME(adotinthevoid): Currently, the index is duplicated. This is a sanity check
            // to make sure the items are unique. The main place this happens is when an item, is
            // reexported in more than one place. See `rustdoc-json/reexport/in_root_and_mod`
            if let Some(old_item) = removed {
                // In case of generic implementations (like `impl<T> Trait for T {}`), all the
                // inner items will be duplicated so we can ignore if they are slightly different.
                if !can_be_ignored {
                    assert_eq!(old_item, new_item);
                }
            }
        }

        Ok(())
    }

    fn mod_item_in(&mut self, _item: &clean::Item) -> Result<(), Error> {
        unreachable!("RUN_ON_MODULE = false should never call mod_item_in")
    }

    fn after_krate(&mut self) -> Result<(), Error> {
        debug!("Done with crate");

        for primitive in Rc::clone(&self.cache).primitive_locations.values() {
            self.get_impls(*primitive);
        }

        let e = ExternalCrate { crate_num: LOCAL_CRATE };

        let mut index = (*self.index).clone().into_inner();
        index.extend(self.get_trait_items());
        // This needs to be the default HashMap for compatibility with the public interface for
        // rustdoc-json-types
        #[allow(rustc::default_hash_types)]
        let output = types::Crate {
            root: types::Id(format!("0:0:{}", e.name(self.tcx).as_u32())),
            crate_version: self.cache.crate_version.clone(),
            includes_private: self.cache.document_private,
            index: index.into_iter().collect(),
            paths: self
                .cache
                .paths
                .iter()
                .chain(&self.cache.external_paths)
                .map(|(&k, &(ref path, kind))| {
                    (
                        from_item_id(k.into(), self.tcx),
                        types::ItemSummary {
                            crate_id: k.krate.as_u32(),
                            path: path.iter().map(|s| s.to_string()).collect(),
                            kind: kind.into_tcx(self.tcx),
                        },
                    )
                })
                .collect(),
            external_crates: self
                .cache
                .extern_locations
                .iter()
                .map(|(crate_num, external_location)| {
                    let e = ExternalCrate { crate_num: *crate_num };
                    (
                        crate_num.as_u32(),
                        types::ExternalCrate {
                            name: e.name(self.tcx).to_string(),
                            html_root_url: match external_location {
                                ExternalLocation::Remote(s) => Some(s.clone()),
                                _ => None,
                            },
                        },
                    )
                })
                .collect(),
            format_version: types::FORMAT_VERSION,
        };
        let out_dir = self.out_path.clone();
        try_err!(create_dir_all(&out_dir), out_dir);

        let mut p = out_dir;
        p.push(output.index.get(&output.root).unwrap().name.clone().unwrap());
        p.set_extension("json");
        let mut file = BufWriter::new(try_err!(File::create(&p), p));
        serde_json::ser::to_writer(&mut file, &output).unwrap();
        try_err!(file.flush(), p);

        Ok(())
    }

    fn cache(&self) -> &Cache {
        &self.cache
    }
}
