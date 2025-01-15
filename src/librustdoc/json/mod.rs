//! Rustdoc's JSON backend
//!
//! This module contains the logic for rendering a crate as JSON rather than the normal static HTML
//! output. See [the RFC](https://github.com/rust-lang/rfcs/pull/2963) and the [`types`] module
//! docs for usage and details.

mod conversions;
mod ids;
mod import_finder;

use std::cell::RefCell;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, stdout};
use std::path::PathBuf;
use std::rc::Rc;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::features::StabilityExt;
use rustc_span::def_id::LOCAL_CRATE;
use rustdoc_json_types as types;
// It's important to use the FxHashMap from rustdoc_json_types here, instead of
// the one from rustc_data_structures, as they're different types due to sysroots.
// See #110051 and #127456 for details
use rustdoc_json_types::FxHashMap;
use tracing::{debug, trace};

use crate::clean::ItemKind;
use crate::clean::types::{ExternalCrate, ExternalLocation};
use crate::config::RenderOptions;
use crate::docfs::PathError;
use crate::error::Error;
use crate::formats::FormatRenderer;
use crate::formats::cache::Cache;
use crate::json::conversions::IntoJson;
use crate::{clean, try_err};

#[derive(Clone)]
pub(crate) struct JsonRenderer<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// A mapping of IDs that contains all local items for this crate which gets output as a top
    /// level field of the JSON blob.
    index: Rc<RefCell<FxHashMap<types::Id, types::Item>>>,
    /// The directory where the JSON blob should be written to.
    ///
    /// If this is `None`, the blob will be printed to `stdout` instead.
    out_dir: Option<PathBuf>,
    cache: Rc<Cache>,
    imported_items: DefIdSet,
    id_interner: Rc<RefCell<ids::IdInterner>>,
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
                        self.id_from_item(item)
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
                        // `rustc_doc_primitive`.
                        let mut is_primitive_impl = false;
                        if let clean::types::ItemKind::ImplItem(ref impl_) = item.kind
                            && impl_.trait_.is_none()
                            && let clean::types::Type::Primitive(_) = impl_.for_
                        {
                            is_primitive_impl = true;
                        }

                        if item.item_id.is_local() || is_primitive_impl {
                            self.item(item.clone()).unwrap();
                            Some(self.id_from_item(item))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn serialize_and_write<T: Write>(
        &self,
        output_crate: types::Crate,
        mut writer: BufWriter<T>,
        path: &str,
    ) -> Result<(), Error> {
        self.sess().time("rustdoc_json_serialize_and_write", || {
            try_err!(
                serde_json::ser::to_writer(&mut writer, &output_crate).map_err(|e| e.to_string()),
                path
            );
            try_err!(writer.flush(), path);
            Ok(())
        })
    }
}

fn target(sess: &rustc_session::Session) -> types::Target {
    // Build a set of which features are enabled on this target
    let globally_enabled_features: FxHashSet<&str> =
        sess.unstable_target_features.iter().map(|name| name.as_str()).collect();

    // Build a map of target feature stability by feature name
    use rustc_target::target_features::Stability;
    let feature_stability: FxHashMap<&str, Stability> = sess
        .target
        .rust_target_features()
        .into_iter()
        .copied()
        .map(|(name, stability, _)| (name, stability))
        .collect();

    types::Target {
        triple: sess.opts.target_triple.tuple().into(),
        target_features: sess
            .target
            .rust_target_features()
            .into_iter()
            .copied()
            .filter(|(_, stability, _)| {
                // Describe only target features which the user can toggle
                stability.is_toggle_permitted(sess).is_ok()
            })
            .map(|(name, stability, implied_features)| {
                types::TargetFeature {
                    name: name.into(),
                    unstable_feature_gate: match stability {
                        Stability::Unstable(feature_gate) => Some(feature_gate.as_str().into()),
                        _ => None,
                    },
                    implies_features: implied_features
                        .into_iter()
                        .copied()
                        .filter(|name| {
                            // Imply only target features which the user can toggle
                            feature_stability
                                .get(name)
                                .map(|stability| stability.is_toggle_permitted(sess).is_ok())
                                .unwrap_or(false)
                        })
                        .map(String::from)
                        .collect(),
                    globally_enabled: globally_enabled_features.contains(name),
                }
            })
            .collect(),
    }
}

impl<'tcx> FormatRenderer<'tcx> for JsonRenderer<'tcx> {
    fn descr() -> &'static str {
        "json"
    }

    const RUN_ON_MODULE: bool = false;
    type ModuleData = ();

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
                out_dir: if options.output_to_stdout { None } else { Some(options.output) },
                cache: Rc::new(cache),
                imported_items,
                id_interner: Default::default(),
            },
            krate,
        ))
    }

    fn save_module_data(&mut self) -> Self::ModuleData {
        unreachable!("RUN_ON_MODULE = false, should never call save_module_data")
    }
    fn restore_module_data(&mut self, _info: Self::ModuleData) {
        unreachable!("RUN_ON_MODULE = false, should never call set_back_info")
    }

    /// Inserts an item into the index. This should be used rather than directly calling insert on
    /// the hashmap because certain items (traits and types) need to have their mappings for trait
    /// implementations filled out before they're inserted.
    fn item(&mut self, item: clean::Item) -> Result<(), Error> {
        let item_type = item.type_();
        let item_name = item.name;
        trace!("rendering {item_type} {item_name:?}");

        // Flatten items that recursively store other items. We include orphaned items from
        // stripped modules and etc that are otherwise reachable.
        if let ItemKind::StrippedItem(inner) = &item.kind {
            inner.inner_items().for_each(|i| self.item(i.clone()).unwrap());
        }

        // Flatten items that recursively store other items
        item.kind.inner_items().for_each(|i| self.item(i.clone()).unwrap());

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
                types::ItemEnum::Primitive(ref mut p) => {
                    p.impls = self.get_impls(item_id.expect_def_id());
                    false
                }

                types::ItemEnum::Function(_)
                | types::ItemEnum::Module(_)
                | types::ItemEnum::Use(_)
                | types::ItemEnum::AssocConst { .. }
                | types::ItemEnum::AssocType { .. } => true,
                types::ItemEnum::ExternCrate { .. }
                | types::ItemEnum::StructField(_)
                | types::ItemEnum::Variant(_)
                | types::ItemEnum::TraitAlias(_)
                | types::ItemEnum::Impl(_)
                | types::ItemEnum::TypeAlias(_)
                | types::ItemEnum::Constant { .. }
                | types::ItemEnum::Static(_)
                | types::ItemEnum::ExternType
                | types::ItemEnum::Macro(_)
                | types::ItemEnum::ProcMacro(_) => false,
            };
            let removed = self.index.borrow_mut().insert(new_item.id, new_item.clone());

            // FIXME(adotinthevoid): Currently, the index is duplicated. This is a sanity check
            // to make sure the items are unique. The main place this happens is when an item, is
            // reexported in more than one place. See `rustdoc-json/reexport/in_root_and_mod`
            if let Some(old_item) = removed {
                // In case of generic implementations (like `impl<T> Trait for T {}`), all the
                // inner items will be duplicated so we can ignore if they are slightly different.
                if !can_be_ignored {
                    assert_eq!(old_item, new_item);
                }
                trace!("replaced {old_item:?}\nwith {new_item:?}");
            }
        }

        trace!("done rendering {item_type} {item_name:?}");
        Ok(())
    }

    fn mod_item_in(&mut self, _item: &clean::Item) -> Result<(), Error> {
        unreachable!("RUN_ON_MODULE = false, should never call mod_item_in")
    }

    fn after_krate(&mut self) -> Result<(), Error> {
        debug!("Done with crate");

        let e = ExternalCrate { crate_num: LOCAL_CRATE };
        let index = (*self.index).clone().into_inner();

        // Note that tcx.rust_target_features is inappropriate here because rustdoc tries to run for
        // multiple targets: https://github.com/rust-lang/rust/pull/137632
        //
        // We want to describe a single target, so pass tcx.sess rather than tcx.
        let target = target(self.tcx.sess);

        debug!("Constructing Output");
        let output_crate = types::Crate {
            root: self.id_from_item_default(e.def_id().into()),
            crate_version: self.cache.crate_version.clone(),
            includes_private: self.cache.document_private,
            index,
            paths: self
                .cache
                .paths
                .iter()
                .chain(&self.cache.external_paths)
                .map(|(&k, &(ref path, kind))| {
                    (
                        self.id_from_item_default(k.into()),
                        types::ItemSummary {
                            crate_id: k.krate.as_u32(),
                            path: path.iter().map(|s| s.to_string()).collect(),
                            kind: kind.into_json(self),
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
            target,
            format_version: types::FORMAT_VERSION,
        };
        if let Some(ref out_dir) = self.out_dir {
            try_err!(create_dir_all(out_dir), out_dir);

            let mut p = out_dir.clone();
            p.push(output_crate.index.get(&output_crate.root).unwrap().name.clone().unwrap());
            p.set_extension("json");

            self.serialize_and_write(
                output_crate,
                try_err!(File::create_buffered(&p), p),
                &p.display().to_string(),
            )
        } else {
            self.serialize_and_write(output_crate, BufWriter::new(stdout().lock()), "<stdout>")
        }
    }

    fn cache(&self) -> &Cache {
        &self.cache
    }
}
