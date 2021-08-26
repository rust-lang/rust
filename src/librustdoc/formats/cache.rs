use std::mem;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX};
use rustc_middle::middle::privacy::AccessLevels;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

use crate::clean::{self, GetDefId, ItemId, PrimitiveType};
use crate::config::RenderOptions;
use crate::fold::DocFolder;
use crate::formats::item_type::ItemType;
use crate::formats::Impl;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::cache::{get_index_search_type, ExternalLocation};
use crate::html::render::IndexItem;

/// This cache is used to store information about the [`clean::Crate`] being
/// rendered in order to provide more useful documentation. This contains
/// information like all implementors of a trait, all traits a type implements,
/// documentation for all known traits, etc.
///
/// This structure purposefully does not implement `Clone` because it's intended
/// to be a fairly large and expensive structure to clone. Instead this adheres
/// to `Send` so it may be stored in an `Arc` instance and shared among the various
/// rendering threads.
#[derive(Default)]
crate struct Cache {
    /// Maps a type ID to all known implementations for that type. This is only
    /// recognized for intra-crate `ResolvedPath` types, and is used to print
    /// out extra documentation on the page of an enum/struct.
    ///
    /// The values of the map are a list of implementations and documentation
    /// found on that implementation.
    crate impls: FxHashMap<DefId, Vec<Impl>>,

    /// Maintains a mapping of local crate `DefId`s to the fully qualified name
    /// and "short type description" of that node. This is used when generating
    /// URLs when a type is being linked to. External paths are not located in
    /// this map because the `External` type itself has all the information
    /// necessary.
    crate paths: FxHashMap<DefId, (Vec<String>, ItemType)>,

    /// Similar to `paths`, but only holds external paths. This is only used for
    /// generating explicit hyperlinks to other crates.
    crate external_paths: FxHashMap<DefId, (Vec<String>, ItemType)>,

    /// Maps local `DefId`s of exported types to fully qualified paths.
    /// Unlike 'paths', this mapping ignores any renames that occur
    /// due to 'use' statements.
    ///
    /// This map is used when writing out the special 'implementors'
    /// javascript file. By using the exact path that the type
    /// is declared with, we ensure that each path will be identical
    /// to the path used if the corresponding type is inlined. By
    /// doing this, we can detect duplicate impls on a trait page, and only display
    /// the impl for the inlined type.
    crate exact_paths: FxHashMap<DefId, Vec<String>>,

    /// This map contains information about all known traits of this crate.
    /// Implementations of a crate should inherit the documentation of the
    /// parent trait if no extra documentation is specified, and default methods
    /// should show up in documentation about trait implementations.
    crate traits: FxHashMap<DefId, clean::TraitWithExtraInfo>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    crate implementors: FxHashMap<DefId, Vec<Impl>>,

    /// Cache of where external crate documentation can be found.
    crate extern_locations: FxHashMap<CrateNum, ExternalLocation>,

    /// Cache of where documentation for primitives can be found.
    crate primitive_locations: FxHashMap<clean::PrimitiveType, DefId>,

    // Note that external items for which `doc(hidden)` applies to are shown as
    // non-reachable while local items aren't. This is because we're reusing
    // the access levels from the privacy check pass.
    crate access_levels: AccessLevels<DefId>,

    /// The version of the crate being documented, if given from the `--crate-version` flag.
    crate crate_version: Option<String>,

    /// Whether to document private items.
    /// This is stored in `Cache` so it doesn't need to be passed through all rustdoc functions.
    crate document_private: bool,

    /// Crates marked with [`#[doc(masked)]`][doc_masked].
    ///
    /// [doc_masked]: https://doc.rust-lang.org/nightly/unstable-book/language-features/doc-masked.html
    crate masked_crates: FxHashSet<CrateNum>,

    // Private fields only used when initially crawling a crate to build a cache
    stack: Vec<String>,
    parent_stack: Vec<DefId>,
    parent_is_trait_impl: bool,
    stripped_mod: bool,

    crate search_index: Vec<IndexItem>,
    crate deref_trait_did: Option<DefId>,
    crate deref_mut_trait_did: Option<DefId>,
    crate owned_box_did: Option<DefId>,

    // In rare case where a structure is defined in one module but implemented
    // in another, if the implementing module is parsed before defining module,
    // then the fully qualified name of the structure isn't presented in `paths`
    // yet when its implementation methods are being indexed. Caches such methods
    // and their parent id here and indexes them at the end of crate parsing.
    crate orphan_impl_items: Vec<(DefId, clean::Item)>,

    // Similarly to `orphan_impl_items`, sometimes trait impls are picked up
    // even though the trait itself is not exported. This can happen if a trait
    // was defined in function/expression scope, since the impl will be picked
    // up by `collect-trait-impls` but the trait won't be scraped out in the HIR
    // crawl. In order to prevent crashes when looking for notable traits or
    // when gathering trait documentation on a type, hold impls here while
    // folding and add them to the cache later on if we find the trait.
    orphan_trait_impls: Vec<(DefId, FxHashSet<DefId>, Impl)>,

    /// All intra-doc links resolved so far.
    ///
    /// Links are indexed by the DefId of the item they document.
    crate intra_doc_links: FxHashMap<ItemId, Vec<clean::ItemLink>>,
}

/// This struct is used to wrap the `cache` and `tcx` in order to run `DocFolder`.
struct CacheBuilder<'a, 'tcx> {
    cache: &'a mut Cache,
    tcx: TyCtxt<'tcx>,
}

impl Cache {
    crate fn new(access_levels: AccessLevels<DefId>, document_private: bool) -> Self {
        Cache { access_levels, document_private, ..Cache::default() }
    }

    /// Populates the `Cache` with more data. The returned `Crate` will be missing some data that was
    /// in `krate` due to the data being moved into the `Cache`.
    crate fn populate(
        &mut self,
        mut krate: clean::Crate,
        tcx: TyCtxt<'_>,
        render_options: &RenderOptions,
    ) -> clean::Crate {
        // Crawl the crate to build various caches used for the output
        debug!(?self.crate_version);
        self.traits = krate.external_traits.take();
        let RenderOptions { extern_html_root_takes_precedence, output: dst, .. } = render_options;

        // Cache where all our extern crates are located
        // FIXME: this part is specific to HTML so it'd be nice to remove it from the common code
        for &e in &krate.externs {
            let name = e.name(tcx);
            let extern_url =
                render_options.extern_html_root_urls.get(&*name.as_str()).map(|u| &**u);
            let location = e.location(extern_url, *extern_html_root_takes_precedence, dst, tcx);
            self.extern_locations.insert(e.crate_num, location);
            self.external_paths.insert(e.def_id(), (vec![name.to_string()], ItemType::Module));
        }

        // FIXME: avoid this clone (requires implementing Default manually)
        self.primitive_locations = PrimitiveType::primitive_locations(tcx).clone();
        for (prim, &def_id) in &self.primitive_locations {
            let crate_name = tcx.crate_name(def_id.krate);
            // Recall that we only allow primitive modules to be at the root-level of the crate.
            // If that restriction is ever lifted, this will have to include the relative paths instead.
            self.external_paths.insert(
                def_id,
                (vec![crate_name.to_string(), prim.as_sym().to_string()], ItemType::Primitive),
            );
        }

        krate = CacheBuilder { tcx, cache: self }.fold_crate(krate);

        for (trait_did, dids, impl_) in self.orphan_trait_impls.drain(..) {
            if self.traits.contains_key(&trait_did) {
                for did in dids {
                    self.impls.entry(did).or_default().push(impl_.clone());
                }
            }
        }

        krate
    }
}

impl<'a, 'tcx> DocFolder for CacheBuilder<'a, 'tcx> {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        if item.def_id.is_local() {
            debug!("folding {} \"{:?}\", id {:?}", item.type_(), item.name, item.def_id);
        }

        // If this is a stripped module,
        // we don't want it or its children in the search index.
        let orig_stripped_mod = match *item.kind {
            clean::StrippedItem(box clean::ModuleItem(..)) => {
                mem::replace(&mut self.cache.stripped_mod, true)
            }
            _ => self.cache.stripped_mod,
        };

        // If the impl is from a masked crate or references something from a
        // masked crate then remove it completely.
        if let clean::ImplItem(ref i) = *item.kind {
            if self.cache.masked_crates.contains(&item.def_id.krate())
                || i.trait_.def_id().map_or(false, |d| self.cache.masked_crates.contains(&d.krate))
                || i.for_.def_id().map_or(false, |d| self.cache.masked_crates.contains(&d.krate))
            {
                return None;
            }
        }

        // Propagate a trait method's documentation to all implementors of the
        // trait.
        if let clean::TraitItem(ref t) = *item.kind {
            self.cache.traits.entry(item.def_id.expect_def_id()).or_insert_with(|| {
                clean::TraitWithExtraInfo {
                    trait_: t.clone(),
                    is_notable: item.attrs.has_doc_flag(sym::notable_trait),
                }
            });
        }

        // Collect all the implementors of traits.
        if let clean::ImplItem(ref i) = *item.kind {
            if let Some(did) = i.trait_.def_id() {
                if i.blanket_impl.is_none() {
                    self.cache
                        .implementors
                        .entry(did)
                        .or_default()
                        .push(Impl { impl_item: item.clone() });
                }
            }
        }

        // Index this method for searching later on.
        if let Some(ref s) = item.name {
            let (parent, is_inherent_impl_item) = match *item.kind {
                clean::StrippedItem(..) => ((None, None), false),
                clean::AssocConstItem(..) | clean::TypedefItem(_, true)
                    if self.cache.parent_is_trait_impl =>
                {
                    // skip associated items in trait impls
                    ((None, None), false)
                }
                clean::AssocTypeItem(..)
                | clean::TyMethodItem(..)
                | clean::StructFieldItem(..)
                | clean::VariantItem(..) => (
                    (
                        Some(*self.cache.parent_stack.last().expect("parent_stack is empty")),
                        Some(&self.cache.stack[..self.cache.stack.len() - 1]),
                    ),
                    false,
                ),
                clean::MethodItem(..) | clean::AssocConstItem(..) => {
                    if self.cache.parent_stack.is_empty() {
                        ((None, None), false)
                    } else {
                        let last = self.cache.parent_stack.last().expect("parent_stack is empty 2");
                        let did = *last;
                        let path = match self.cache.paths.get(&did) {
                            // The current stack not necessarily has correlation
                            // for where the type was defined. On the other
                            // hand, `paths` always has the right
                            // information if present.
                            Some(&(
                                ref fqp,
                                ItemType::Trait
                                | ItemType::Struct
                                | ItemType::Union
                                | ItemType::Enum,
                            )) => Some(&fqp[..fqp.len() - 1]),
                            Some(..) => Some(&*self.cache.stack),
                            None => None,
                        };
                        ((Some(*last), path), true)
                    }
                }
                _ => ((None, Some(&*self.cache.stack)), false),
            };

            match parent {
                (parent, Some(path)) if is_inherent_impl_item || !self.cache.stripped_mod => {
                    debug_assert!(!item.is_stripped());

                    // A crate has a module at its root, containing all items,
                    // which should not be indexed. The crate-item itself is
                    // inserted later on when serializing the search-index.
                    if item.def_id.index().map_or(false, |idx| idx != CRATE_DEF_INDEX) {
                        let desc = item.doc_value().map_or_else(String::new, |x| {
                            short_markdown_summary(&x.as_str(), &item.link_names(&self.cache))
                        });
                        self.cache.search_index.push(IndexItem {
                            ty: item.type_(),
                            name: s.to_string(),
                            path: path.join("::"),
                            desc,
                            parent,
                            parent_idx: None,
                            search_type: get_index_search_type(&item, self.tcx),
                            aliases: item.attrs.get_doc_aliases(),
                        });
                    }
                }
                (Some(parent), None) if is_inherent_impl_item => {
                    // We have a parent, but we don't know where they're
                    // defined yet. Wait for later to index this item.
                    self.cache.orphan_impl_items.push((parent, item.clone()));
                }
                _ => {}
            }
        }

        // Keep track of the fully qualified path for this item.
        let pushed = match item.name {
            Some(n) if !n.is_empty() => {
                self.cache.stack.push(n.to_string());
                true
            }
            _ => false,
        };

        match *item.kind {
            clean::StructItem(..)
            | clean::EnumItem(..)
            | clean::TypedefItem(..)
            | clean::TraitItem(..)
            | clean::TraitAliasItem(..)
            | clean::FunctionItem(..)
            | clean::ModuleItem(..)
            | clean::ForeignFunctionItem(..)
            | clean::ForeignStaticItem(..)
            | clean::ConstantItem(..)
            | clean::StaticItem(..)
            | clean::UnionItem(..)
            | clean::ForeignTypeItem
            | clean::MacroItem(..)
            | clean::ProcMacroItem(..)
            | clean::VariantItem(..) => {
                if !self.cache.stripped_mod {
                    // Re-exported items mean that the same id can show up twice
                    // in the rustdoc ast that we're looking at. We know,
                    // however, that a re-exported item doesn't show up in the
                    // `public_items` map, so we can skip inserting into the
                    // paths map if there was already an entry present and we're
                    // not a public item.
                    if !self.cache.paths.contains_key(&item.def_id.expect_def_id())
                        || self.cache.access_levels.is_public(item.def_id.expect_def_id())
                    {
                        self.cache.paths.insert(
                            item.def_id.expect_def_id(),
                            (self.cache.stack.clone(), item.type_()),
                        );
                    }
                }
            }
            clean::PrimitiveItem(..) => {
                self.cache
                    .paths
                    .insert(item.def_id.expect_def_id(), (self.cache.stack.clone(), item.type_()));
            }

            clean::ExternCrateItem { .. }
            | clean::ImportItem(..)
            | clean::OpaqueTyItem(..)
            | clean::ImplItem(..)
            | clean::TyMethodItem(..)
            | clean::MethodItem(..)
            | clean::StructFieldItem(..)
            | clean::AssocConstItem(..)
            | clean::AssocTypeItem(..)
            | clean::StrippedItem(..)
            | clean::KeywordItem(..) => {
                // FIXME: Do these need handling?
                // The person writing this comment doesn't know.
                // So would rather leave them to an expert,
                // as at least the list is better than `_ => {}`.
            }
        }

        // Maintain the parent stack
        let orig_parent_is_trait_impl = self.cache.parent_is_trait_impl;
        let parent_pushed = match *item.kind {
            clean::TraitItem(..)
            | clean::EnumItem(..)
            | clean::ForeignTypeItem
            | clean::StructItem(..)
            | clean::UnionItem(..)
            | clean::VariantItem(..) => {
                self.cache.parent_stack.push(item.def_id.expect_def_id());
                self.cache.parent_is_trait_impl = false;
                true
            }
            clean::ImplItem(ref i) => {
                self.cache.parent_is_trait_impl = i.trait_.is_some();
                match i.for_ {
                    clean::ResolvedPath { did, .. } => {
                        self.cache.parent_stack.push(did);
                        true
                    }
                    clean::DynTrait(ref bounds, _)
                    | clean::BorrowedRef { type_: box clean::DynTrait(ref bounds, _), .. } => {
                        if let Some(did) = bounds[0].trait_.def_id() {
                            self.cache.parent_stack.push(did);
                            true
                        } else {
                            false
                        }
                    }
                    ref t => {
                        let prim_did = t
                            .primitive_type()
                            .and_then(|t| self.cache.primitive_locations.get(&t).cloned());
                        match prim_did {
                            Some(did) => {
                                self.cache.parent_stack.push(did);
                                true
                            }
                            None => false,
                        }
                    }
                }
            }
            _ => false,
        };

        // Once we've recursively found all the generics, hoard off all the
        // implementations elsewhere.
        let item = self.fold_item_recur(item);
        let ret = if let clean::Item { kind: box clean::ImplItem(ref i), .. } = item {
            // Figure out the id of this impl. This may map to a
            // primitive rather than always to a struct/enum.
            // Note: matching twice to restrict the lifetime of the `i` borrow.
            let mut dids = FxHashSet::default();
            match i.for_ {
                clean::ResolvedPath { did, .. }
                | clean::BorrowedRef { type_: box clean::ResolvedPath { did, .. }, .. } => {
                    dids.insert(did);
                }
                clean::DynTrait(ref bounds, _)
                | clean::BorrowedRef { type_: box clean::DynTrait(ref bounds, _), .. } => {
                    if let Some(did) = bounds[0].trait_.def_id() {
                        dids.insert(did);
                    }
                }
                ref t => {
                    let did = t
                        .primitive_type()
                        .and_then(|t| self.cache.primitive_locations.get(&t).cloned());

                    if let Some(did) = did {
                        dids.insert(did);
                    }
                }
            }

            if let Some(generics) = i.trait_.as_ref().and_then(|t| t.generics()) {
                for bound in generics {
                    if let Some(did) = bound.def_id() {
                        dids.insert(did);
                    }
                }
            }
            let impl_item = Impl { impl_item: item };
            if impl_item.trait_did().map_or(true, |d| self.cache.traits.contains_key(&d)) {
                for did in dids {
                    self.cache.impls.entry(did).or_insert(vec![]).push(impl_item.clone());
                }
            } else {
                let trait_did = impl_item.trait_did().expect("no trait did");
                self.cache.orphan_trait_impls.push((trait_did, dids, impl_item));
            }
            None
        } else {
            Some(item)
        };

        if pushed {
            self.cache.stack.pop().expect("stack already empty");
        }
        if parent_pushed {
            self.cache.parent_stack.pop().expect("parent stack already empty");
        }
        self.cache.stripped_mod = orig_stripped_mod;
        self.cache.parent_is_trait_impl = orig_parent_is_trait_impl;
        ret
    }
}
