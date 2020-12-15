use std::cell::RefCell;
use std::collections::BTreeMap;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX};
use rustc_middle::middle::privacy::AccessLevels;
use rustc_span::source_map::FileName;
use rustc_span::Symbol;

use crate::clean::{self, GetDefId};
use crate::config::RenderInfo;
use crate::fold::DocFolder;
use crate::formats::item_type::ItemType;
use crate::formats::Impl;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::cache::{extern_location, get_index_search_type, ExternalLocation};
use crate::html::render::IndexItem;

thread_local!(crate static CACHE_KEY: RefCell<Arc<Cache>> = Default::default());

/// This cache is used to store information about the [`clean::Crate`] being
/// rendered in order to provide more useful documentation. This contains
/// information like all implementors of a trait, all traits a type implements,
/// documentation for all known traits, etc.
///
/// This structure purposefully does not implement `Clone` because it's intended
/// to be a fairly large and expensive structure to clone. Instead this adheres
/// to `Send` so it may be stored in a `Arc` instance and shared among the various
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
    crate traits: FxHashMap<DefId, clean::Trait>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    crate implementors: FxHashMap<DefId, Vec<Impl>>,

    /// Cache of where external crate documentation can be found.
    crate extern_locations: FxHashMap<CrateNum, (Symbol, PathBuf, ExternalLocation)>,

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

    // Private fields only used when initially crawling a crate to build a cache
    stack: Vec<String>,
    parent_stack: Vec<DefId>,
    parent_is_trait_impl: bool,
    stripped_mod: bool,
    masked_crates: FxHashSet<CrateNum>,

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
    // crawl. In order to prevent crashes when looking for spotlight traits or
    // when gathering trait documentation on a type, hold impls here while
    // folding and add them to the cache later on if we find the trait.
    orphan_trait_impls: Vec<(DefId, FxHashSet<DefId>, Impl)>,

    /// Aliases added through `#[doc(alias = "...")]`. Since a few items can have the same alias,
    /// we need the alias element to have an array of items.
    crate aliases: BTreeMap<String, Vec<usize>>,
}

impl Cache {
    crate fn from_krate(
        render_info: RenderInfo,
        document_private: bool,
        extern_html_root_urls: &BTreeMap<String, String>,
        dst: &Path,
        mut krate: clean::Crate,
    ) -> (clean::Crate, Cache) {
        // Crawl the crate to build various caches used for the output
        let RenderInfo {
            inlined: _,
            external_paths,
            exact_paths,
            access_levels,
            deref_trait_did,
            deref_mut_trait_did,
            owned_box_did,
            ..
        } = render_info;

        let external_paths =
            external_paths.into_iter().map(|(k, (v, t))| (k, (v, ItemType::from(t)))).collect();

        let mut cache = Cache {
            external_paths,
            exact_paths,
            parent_is_trait_impl: false,
            stripped_mod: false,
            access_levels,
            crate_version: krate.version.take(),
            document_private,
            traits: krate.external_traits.replace(Default::default()),
            deref_trait_did,
            deref_mut_trait_did,
            owned_box_did,
            masked_crates: mem::take(&mut krate.masked_crates),
            ..Cache::default()
        };

        // Cache where all our extern crates are located
        // FIXME: this part is specific to HTML so it'd be nice to remove it from the common code
        for &(n, ref e) in &krate.externs {
            let src_root = match e.src {
                FileName::Real(ref p) => match p.local_path().parent() {
                    Some(p) => p.to_path_buf(),
                    None => PathBuf::new(),
                },
                _ => PathBuf::new(),
            };
            let extern_url = extern_html_root_urls.get(&*e.name.as_str()).map(|u| &**u);
            cache
                .extern_locations
                .insert(n, (e.name, src_root, extern_location(e, extern_url, &dst)));

            let did = DefId { krate: n, index: CRATE_DEF_INDEX };
            cache.external_paths.insert(did, (vec![e.name.to_string()], ItemType::Module));
        }

        // Cache where all known primitives have their documentation located.
        //
        // Favor linking to as local extern as possible, so iterate all crates in
        // reverse topological order.
        for &(_, ref e) in krate.externs.iter().rev() {
            for &(def_id, prim) in &e.primitives {
                cache.primitive_locations.insert(prim, def_id);
            }
        }
        for &(def_id, prim) in &krate.primitives {
            cache.primitive_locations.insert(prim, def_id);
        }

        cache.stack.push(krate.name.to_string());
        krate = cache.fold_crate(krate);

        for (trait_did, dids, impl_) in cache.orphan_trait_impls.drain(..) {
            if cache.traits.contains_key(&trait_did) {
                for did in dids {
                    cache.impls.entry(did).or_default().push(impl_.clone());
                }
            }
        }

        (krate, cache)
    }
}

impl DocFolder for Cache {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        if item.def_id.is_local() {
            debug!("folding {} \"{:?}\", id {:?}", item.type_(), item.name, item.def_id);
        }

        // If this is a stripped module,
        // we don't want it or its children in the search index.
        let orig_stripped_mod = match item.kind {
            clean::StrippedItem(box clean::ModuleItem(..)) => {
                mem::replace(&mut self.stripped_mod, true)
            }
            _ => self.stripped_mod,
        };

        // If the impl is from a masked crate or references something from a
        // masked crate then remove it completely.
        if let clean::ImplItem(ref i) = item.kind {
            if self.masked_crates.contains(&item.def_id.krate)
                || i.trait_.def_id().map_or(false, |d| self.masked_crates.contains(&d.krate))
                || i.for_.def_id().map_or(false, |d| self.masked_crates.contains(&d.krate))
            {
                return None;
            }
        }

        // Propagate a trait method's documentation to all implementors of the
        // trait.
        if let clean::TraitItem(ref t) = item.kind {
            self.traits.entry(item.def_id).or_insert_with(|| t.clone());
        }

        // Collect all the implementors of traits.
        if let clean::ImplItem(ref i) = item.kind {
            if let Some(did) = i.trait_.def_id() {
                if i.blanket_impl.is_none() {
                    self.implementors
                        .entry(did)
                        .or_default()
                        .push(Impl { impl_item: item.clone() });
                }
            }
        }

        // Index this method for searching later on.
        if let Some(ref s) = item.name {
            let (parent, is_inherent_impl_item) = match item.kind {
                clean::StrippedItem(..) => ((None, None), false),
                clean::AssocConstItem(..) | clean::TypedefItem(_, true)
                    if self.parent_is_trait_impl =>
                {
                    // skip associated items in trait impls
                    ((None, None), false)
                }
                clean::AssocTypeItem(..)
                | clean::TyMethodItem(..)
                | clean::StructFieldItem(..)
                | clean::VariantItem(..) => (
                    (
                        Some(*self.parent_stack.last().expect("parent_stack is empty")),
                        Some(&self.stack[..self.stack.len() - 1]),
                    ),
                    false,
                ),
                clean::MethodItem(..) | clean::AssocConstItem(..) => {
                    if self.parent_stack.is_empty() {
                        ((None, None), false)
                    } else {
                        let last = self.parent_stack.last().expect("parent_stack is empty 2");
                        let did = *last;
                        let path = match self.paths.get(&did) {
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
                            Some(..) => Some(&*self.stack),
                            None => None,
                        };
                        ((Some(*last), path), true)
                    }
                }
                _ => ((None, Some(&*self.stack)), false),
            };

            match parent {
                (parent, Some(path)) if is_inherent_impl_item || !self.stripped_mod => {
                    debug_assert!(!item.is_stripped());

                    // A crate has a module at its root, containing all items,
                    // which should not be indexed. The crate-item itself is
                    // inserted later on when serializing the search-index.
                    if item.def_id.index != CRATE_DEF_INDEX {
                        self.search_index.push(IndexItem {
                            ty: item.type_(),
                            name: s.to_string(),
                            path: path.join("::"),
                            desc: item
                                .doc_value()
                                .map_or_else(|| String::new(), short_markdown_summary),
                            parent,
                            parent_idx: None,
                            search_type: get_index_search_type(&item),
                        });

                        for alias in item.attrs.get_doc_aliases() {
                            self.aliases
                                .entry(alias.to_lowercase())
                                .or_insert(Vec::new())
                                .push(self.search_index.len() - 1);
                        }
                    }
                }
                (Some(parent), None) if is_inherent_impl_item => {
                    // We have a parent, but we don't know where they're
                    // defined yet. Wait for later to index this item.
                    self.orphan_impl_items.push((parent, item.clone()));
                }
                _ => {}
            }
        }

        // Keep track of the fully qualified path for this item.
        let pushed = match item.name {
            Some(n) if !n.is_empty() => {
                self.stack.push(n.to_string());
                true
            }
            _ => false,
        };

        match item.kind {
            clean::StructItem(..)
            | clean::EnumItem(..)
            | clean::TypedefItem(..)
            | clean::TraitItem(..)
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
            | clean::VariantItem(..)
                if !self.stripped_mod =>
            {
                // Re-exported items mean that the same id can show up twice
                // in the rustdoc ast that we're looking at. We know,
                // however, that a re-exported item doesn't show up in the
                // `public_items` map, so we can skip inserting into the
                // paths map if there was already an entry present and we're
                // not a public item.
                if !self.paths.contains_key(&item.def_id)
                    || self.access_levels.is_public(item.def_id)
                {
                    self.paths.insert(item.def_id, (self.stack.clone(), item.type_()));
                }
            }
            clean::PrimitiveItem(..) => {
                self.paths.insert(item.def_id, (self.stack.clone(), item.type_()));
            }

            _ => {}
        }

        // Maintain the parent stack
        let orig_parent_is_trait_impl = self.parent_is_trait_impl;
        let parent_pushed = match item.kind {
            clean::TraitItem(..)
            | clean::EnumItem(..)
            | clean::ForeignTypeItem
            | clean::StructItem(..)
            | clean::UnionItem(..)
            | clean::VariantItem(..) => {
                self.parent_stack.push(item.def_id);
                self.parent_is_trait_impl = false;
                true
            }
            clean::ImplItem(ref i) => {
                self.parent_is_trait_impl = i.trait_.is_some();
                match i.for_ {
                    clean::ResolvedPath { did, .. } => {
                        self.parent_stack.push(did);
                        true
                    }
                    ref t => {
                        let prim_did = t
                            .primitive_type()
                            .and_then(|t| self.primitive_locations.get(&t).cloned());
                        match prim_did {
                            Some(did) => {
                                self.parent_stack.push(did);
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
        let ret = if let clean::Item { kind: clean::ImplItem(_), .. } = item {
            // Figure out the id of this impl. This may map to a
            // primitive rather than always to a struct/enum.
            // Note: matching twice to restrict the lifetime of the `i` borrow.
            let mut dids = FxHashSet::default();
            if let clean::Item { kind: clean::ImplItem(ref i), .. } = item {
                match i.for_ {
                    clean::ResolvedPath { did, .. }
                    | clean::BorrowedRef { type_: box clean::ResolvedPath { did, .. }, .. } => {
                        dids.insert(did);
                    }
                    ref t => {
                        let did = t
                            .primitive_type()
                            .and_then(|t| self.primitive_locations.get(&t).cloned());

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
            } else {
                unreachable!()
            };
            let impl_item = Impl { impl_item: item };
            if impl_item.trait_did().map_or(true, |d| self.traits.contains_key(&d)) {
                for did in dids {
                    self.impls.entry(did).or_insert(vec![]).push(impl_item.clone());
                }
            } else {
                let trait_did = impl_item.trait_did().expect("no trait did");
                self.orphan_trait_impls.push((trait_did, dids, impl_item));
            }
            None
        } else {
            Some(item)
        };

        if pushed {
            self.stack.pop().expect("stack already empty");
        }
        if parent_pushed {
            self.parent_stack.pop().expect("parent stack already empty");
        }
        self.stripped_mod = orig_stripped_mod;
        self.parent_is_trait_impl = orig_parent_is_trait_impl;
        ret
    }
}

crate fn cache() -> Arc<Cache> {
    CACHE_KEY.with(|c| c.borrow().clone())
}
