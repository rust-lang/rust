use crate::clean::{self, AttributesExt, GetDefId};
use crate::fold::DocFolder;
use rustc::middle::privacy::AccessLevels;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX};
use rustc_span::source_map::FileName;
use rustc_span::symbol::sym;
use std::collections::BTreeMap;
use std::mem;
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::{plain_summary_line, shorten, Impl, IndexItem, IndexItemFunctionType, ItemType};
use super::{RenderInfo, Type};

/// Indicates where an external crate can be found.
pub enum ExternalLocation {
    /// Remote URL root of the external crate
    Remote(String),
    /// This external crate can be found in the local doc/ folder
    Local,
    /// The external crate could not be found.
    Unknown,
}

/// This cache is used to store information about the `clean::Crate` being
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
    pub impls: FxHashMap<DefId, Vec<Impl>>,

    /// Maintains a mapping of local crate `DefId`s to the fully qualified name
    /// and "short type description" of that node. This is used when generating
    /// URLs when a type is being linked to. External paths are not located in
    /// this map because the `External` type itself has all the information
    /// necessary.
    pub paths: FxHashMap<DefId, (Vec<String>, ItemType)>,

    /// Similar to `paths`, but only holds external paths. This is only used for
    /// generating explicit hyperlinks to other crates.
    pub external_paths: FxHashMap<DefId, (Vec<String>, ItemType)>,

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
    pub exact_paths: FxHashMap<DefId, Vec<String>>,

    /// This map contains information about all known traits of this crate.
    /// Implementations of a crate should inherit the documentation of the
    /// parent trait if no extra documentation is specified, and default methods
    /// should show up in documentation about trait implementations.
    pub traits: FxHashMap<DefId, clean::Trait>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    pub implementors: FxHashMap<DefId, Vec<Impl>>,

    /// Cache of where external crate documentation can be found.
    pub extern_locations: FxHashMap<CrateNum, (String, PathBuf, ExternalLocation)>,

    /// Cache of where documentation for primitives can be found.
    pub primitive_locations: FxHashMap<clean::PrimitiveType, DefId>,

    // Note that external items for which `doc(hidden)` applies to are shown as
    // non-reachable while local items aren't. This is because we're reusing
    // the access levels from the privacy check pass.
    pub access_levels: AccessLevels<DefId>,

    /// The version of the crate being documented, if given from the `--crate-version` flag.
    pub crate_version: Option<String>,

    // Private fields only used when initially crawling a crate to build a cache
    stack: Vec<String>,
    parent_stack: Vec<DefId>,
    parent_is_trait_impl: bool,
    search_index: Vec<IndexItem>,
    stripped_mod: bool,
    pub deref_trait_did: Option<DefId>,
    pub deref_mut_trait_did: Option<DefId>,
    pub owned_box_did: Option<DefId>,
    masked_crates: FxHashSet<CrateNum>,

    // In rare case where a structure is defined in one module but implemented
    // in another, if the implementing module is parsed before defining module,
    // then the fully qualified name of the structure isn't presented in `paths`
    // yet when its implementation methods are being indexed. Caches such methods
    // and their parent id here and indexes them at the end of crate parsing.
    orphan_impl_items: Vec<(DefId, clean::Item)>,

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
    pub(super) aliases: FxHashMap<String, Vec<IndexItem>>,
}

impl Cache {
    pub fn from_krate(
        renderinfo: RenderInfo,
        extern_html_root_urls: &BTreeMap<String, String>,
        dst: &Path,
        mut krate: clean::Crate,
    ) -> (clean::Crate, String, Cache) {
        // Crawl the crate to build various caches used for the output
        let RenderInfo {
            inlined: _,
            external_paths,
            exact_paths,
            access_levels,
            deref_trait_did,
            deref_mut_trait_did,
            owned_box_did,
        } = renderinfo;

        let external_paths =
            external_paths.into_iter().map(|(k, (v, t))| (k, (v, ItemType::from(t)))).collect();

        let mut cache = Cache {
            impls: Default::default(),
            external_paths,
            exact_paths,
            paths: Default::default(),
            implementors: Default::default(),
            stack: Vec::new(),
            parent_stack: Vec::new(),
            search_index: Vec::new(),
            parent_is_trait_impl: false,
            extern_locations: Default::default(),
            primitive_locations: Default::default(),
            stripped_mod: false,
            access_levels,
            crate_version: krate.version.take(),
            orphan_impl_items: Vec::new(),
            orphan_trait_impls: Vec::new(),
            traits: krate.external_traits.replace(Default::default()),
            deref_trait_did,
            deref_mut_trait_did,
            owned_box_did,
            masked_crates: mem::take(&mut krate.masked_crates),
            aliases: Default::default(),
        };

        // Cache where all our extern crates are located
        for &(n, ref e) in &krate.externs {
            let src_root = match e.src {
                FileName::Real(ref p) => match p.parent() {
                    Some(p) => p.to_path_buf(),
                    None => PathBuf::new(),
                },
                _ => PathBuf::new(),
            };
            let extern_url = extern_html_root_urls.get(&e.name).map(|u| &**u);
            cache
                .extern_locations
                .insert(n, (e.name.clone(), src_root, extern_location(e, extern_url, &dst)));

            let did = DefId { krate: n, index: CRATE_DEF_INDEX };
            cache.external_paths.insert(did, (vec![e.name.to_string()], ItemType::Module));
        }

        // Cache where all known primitives have their documentation located.
        //
        // Favor linking to as local extern as possible, so iterate all crates in
        // reverse topological order.
        for &(_, ref e) in krate.externs.iter().rev() {
            for &(def_id, prim, _) in &e.primitives {
                cache.primitive_locations.insert(prim, def_id);
            }
        }
        for &(def_id, prim, _) in &krate.primitives {
            cache.primitive_locations.insert(prim, def_id);
        }

        cache.stack.push(krate.name.clone());
        krate = cache.fold_crate(krate);

        for (trait_did, dids, impl_) in cache.orphan_trait_impls.drain(..) {
            if cache.traits.contains_key(&trait_did) {
                for did in dids {
                    cache.impls.entry(did).or_insert(vec![]).push(impl_.clone());
                }
            }
        }

        // Build our search index
        let index = build_index(&krate, &mut cache);

        (krate, index, cache)
    }
}

impl DocFolder for Cache {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        if item.def_id.is_local() {
            debug!("folding {} \"{:?}\", id {:?}", item.type_(), item.name, item.def_id);
        }

        // If this is a stripped module,
        // we don't want it or its children in the search index.
        let orig_stripped_mod = match item.inner {
            clean::StrippedItem(box clean::ModuleItem(..)) => {
                mem::replace(&mut self.stripped_mod, true)
            }
            _ => self.stripped_mod,
        };

        // If the impl is from a masked crate or references something from a
        // masked crate then remove it completely.
        if let clean::ImplItem(ref i) = item.inner {
            if self.masked_crates.contains(&item.def_id.krate)
                || i.trait_.def_id().map_or(false, |d| self.masked_crates.contains(&d.krate))
                || i.for_.def_id().map_or(false, |d| self.masked_crates.contains(&d.krate))
            {
                return None;
            }
        }

        // Propagate a trait method's documentation to all implementors of the
        // trait.
        if let clean::TraitItem(ref t) = item.inner {
            self.traits.entry(item.def_id).or_insert_with(|| t.clone());
        }

        // Collect all the implementors of traits.
        if let clean::ImplItem(ref i) = item.inner {
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
            let (parent, is_inherent_impl_item) = match item.inner {
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
                            Some(&(ref fqp, ItemType::Trait))
                            | Some(&(ref fqp, ItemType::Struct))
                            | Some(&(ref fqp, ItemType::Union))
                            | Some(&(ref fqp, ItemType::Enum)) => Some(&fqp[..fqp.len() - 1]),
                            Some(..) => Some(&*self.stack),
                            None => None,
                        };
                        ((Some(*last), path), true)
                    }
                }
                _ => ((None, Some(&*self.stack)), false),
            };

            match parent {
                (parent, Some(path)) if is_inherent_impl_item || (!self.stripped_mod) => {
                    debug_assert!(!item.is_stripped());

                    // A crate has a module at its root, containing all items,
                    // which should not be indexed. The crate-item itself is
                    // inserted later on when serializing the search-index.
                    if item.def_id.index != CRATE_DEF_INDEX {
                        self.search_index.push(IndexItem {
                            ty: item.type_(),
                            name: s.to_string(),
                            path: path.join("::"),
                            desc: shorten(plain_summary_line(item.doc_value())),
                            parent,
                            parent_idx: None,
                            search_type: get_index_search_type(&item),
                        });
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
            Some(ref n) if !n.is_empty() => {
                self.stack.push(n.to_string());
                true
            }
            _ => false,
        };

        match item.inner {
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
                self.add_aliases(&item);
            }

            clean::PrimitiveItem(..) => {
                self.add_aliases(&item);
                self.paths.insert(item.def_id, (self.stack.clone(), item.type_()));
            }

            _ => {}
        }

        // Maintain the parent stack
        let orig_parent_is_trait_impl = self.parent_is_trait_impl;
        let parent_pushed = match item.inner {
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
        let ret = self.fold_item_recur(item).and_then(|item| {
            if let clean::Item { inner: clean::ImplItem(_), .. } = item {
                // Figure out the id of this impl. This may map to a
                // primitive rather than always to a struct/enum.
                // Note: matching twice to restrict the lifetime of the `i` borrow.
                let mut dids = FxHashSet::default();
                if let clean::Item { inner: clean::ImplItem(ref i), .. } = item {
                    match i.for_ {
                        clean::ResolvedPath { did, .. }
                        | clean::BorrowedRef {
                            type_: box clean::ResolvedPath { did, .. }, ..
                        } => {
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
            }
        });

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

impl Cache {
    fn add_aliases(&mut self, item: &clean::Item) {
        if item.def_id.index == CRATE_DEF_INDEX {
            return;
        }
        if let Some(ref item_name) = item.name {
            let path = self
                .paths
                .get(&item.def_id)
                .map(|p| p.0[..p.0.len() - 1].join("::"))
                .unwrap_or("std".to_owned());
            for alias in item
                .attrs
                .lists(sym::doc)
                .filter(|a| a.check_name(sym::alias))
                .filter_map(|a| a.value_str().map(|s| s.to_string().replace("\"", "")))
                .filter(|v| !v.is_empty())
                .collect::<FxHashSet<_>>()
                .into_iter()
            {
                self.aliases.entry(alias).or_insert(Vec::with_capacity(1)).push(IndexItem {
                    ty: item.type_(),
                    name: item_name.to_string(),
                    path: path.clone(),
                    desc: shorten(plain_summary_line(item.doc_value())),
                    parent: None,
                    parent_idx: None,
                    search_type: get_index_search_type(&item),
                });
            }
        }
    }
}

/// Attempts to find where an external crate is located, given that we're
/// rendering in to the specified source destination.
fn extern_location(
    e: &clean::ExternalCrate,
    extern_url: Option<&str>,
    dst: &Path,
) -> ExternalLocation {
    use ExternalLocation::*;
    // See if there's documentation generated into the local directory
    let local_location = dst.join(&e.name);
    if local_location.is_dir() {
        return Local;
    }

    if let Some(url) = extern_url {
        let mut url = url.to_string();
        if !url.ends_with('/') {
            url.push('/');
        }
        return Remote(url);
    }

    // Failing that, see if there's an attribute specifying where to find this
    // external crate
    e.attrs
        .lists(sym::doc)
        .filter(|a| a.check_name(sym::html_root_url))
        .filter_map(|a| a.value_str())
        .map(|url| {
            let mut url = url.to_string();
            if !url.ends_with('/') {
                url.push('/')
            }
            Remote(url)
        })
        .next()
        .unwrap_or(Unknown) // Well, at least we tried.
}

/// Builds the search index from the collected metadata
fn build_index(krate: &clean::Crate, cache: &mut Cache) -> String {
    let mut defid_to_pathid = FxHashMap::default();
    let mut crate_items = Vec::with_capacity(cache.search_index.len());
    let mut crate_paths = vec![];

    let Cache { ref mut search_index, ref orphan_impl_items, ref paths, .. } = *cache;

    // Attach all orphan items to the type's definition if the type
    // has since been learned.
    for &(did, ref item) in orphan_impl_items {
        if let Some(&(ref fqp, _)) = paths.get(&did) {
            search_index.push(IndexItem {
                ty: item.type_(),
                name: item.name.clone().unwrap(),
                path: fqp[..fqp.len() - 1].join("::"),
                desc: shorten(plain_summary_line(item.doc_value())),
                parent: Some(did),
                parent_idx: None,
                search_type: get_index_search_type(&item),
            });
        }
    }

    // Reduce `DefId` in paths into smaller sequential numbers,
    // and prune the paths that do not appear in the index.
    let mut lastpath = String::new();
    let mut lastpathid = 0usize;

    for item in search_index {
        item.parent_idx = item.parent.map(|defid| {
            if defid_to_pathid.contains_key(&defid) {
                *defid_to_pathid.get(&defid).expect("no pathid")
            } else {
                let pathid = lastpathid;
                defid_to_pathid.insert(defid, pathid);
                lastpathid += 1;

                let &(ref fqp, short) = paths.get(&defid).unwrap();
                crate_paths.push((short, fqp.last().unwrap().clone()));
                pathid
            }
        });

        // Omit the parent path if it is same to that of the prior item.
        if lastpath == item.path {
            item.path.clear();
        } else {
            lastpath = item.path.clone();
        }
        crate_items.push(&*item);
    }

    let crate_doc = krate
        .module
        .as_ref()
        .map(|module| shorten(plain_summary_line(module.doc_value())))
        .unwrap_or(String::new());

    #[derive(Serialize)]
    struct CrateData<'a> {
        doc: String,
        #[serde(rename = "i")]
        items: Vec<&'a IndexItem>,
        #[serde(rename = "p")]
        paths: Vec<(ItemType, String)>,
    }

    // Collect the index into a string
    format!(
        r#"searchIndex["{}"] = {};"#,
        krate.name,
        serde_json::to_string(&CrateData {
            doc: crate_doc,
            items: crate_items,
            paths: crate_paths,
        })
        .expect("failed serde conversion")
    )
}

fn get_index_search_type(item: &clean::Item) -> Option<IndexItemFunctionType> {
    let (all_types, ret_types) = match item.inner {
        clean::FunctionItem(ref f) => (&f.all_types, &f.ret_types),
        clean::MethodItem(ref m) => (&m.all_types, &m.ret_types),
        clean::TyMethodItem(ref m) => (&m.all_types, &m.ret_types),
        _ => return None,
    };

    let inputs =
        all_types.iter().map(|arg| get_index_type(&arg)).filter(|a| a.name.is_some()).collect();
    let output = ret_types
        .iter()
        .map(|arg| get_index_type(&arg))
        .filter(|a| a.name.is_some())
        .collect::<Vec<_>>();
    let output = if output.is_empty() { None } else { Some(output) };

    Some(IndexItemFunctionType { inputs, output })
}

fn get_index_type(clean_type: &clean::Type) -> Type {
    let t = Type {
        name: get_index_type_name(clean_type, true).map(|s| s.to_ascii_lowercase()),
        generics: get_generics(clean_type),
    };
    t
}

fn get_index_type_name(clean_type: &clean::Type, accept_generic: bool) -> Option<String> {
    match *clean_type {
        clean::ResolvedPath { ref path, .. } => {
            let segments = &path.segments;
            let path_segment = segments.iter().last().unwrap_or_else(|| panic!(
                "get_index_type_name(clean_type: {:?}, accept_generic: {:?}) had length zero path",
                clean_type, accept_generic
            ));
            Some(path_segment.name.clone())
        }
        clean::Generic(ref s) if accept_generic => Some(s.clone()),
        clean::Primitive(ref p) => Some(format!("{:?}", p)),
        clean::BorrowedRef { ref type_, .. } => get_index_type_name(type_, accept_generic),
        // FIXME: add all from clean::Type.
        _ => None,
    }
}

fn get_generics(clean_type: &clean::Type) -> Option<Vec<String>> {
    clean_type.generics().and_then(|types| {
        let r = types
            .iter()
            .filter_map(|t| get_index_type_name(t, false))
            .map(|s| s.to_ascii_lowercase())
            .collect::<Vec<_>>();
        if r.is_empty() { None } else { Some(r) }
    })
}
