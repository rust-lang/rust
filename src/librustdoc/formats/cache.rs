use std::mem;

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, DefIdSet};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Symbol;

use crate::clean::{self, types::ExternalLocation, ExternalCrate, ItemId, PrimitiveType};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::formats::item_type::ItemType;
use crate::formats::Impl;
use crate::html::format::join_with_double_colon;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::search_index::get_function_type_for_search;
use crate::html::render::IndexItem;
use crate::visit_lib::RustdocEffectiveVisibilities;

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
pub(crate) struct Cache {
    /// Maps a type ID to all known implementations for that type. This is only
    /// recognized for intra-crate [`clean::Type::Path`]s, and is used to print
    /// out extra documentation on the page of an enum/struct.
    ///
    /// The values of the map are a list of implementations and documentation
    /// found on that implementation.
    pub(crate) impls: DefIdMap<Vec<Impl>>,

    /// Maintains a mapping of local crate `DefId`s to the fully qualified name
    /// and "short type description" of that node. This is used when generating
    /// URLs when a type is being linked to. External paths are not located in
    /// this map because the `External` type itself has all the information
    /// necessary.
    pub(crate) paths: FxHashMap<DefId, (Vec<Symbol>, ItemType)>,

    /// Similar to `paths`, but only holds external paths. This is only used for
    /// generating explicit hyperlinks to other crates.
    pub(crate) external_paths: FxHashMap<DefId, (Vec<Symbol>, ItemType)>,

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
    pub(crate) exact_paths: DefIdMap<Vec<Symbol>>,

    /// This map contains information about all known traits of this crate.
    /// Implementations of a crate should inherit the documentation of the
    /// parent trait if no extra documentation is specified, and default methods
    /// should show up in documentation about trait implementations.
    pub(crate) traits: FxHashMap<DefId, clean::Trait>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    pub(crate) implementors: FxHashMap<DefId, Vec<Impl>>,

    /// Cache of where external crate documentation can be found.
    pub(crate) extern_locations: FxHashMap<CrateNum, ExternalLocation>,

    /// Cache of where documentation for primitives can be found.
    pub(crate) primitive_locations: FxHashMap<clean::PrimitiveType, DefId>,

    // Note that external items for which `doc(hidden)` applies to are shown as
    // non-reachable while local items aren't. This is because we're reusing
    // the effective visibilities from the privacy check pass.
    pub(crate) effective_visibilities: RustdocEffectiveVisibilities,

    /// The version of the crate being documented, if given from the `--crate-version` flag.
    pub(crate) crate_version: Option<String>,

    /// Whether to document private items.
    /// This is stored in `Cache` so it doesn't need to be passed through all rustdoc functions.
    pub(crate) document_private: bool,

    /// Crates marked with [`#[doc(masked)]`][doc_masked].
    ///
    /// [doc_masked]: https://doc.rust-lang.org/nightly/unstable-book/language-features/doc-masked.html
    pub(crate) masked_crates: FxHashSet<CrateNum>,

    // Private fields only used when initially crawling a crate to build a cache
    stack: Vec<Symbol>,
    parent_stack: Vec<ParentStackItem>,
    stripped_mod: bool,

    pub(crate) search_index: Vec<IndexItem>,

    // In rare case where a structure is defined in one module but implemented
    // in another, if the implementing module is parsed before defining module,
    // then the fully qualified name of the structure isn't presented in `paths`
    // yet when its implementation methods are being indexed. Caches such methods
    // and their parent id here and indexes them at the end of crate parsing.
    pub(crate) orphan_impl_items: Vec<OrphanImplItem>,

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
    pub(crate) intra_doc_links: FxHashMap<ItemId, FxIndexSet<clean::ItemLink>>,
    /// Cfg that have been hidden via #![doc(cfg_hide(...))]
    pub(crate) hidden_cfg: FxHashSet<clean::cfg::Cfg>,

    /// Contains the list of `DefId`s which have been inlined. It is used when generating files
    /// to check if a stripped item should get its file generated or not: if it's inside a
    /// `#[doc(hidden)]` item or a private one and not inlined, it shouldn't get a file.
    pub(crate) inlined_items: DefIdSet,
}

/// This struct is used to wrap the `cache` and `tcx` in order to run `DocFolder`.
struct CacheBuilder<'a, 'tcx> {
    cache: &'a mut Cache,
    /// This field is used to prevent duplicated impl blocks.
    impl_ids: DefIdMap<DefIdSet>,
    tcx: TyCtxt<'tcx>,
}

impl Cache {
    pub(crate) fn new(document_private: bool) -> Self {
        Cache { document_private, ..Cache::default() }
    }

    /// Populates the `Cache` with more data. The returned `Crate` will be missing some data that was
    /// in `krate` due to the data being moved into the `Cache`.
    pub(crate) fn populate(cx: &mut DocContext<'_>, mut krate: clean::Crate) -> clean::Crate {
        let tcx = cx.tcx;

        // Crawl the crate to build various caches used for the output
        debug!(?cx.cache.crate_version);
        cx.cache.traits = krate.external_traits.take();

        // Cache where all our extern crates are located
        // FIXME: this part is specific to HTML so it'd be nice to remove it from the common code
        for &crate_num in tcx.crates(()) {
            let e = ExternalCrate { crate_num };

            let name = e.name(tcx);
            let render_options = &cx.render_options;
            let extern_url = render_options.extern_html_root_urls.get(name.as_str()).map(|u| &**u);
            let extern_url_takes_precedence = render_options.extern_html_root_takes_precedence;
            let dst = &render_options.output;
            let location = e.location(extern_url, extern_url_takes_precedence, dst, tcx);
            cx.cache.extern_locations.insert(e.crate_num, location);
            cx.cache.external_paths.insert(e.def_id(), (vec![name], ItemType::Module));
        }

        // FIXME: avoid this clone (requires implementing Default manually)
        cx.cache.primitive_locations = PrimitiveType::primitive_locations(tcx).clone();
        for (prim, &def_id) in &cx.cache.primitive_locations {
            let crate_name = tcx.crate_name(def_id.krate);
            // Recall that we only allow primitive modules to be at the root-level of the crate.
            // If that restriction is ever lifted, this will have to include the relative paths instead.
            cx.cache
                .external_paths
                .insert(def_id, (vec![crate_name, prim.as_sym()], ItemType::Primitive));
        }

        let (krate, mut impl_ids) = {
            let mut cache_builder =
                CacheBuilder { tcx, cache: &mut cx.cache, impl_ids: Default::default() };
            krate = cache_builder.fold_crate(krate);
            (krate, cache_builder.impl_ids)
        };

        for (trait_did, dids, impl_) in cx.cache.orphan_trait_impls.drain(..) {
            if cx.cache.traits.contains_key(&trait_did) {
                for did in dids {
                    if impl_ids.entry(did).or_default().insert(impl_.def_id()) {
                        cx.cache.impls.entry(did).or_default().push(impl_.clone());
                    }
                }
            }
        }

        krate
    }
}

impl<'a, 'tcx> DocFolder for CacheBuilder<'a, 'tcx> {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        if item.item_id.is_local() {
            let is_stripped = matches!(*item.kind, clean::ItemKind::StrippedItem(..));
            debug!(
                "folding {} (stripped: {is_stripped:?}) \"{:?}\", id {:?}",
                item.type_(),
                item.name,
                item.item_id
            );
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
            if self.cache.masked_crates.contains(&item.item_id.krate())
                || i.trait_
                    .as_ref()
                    .map_or(false, |t| self.cache.masked_crates.contains(&t.def_id().krate))
                || i.for_
                    .def_id(self.cache)
                    .map_or(false, |d| self.cache.masked_crates.contains(&d.krate))
            {
                return None;
            }
        }

        // Propagate a trait method's documentation to all implementors of the
        // trait.
        if let clean::TraitItem(ref t) = *item.kind {
            self.cache.traits.entry(item.item_id.expect_def_id()).or_insert_with(|| (**t).clone());
        }

        // Collect all the implementors of traits.
        if let clean::ImplItem(ref i) = *item.kind &&
            let Some(trait_) = &i.trait_ &&
            !i.kind.is_blanket()
        {
            self.cache
                .implementors
                .entry(trait_.def_id())
                .or_default()
                .push(Impl { impl_item: item.clone() });
        }

        // Index this method for searching later on.
        if let Some(s) = item.name.or_else(|| {
            if item.is_stripped() {
                None
            } else if let clean::ImportItem(ref i) = *item.kind &&
                let clean::ImportKind::Simple(s) = i.kind {
                Some(s)
            } else {
                None
            }
        }) {
            let (parent, is_inherent_impl_item) = match *item.kind {
                clean::StrippedItem(..) => ((None, None), false),
                clean::AssocConstItem(..) | clean::AssocTypeItem(..)
                    if self
                        .cache
                        .parent_stack
                        .last()
                        .map_or(false, |parent| parent.is_trait_impl()) =>
                {
                    // skip associated items in trait impls
                    ((None, None), false)
                }
                clean::TyMethodItem(..)
                | clean::TyAssocConstItem(..)
                | clean::TyAssocTypeItem(..)
                | clean::StructFieldItem(..)
                | clean::VariantItem(..) => (
                    (
                        Some(
                            self.cache
                                .parent_stack
                                .last()
                                .expect("parent_stack is empty")
                                .item_id()
                                .expect_def_id(),
                        ),
                        Some(&self.cache.stack[..self.cache.stack.len() - 1]),
                    ),
                    false,
                ),
                clean::MethodItem(..) | clean::AssocConstItem(..) | clean::AssocTypeItem(..) => {
                    if self.cache.parent_stack.is_empty() {
                        ((None, None), false)
                    } else {
                        let last = self.cache.parent_stack.last().expect("parent_stack is empty 2");
                        let did = match &*last {
                            ParentStackItem::Impl {
                                // impl Trait for &T { fn method(self); }
                                //
                                // When generating a function index with the above shape, we want it
                                // associated with `T`, not with the primitive reference type. It should
                                // show up as `T::method`, rather than `reference::method`, in the search
                                // results page.
                                for_: clean::Type::BorrowedRef { type_, .. },
                                ..
                            } => type_.def_id(&self.cache),
                            ParentStackItem::Impl { for_, .. } => for_.def_id(&self.cache),
                            ParentStackItem::Type(item_id) => item_id.as_def_id(),
                        };
                        let path = did
                            .and_then(|did| self.cache.paths.get(&did))
                            // The current stack not necessarily has correlation
                            // for where the type was defined. On the other
                            // hand, `paths` always has the right
                            // information if present.
                            .map(|(fqp, _)| &fqp[..fqp.len() - 1]);
                        ((did, path), true)
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
                    if item.item_id.as_def_id().map_or(false, |idx| !idx.is_crate_root()) {
                        let desc =
                            short_markdown_summary(&item.doc_value(), &item.link_names(self.cache));
                        let ty = item.type_();
                        if ty != ItemType::StructField
                            || u16::from_str_radix(s.as_str(), 10).is_err()
                        {
                            // In case this is a field from a tuple struct, we don't add it into
                            // the search index because its name is something like "0", which is
                            // not useful for rustdoc search.
                            self.cache.search_index.push(IndexItem {
                                ty,
                                name: s,
                                path: join_with_double_colon(path),
                                desc,
                                parent,
                                parent_idx: None,
                                search_type: get_function_type_for_search(
                                    &item,
                                    self.tcx,
                                    clean_impl_generics(self.cache.parent_stack.last()).as_ref(),
                                    self.cache,
                                ),
                                aliases: item.attrs.get_doc_aliases(),
                                deprecation: item.deprecation(self.tcx),
                            });
                        }
                    }
                }
                (Some(parent), None) if is_inherent_impl_item => {
                    // We have a parent, but we don't know where they're
                    // defined yet. Wait for later to index this item.
                    let impl_generics = clean_impl_generics(self.cache.parent_stack.last());
                    self.cache.orphan_impl_items.push(OrphanImplItem {
                        parent,
                        item: item.clone(),
                        impl_generics,
                    });
                }
                _ => {}
            }
        }

        // Keep track of the fully qualified path for this item.
        let pushed = match item.name {
            Some(n) if !n.is_empty() => {
                self.cache.stack.push(n);
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
                    if !self.cache.paths.contains_key(&item.item_id.expect_def_id())
                        || self
                            .cache
                            .effective_visibilities
                            .is_directly_public(self.tcx, item.item_id.expect_def_id())
                    {
                        self.cache.paths.insert(
                            item.item_id.expect_def_id(),
                            (self.cache.stack.clone(), item.type_()),
                        );
                    }
                }
            }
            clean::PrimitiveItem(..) => {
                self.cache
                    .paths
                    .insert(item.item_id.expect_def_id(), (self.cache.stack.clone(), item.type_()));
            }

            clean::ExternCrateItem { .. }
            | clean::ImportItem(..)
            | clean::OpaqueTyItem(..)
            | clean::ImplItem(..)
            | clean::TyMethodItem(..)
            | clean::MethodItem(..)
            | clean::StructFieldItem(..)
            | clean::TyAssocConstItem(..)
            | clean::AssocConstItem(..)
            | clean::TyAssocTypeItem(..)
            | clean::AssocTypeItem(..)
            | clean::StrippedItem(..)
            | clean::KeywordItem => {
                // FIXME: Do these need handling?
                // The person writing this comment doesn't know.
                // So would rather leave them to an expert,
                // as at least the list is better than `_ => {}`.
            }
        }

        // Maintain the parent stack.
        let (item, parent_pushed) = match *item.kind {
            clean::TraitItem(..)
            | clean::EnumItem(..)
            | clean::ForeignTypeItem
            | clean::StructItem(..)
            | clean::UnionItem(..)
            | clean::VariantItem(..)
            | clean::ImplItem(..) => {
                self.cache.parent_stack.push(ParentStackItem::new(&item));
                (self.fold_item_recur(item), true)
            }
            _ => (self.fold_item_recur(item), false),
        };

        // Once we've recursively found all the generics, hoard off all the
        // implementations elsewhere.
        let ret = if let clean::Item { kind: box clean::ImplItem(ref i), .. } = item {
            // Figure out the id of this impl. This may map to a
            // primitive rather than always to a struct/enum.
            // Note: matching twice to restrict the lifetime of the `i` borrow.
            let mut dids = FxHashSet::default();
            match i.for_ {
                clean::Type::Path { ref path }
                | clean::BorrowedRef { type_: box clean::Type::Path { ref path }, .. } => {
                    dids.insert(path.def_id());
                    if let Some(generics) = path.generics() &&
                        let ty::Adt(adt, _) = self.tcx.type_of(path.def_id()).subst_identity().kind() &&
                        adt.is_fundamental() {
                        for ty in generics {
                            if let Some(did) = ty.def_id(self.cache) {
                                dids.insert(did);
                            }
                        }
                    }
                }
                clean::DynTrait(ref bounds, _)
                | clean::BorrowedRef { type_: box clean::DynTrait(ref bounds, _), .. } => {
                    dids.insert(bounds[0].trait_.def_id());
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
                    if let Some(did) = bound.def_id(self.cache) {
                        dids.insert(did);
                    }
                }
            }
            let impl_item = Impl { impl_item: item };
            if impl_item.trait_did().map_or(true, |d| self.cache.traits.contains_key(&d)) {
                for did in dids {
                    if self.impl_ids.entry(did).or_default().insert(impl_item.def_id()) {
                        self.cache
                            .impls
                            .entry(did)
                            .or_insert_with(Vec::new)
                            .push(impl_item.clone());
                    }
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
        ret
    }
}

pub(crate) struct OrphanImplItem {
    pub(crate) parent: DefId,
    pub(crate) item: clean::Item,
    pub(crate) impl_generics: Option<(clean::Type, clean::Generics)>,
}

/// Information about trait and type parents is tracked while traversing the item tree to build
/// the cache.
///
/// We don't just store `Item` in there, because `Item` contains the list of children being
/// traversed and it would be wasteful to clone all that. We also need the item id, so just
/// storing `ItemKind` won't work, either.
enum ParentStackItem {
    Impl {
        for_: clean::Type,
        trait_: Option<clean::Path>,
        generics: clean::Generics,
        kind: clean::ImplKind,
        item_id: ItemId,
    },
    Type(ItemId),
}

impl ParentStackItem {
    fn new(item: &clean::Item) -> Self {
        match &*item.kind {
            clean::ItemKind::ImplItem(box clean::Impl { for_, trait_, generics, kind, .. }) => {
                ParentStackItem::Impl {
                    for_: for_.clone(),
                    trait_: trait_.clone(),
                    generics: generics.clone(),
                    kind: kind.clone(),
                    item_id: item.item_id,
                }
            }
            _ => ParentStackItem::Type(item.item_id),
        }
    }
    fn is_trait_impl(&self) -> bool {
        matches!(self, ParentStackItem::Impl { trait_: Some(..), .. })
    }
    fn item_id(&self) -> ItemId {
        match self {
            ParentStackItem::Impl { item_id, .. } => *item_id,
            ParentStackItem::Type(item_id) => *item_id,
        }
    }
}

fn clean_impl_generics(item: Option<&ParentStackItem>) -> Option<(clean::Type, clean::Generics)> {
    if let Some(ParentStackItem::Impl { for_, generics, kind: clean::ImplKind::Normal, .. }) = item
    {
        Some((for_.clone(), generics.clone()))
    } else {
        None
    }
}
