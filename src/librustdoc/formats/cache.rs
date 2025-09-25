use std::mem;

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_hir::StabilityLevel;
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, DefIdSet};
use rustc_metadata::creader::CStore;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Symbol;
use tracing::debug;

use crate::clean::types::ExternalLocation;
use crate::clean::{self, ExternalCrate, ItemId, PrimitiveType};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::formats::Impl;
use crate::formats::item_type::ItemType;
use crate::html::markdown::short_markdown_summary;
use crate::html::render::IndexItem;
use crate::html::render::search_index::get_function_type_for_search;
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
    pub(crate) paths: FxIndexMap<DefId, (Vec<Symbol>, ItemType)>,

    /// Similar to `paths`, but only holds external paths. This is only used for
    /// generating explicit hyperlinks to other crates.
    pub(crate) external_paths: FxIndexMap<DefId, (Vec<Symbol>, ItemType)>,

    /// Maps local `DefId`s of exported types to fully qualified paths.
    /// Unlike 'paths', this mapping ignores any renames that occur
    /// due to 'use' statements.
    ///
    /// This map is used when writing out the `impl.trait` and `impl.type`
    /// javascript files. By using the exact path that the type
    /// is declared with, we ensure that each path will be identical
    /// to the path used if the corresponding type is inlined. By
    /// doing this, we can detect duplicate impls on a trait page, and only display
    /// the impl for the inlined type.
    pub(crate) exact_paths: DefIdMap<Vec<Symbol>>,

    /// This map contains information about all known traits of this crate.
    /// Implementations of a crate should inherit the documentation of the
    /// parent trait if no extra documentation is specified, and default methods
    /// should show up in documentation about trait implementations.
    pub(crate) traits: FxIndexMap<DefId, clean::Trait>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    pub(crate) implementors: FxIndexMap<DefId, Vec<Impl>>,

    /// Cache of where external crate documentation can be found.
    pub(crate) extern_locations: FxIndexMap<CrateNum, ExternalLocation>,

    /// Cache of where documentation for primitives can be found.
    pub(crate) primitive_locations: FxIndexMap<clean::PrimitiveType, DefId>,

    // Note that external items for which `doc(hidden)` applies to are shown as
    // non-reachable while local items aren't. This is because we're reusing
    // the effective visibilities from the privacy check pass.
    pub(crate) effective_visibilities: RustdocEffectiveVisibilities,

    /// The version of the crate being documented, if given from the `--crate-version` flag.
    pub(crate) crate_version: Option<String>,

    /// Whether to document private items.
    /// This is stored in `Cache` so it doesn't need to be passed through all rustdoc functions.
    pub(crate) document_private: bool,
    /// Whether to document hidden items.
    /// This is stored in `Cache` so it doesn't need to be passed through all rustdoc functions.
    pub(crate) document_hidden: bool,

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
    orphan_trait_impls: Vec<(DefId, FxIndexSet<DefId>, Impl)>,

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
    is_json_output: bool,
}

impl Cache {
    pub(crate) fn new(document_private: bool, document_hidden: bool) -> Self {
        Cache { document_private, document_hidden, ..Cache::default() }
    }

    /// Populates the `Cache` with more data. The returned `Crate` will be missing some data that was
    /// in `krate` due to the data being moved into the `Cache`.
    pub(crate) fn populate(cx: &mut DocContext<'_>, mut krate: clean::Crate) -> clean::Crate {
        let tcx = cx.tcx;

        // Crawl the crate to build various caches used for the output
        debug!(?cx.cache.crate_version);
        assert!(cx.external_traits.is_empty());
        cx.cache.traits = mem::take(&mut krate.external_traits);

        let render_options = &cx.render_options;
        let extern_url_takes_precedence = render_options.extern_html_root_takes_precedence;
        let dst = &render_options.output;

        // Make `--extern-html-root-url` support the same names as `--extern` whenever possible
        let cstore = CStore::from_tcx(tcx);
        for (name, extern_url) in &render_options.extern_html_root_urls {
            if let Some(crate_num) = cstore.resolved_extern_crate(Symbol::intern(name)) {
                let e = ExternalCrate { crate_num };
                let location = e.location(Some(extern_url), extern_url_takes_precedence, dst, tcx);
                cx.cache.extern_locations.insert(e.crate_num, location);
            }
        }

        // Cache where all our extern crates are located
        // This is also used in the JSON output.
        for &crate_num in tcx.crates(()) {
            let e = ExternalCrate { crate_num };

            let name = e.name(tcx);
            cx.cache.extern_locations.entry(e.crate_num).or_insert_with(|| {
                // falls back to matching by crates' own names, because
                // transitive dependencies and injected crates may be loaded without `--extern`
                let extern_url =
                    render_options.extern_html_root_urls.get(name.as_str()).map(|u| &**u);
                e.location(extern_url, extern_url_takes_precedence, dst, tcx)
            });
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
            let is_json_output = cx.is_json_output();
            let mut cache_builder = CacheBuilder {
                tcx,
                cache: &mut cx.cache,
                impl_ids: Default::default(),
                is_json_output,
            };
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

impl DocFolder for CacheBuilder<'_, '_> {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        if item.item_id.is_local() {
            debug!(
                "folding {} (stripped: {:?}) \"{:?}\", id {:?}",
                item.type_(),
                item.is_stripped(),
                item.name,
                item.item_id
            );
        }

        // If this is a stripped module,
        // we don't want it or its children in the search index.
        let orig_stripped_mod = match item.kind {
            clean::StrippedItem(box clean::ModuleItem(..)) => {
                mem::replace(&mut self.cache.stripped_mod, true)
            }
            _ => self.cache.stripped_mod,
        };

        #[inline]
        fn is_from_private_dep(tcx: TyCtxt<'_>, cache: &Cache, def_id: DefId) -> bool {
            let krate = def_id.krate;

            cache.masked_crates.contains(&krate) || tcx.is_private_dep(krate)
        }

        // If the impl is from a masked crate or references something from a
        // masked crate then remove it completely.
        if let clean::ImplItem(ref i) = item.kind
            && (self.cache.masked_crates.contains(&item.item_id.krate())
                || i.trait_
                    .as_ref()
                    .is_some_and(|t| is_from_private_dep(self.tcx, self.cache, t.def_id()))
                || i.for_
                    .def_id(self.cache)
                    .is_some_and(|d| is_from_private_dep(self.tcx, self.cache, d)))
        {
            return None;
        }

        // Propagate a trait method's documentation to all implementors of the
        // trait.
        if let clean::TraitItem(ref t) = item.kind {
            self.cache.traits.entry(item.item_id.expect_def_id()).or_insert_with(|| (**t).clone());
        } else if let clean::ImplItem(ref i) = item.kind
            && let Some(trait_) = &i.trait_
            && !i.kind.is_blanket()
        {
            // Collect all the implementors of traits.
            self.cache
                .implementors
                .entry(trait_.def_id())
                .or_default()
                .push(Impl { impl_item: item.clone() });
        }

        // Index this method for searching later on.
        let search_name = if !item.is_stripped() {
            item.name.or_else(|| {
                if let clean::ImportItem(ref i) = item.kind
                    && let clean::ImportKind::Simple(s) = i.kind
                {
                    Some(s)
                } else {
                    None
                }
            })
        } else {
            None
        };
        if let Some(name) = search_name {
            add_item_to_search_index(self.tcx, self.cache, &item, name)
        }

        // Keep track of the fully qualified path for this item.
        let pushed = match item.name {
            Some(n) => {
                self.cache.stack.push(n);
                true
            }
            _ => false,
        };

        match item.kind {
            clean::StructItem(..)
            | clean::EnumItem(..)
            | clean::TypeAliasItem(..)
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
                use rustc_data_structures::fx::IndexEntry as Entry;

                let skip_because_unstable = matches!(
                    item.stability.map(|stab| stab.level),
                    Some(StabilityLevel::Stable { allowed_through_unstable_modules: Some(_), .. })
                );

                if (!self.cache.stripped_mod && !skip_because_unstable) || self.is_json_output {
                    // Re-exported items mean that the same id can show up twice
                    // in the rustdoc ast that we're looking at. We know,
                    // however, that a re-exported item doesn't show up in the
                    // `public_items` map, so we can skip inserting into the
                    // paths map if there was already an entry present and we're
                    // not a public item.
                    let item_def_id = item.item_id.expect_def_id();
                    match self.cache.paths.entry(item_def_id) {
                        Entry::Vacant(entry) => {
                            entry.insert((self.cache.stack.clone(), item.type_()));
                        }
                        Entry::Occupied(mut entry) => {
                            if entry.get().0.len() > self.cache.stack.len() {
                                entry.insert((self.cache.stack.clone(), item.type_()));
                            }
                        }
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
            | clean::ImplItem(..)
            | clean::RequiredMethodItem(..)
            | clean::MethodItem(..)
            | clean::StructFieldItem(..)
            | clean::RequiredAssocConstItem(..)
            | clean::ProvidedAssocConstItem(..)
            | clean::ImplAssocConstItem(..)
            | clean::RequiredAssocTypeItem(..)
            | clean::AssocTypeItem(..)
            | clean::StrippedItem(..)
            | clean::KeywordItem
            | clean::AttributeItem => {
                // FIXME: Do these need handling?
                // The person writing this comment doesn't know.
                // So would rather leave them to an expert,
                // as at least the list is better than `_ => {}`.
            }
        }

        // Maintain the parent stack.
        let (item, parent_pushed) = match item.kind {
            clean::TraitItem(..)
            | clean::EnumItem(..)
            | clean::ForeignTypeItem
            | clean::StructItem(..)
            | clean::UnionItem(..)
            | clean::VariantItem(..)
            | clean::TypeAliasItem(..)
            | clean::ImplItem(..) => {
                self.cache.parent_stack.push(ParentStackItem::new(&item));
                (self.fold_item_recur(item), true)
            }
            _ => (self.fold_item_recur(item), false),
        };

        // Once we've recursively found all the generics, hoard off all the
        // implementations elsewhere.
        let ret = if let clean::Item {
            inner: box clean::ItemInner { kind: clean::ImplItem(ref i), .. },
        } = item
        {
            // Figure out the id of this impl. This may map to a
            // primitive rather than always to a struct/enum.
            // Note: matching twice to restrict the lifetime of the `i` borrow.
            let mut dids = FxIndexSet::default();
            match i.for_ {
                clean::Type::Path { ref path }
                | clean::BorrowedRef { type_: box clean::Type::Path { ref path }, .. } => {
                    dids.insert(path.def_id());
                    if let Some(generics) = path.generics()
                        && let ty::Adt(adt, _) =
                            self.tcx.type_of(path.def_id()).instantiate_identity().kind()
                        && adt.is_fundamental()
                    {
                        for ty in generics {
                            dids.extend(ty.def_id(self.cache));
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

                    dids.extend(did);
                }
            }

            if let Some(trait_) = &i.trait_
                && let Some(generics) = trait_.generics()
            {
                for bound in generics {
                    dids.extend(bound.def_id(self.cache));
                }
            }
            let impl_item = Impl { impl_item: item };
            let impl_did = impl_item.def_id();
            let trait_did = impl_item.trait_did();
            if trait_did.is_none_or(|d| self.cache.traits.contains_key(&d)) {
                for did in dids {
                    if self.impl_ids.entry(did).or_default().insert(impl_did) {
                        self.cache.impls.entry(did).or_default().push(impl_item.clone());
                    }
                }
            } else {
                let trait_did = trait_did.expect("no trait did");
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

fn add_item_to_search_index(tcx: TyCtxt<'_>, cache: &mut Cache, item: &clean::Item, name: Symbol) {
    // Item has a name, so it must also have a DefId (can't be an impl, let alone a blanket or auto impl).
    let item_def_id = item.item_id.as_def_id().unwrap();
    let (parent_did, parent_path) = match item.kind {
        clean::StrippedItem(..) => return,
        clean::ProvidedAssocConstItem(..)
        | clean::ImplAssocConstItem(..)
        | clean::AssocTypeItem(..)
            if cache.parent_stack.last().is_some_and(|parent| parent.is_trait_impl()) =>
        {
            // skip associated items in trait impls
            return;
        }
        clean::RequiredMethodItem(..)
        | clean::RequiredAssocConstItem(..)
        | clean::RequiredAssocTypeItem(..)
        | clean::StructFieldItem(..)
        | clean::VariantItem(..) => {
            // Don't index if containing module is stripped (i.e., private),
            // or if item is tuple struct/variant field (name is a number -> not useful for search).
            if cache.stripped_mod
                || item.type_() == ItemType::StructField
                    && name.as_str().chars().all(|c| c.is_ascii_digit())
            {
                return;
            }
            let parent_did =
                cache.parent_stack.last().expect("parent_stack is empty").item_id().expect_def_id();
            let parent_path = &cache.stack[..cache.stack.len() - 1];
            (Some(parent_did), parent_path)
        }
        clean::MethodItem(..)
        | clean::ProvidedAssocConstItem(..)
        | clean::ImplAssocConstItem(..)
        | clean::AssocTypeItem(..) => {
            let last = cache.parent_stack.last().expect("parent_stack is empty 2");
            let parent_did = match last {
                // impl Trait for &T { fn method(self); }
                //
                // When generating a function index with the above shape, we want it
                // associated with `T`, not with the primitive reference type. It should
                // show up as `T::method`, rather than `reference::method`, in the search
                // results page.
                ParentStackItem::Impl { for_: clean::Type::BorrowedRef { type_, .. }, .. } => {
                    type_.def_id(cache)
                }
                ParentStackItem::Impl { for_, .. } => for_.def_id(cache),
                ParentStackItem::Type(item_id) => item_id.as_def_id(),
            };
            let Some(parent_did) = parent_did else { return };
            // The current stack reflects the CacheBuilder's recursive
            // walk over HIR. For associated items, this is the module
            // where the `impl` block is defined. That's an implementation
            // detail that we don't want to affect the search engine.
            //
            // In particular, you can arrange things like this:
            //
            //     #![crate_name="me"]
            //     mod private_mod {
            //         impl Clone for MyThing { fn clone(&self) -> MyThing { MyThing } }
            //     }
            //     pub struct MyThing;
            //
            // When that happens, we need to:
            // - ignore the `cache.stripped_mod` flag, since the Clone impl is actually
            //   part of the public API even though it's defined in a private module
            // - present the method as `me::MyThing::clone`, its publicly-visible path
            // - deal with the fact that the recursive walk hasn't actually reached `MyThing`
            //   until it's already past `private_mod`, since that's first, and doesn't know
            //   yet if `MyThing` will actually be public or not (it could be re-exported)
            //
            // We accomplish the last two points by recording children of "orphan impls"
            // in a field of the cache whose elements are added to the search index later,
            // after cache building is complete (see `handle_orphan_impl_child`).
            match cache.paths.get(&parent_did) {
                Some((fqp, _)) => (Some(parent_did), &fqp[..fqp.len() - 1]),
                None => {
                    handle_orphan_impl_child(cache, item, parent_did);
                    return;
                }
            }
        }
        _ => {
            // Don't index if item is crate root, which is inserted later on when serializing the index.
            // Don't index if containing module is stripped (i.e., private),
            if item_def_id.is_crate_root() || cache.stripped_mod {
                return;
            }
            (None, &*cache.stack)
        }
    };

    debug_assert!(!item.is_stripped());

    let desc = short_markdown_summary(&item.doc_value(), &item.link_names(cache));
    // For searching purposes, a re-export is a duplicate if:
    //
    // - It's either an inline, or a true re-export
    // - It's got the same name
    // - Both of them have the same exact path
    let defid = match &item.kind {
        clean::ItemKind::ImportItem(import) => import.source.did.unwrap_or(item_def_id),
        _ => item_def_id,
    };
    let impl_id = if let Some(ParentStackItem::Impl { item_id, .. }) = cache.parent_stack.last() {
        item_id.as_def_id()
    } else {
        None
    };
    let search_type = get_function_type_for_search(
        item,
        tcx,
        clean_impl_generics(cache.parent_stack.last()).as_ref(),
        parent_did,
        cache,
    );
    let aliases = item.attrs.get_doc_aliases();
    let deprecation = item.deprecation(tcx);
    let index_item = IndexItem {
        ty: item.type_(),
        defid: Some(defid),
        name,
        module_path: parent_path.to_vec(),
        desc,
        parent: parent_did,
        parent_idx: None,
        exact_module_path: None,
        impl_id,
        search_type,
        aliases,
        deprecation,
    };
    cache.search_index.push(index_item);
}

/// We have a parent, but we don't know where they're
/// defined yet. Wait for later to index this item.
/// See [`Cache::orphan_impl_items`].
fn handle_orphan_impl_child(cache: &mut Cache, item: &clean::Item, parent_did: DefId) {
    let impl_generics = clean_impl_generics(cache.parent_stack.last());
    let impl_id = if let Some(ParentStackItem::Impl { item_id, .. }) = cache.parent_stack.last() {
        item_id.as_def_id()
    } else {
        None
    };
    let orphan_item =
        OrphanImplItem { parent: parent_did, item: item.clone(), impl_generics, impl_id };
    cache.orphan_impl_items.push(orphan_item);
}

pub(crate) struct OrphanImplItem {
    pub(crate) parent: DefId,
    pub(crate) impl_id: Option<DefId>,
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
        match &item.kind {
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
