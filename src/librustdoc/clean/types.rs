use std::hash::Hash;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock as OnceCell};
use std::{fmt, iter};

use arrayvec::ArrayVec;
use rustc_abi::{ExternAbi, VariantIdx};
use rustc_attr_parsing::{AttributeKind, ConstStability, Deprecation, Stability, StableSince};
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{BodyId, Mutability};
use rustc_index::IndexVec;
use rustc_metadata::rendered_const;
use rustc_middle::span_bug;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{self, TyCtxt, Visibility};
use rustc_resolve::rustdoc::{
    DocFragment, add_doc_fragment, attrs_to_doc_fragments, inner_docs, span_of_fragments,
};
use rustc_session::Session;
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{Ident, Symbol, kw, sym};
use rustc_span::{DUMMY_SP, FileName, Loc};
use thin_vec::ThinVec;
use tracing::{debug, trace};
use {rustc_ast as ast, rustc_hir as hir};

pub(crate) use self::ItemKind::*;
pub(crate) use self::Type::{
    Array, BareFunction, BorrowedRef, DynTrait, Generic, ImplTrait, Infer, Primitive, QPath,
    RawPointer, SelfTy, Slice, Tuple, UnsafeBinder,
};
use crate::clean::cfg::Cfg;
use crate::clean::clean_middle_path;
use crate::clean::inline::{self, print_inlined_const};
use crate::clean::utils::{is_literal_expr, print_evaluated_const};
use crate::core::DocContext;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::render::Context;
use crate::passes::collect_intra_doc_links::UrlFragment;

#[cfg(test)]
mod tests;

pub(crate) type ItemIdSet = FxHashSet<ItemId>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub(crate) enum ItemId {
    /// A "normal" item that uses a [`DefId`] for identification.
    DefId(DefId),
    /// Identifier that is used for auto traits.
    Auto { trait_: DefId, for_: DefId },
    /// Identifier that is used for blanket implementations.
    Blanket { impl_id: DefId, for_: DefId },
}

impl ItemId {
    #[inline]
    pub(crate) fn is_local(self) -> bool {
        match self {
            ItemId::Auto { for_: id, .. }
            | ItemId::Blanket { for_: id, .. }
            | ItemId::DefId(id) => id.is_local(),
        }
    }

    #[inline]
    #[track_caller]
    pub(crate) fn expect_def_id(self) -> DefId {
        self.as_def_id()
            .unwrap_or_else(|| panic!("ItemId::expect_def_id: `{self:?}` isn't a DefId"))
    }

    #[inline]
    pub(crate) fn as_def_id(self) -> Option<DefId> {
        match self {
            ItemId::DefId(id) => Some(id),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn as_local_def_id(self) -> Option<LocalDefId> {
        self.as_def_id().and_then(|id| id.as_local())
    }

    #[inline]
    pub(crate) fn krate(self) -> CrateNum {
        match self {
            ItemId::Auto { for_: id, .. }
            | ItemId::Blanket { for_: id, .. }
            | ItemId::DefId(id) => id.krate,
        }
    }
}

impl From<DefId> for ItemId {
    fn from(id: DefId) -> Self {
        Self::DefId(id)
    }
}

/// The crate currently being documented.
#[derive(Clone, Debug)]
pub(crate) struct Crate {
    pub(crate) module: Item,
    /// Only here so that they can be filtered through the rustdoc passes.
    pub(crate) external_traits: Box<FxIndexMap<DefId, Trait>>,
}

impl Crate {
    pub(crate) fn name(&self, tcx: TyCtxt<'_>) -> Symbol {
        ExternalCrate::LOCAL.name(tcx)
    }

    pub(crate) fn src(&self, tcx: TyCtxt<'_>) -> FileName {
        ExternalCrate::LOCAL.src(tcx)
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct ExternalCrate {
    pub(crate) crate_num: CrateNum,
}

impl ExternalCrate {
    const LOCAL: Self = Self { crate_num: LOCAL_CRATE };

    #[inline]
    pub(crate) fn def_id(&self) -> DefId {
        self.crate_num.as_def_id()
    }

    pub(crate) fn src(&self, tcx: TyCtxt<'_>) -> FileName {
        let krate_span = tcx.def_span(self.def_id());
        tcx.sess.source_map().span_to_filename(krate_span)
    }

    pub(crate) fn name(&self, tcx: TyCtxt<'_>) -> Symbol {
        tcx.crate_name(self.crate_num)
    }

    pub(crate) fn src_root(&self, tcx: TyCtxt<'_>) -> PathBuf {
        match self.src(tcx) {
            FileName::Real(ref p) => match p.local_path_if_available().parent() {
                Some(p) => p.to_path_buf(),
                None => PathBuf::new(),
            },
            _ => PathBuf::new(),
        }
    }

    /// Attempts to find where an external crate is located, given that we're
    /// rendering into the specified source destination.
    pub(crate) fn location(
        &self,
        extern_url: Option<&str>,
        extern_url_takes_precedence: bool,
        dst: &std::path::Path,
        tcx: TyCtxt<'_>,
    ) -> ExternalLocation {
        use ExternalLocation::*;

        fn to_remote(url: impl ToString) -> ExternalLocation {
            let mut url = url.to_string();
            if !url.ends_with('/') {
                url.push('/');
            }
            Remote(url)
        }

        // See if there's documentation generated into the local directory
        // WARNING: since rustdoc creates these directories as it generates documentation, this check is only accurate before rendering starts.
        // Make sure to call `location()` by that time.
        let local_location = dst.join(self.name(tcx).as_str());
        if local_location.is_dir() {
            return Local;
        }

        if extern_url_takes_precedence && let Some(url) = extern_url {
            return to_remote(url);
        }

        // Failing that, see if there's an attribute specifying where to find this
        // external crate
        let did = self.crate_num.as_def_id();
        tcx.get_attrs(did, sym::doc)
            .flat_map(|attr| attr.meta_item_list().unwrap_or_default())
            .filter(|a| a.has_name(sym::html_root_url))
            .filter_map(|a| a.value_str())
            .map(to_remote)
            .next()
            .or_else(|| extern_url.map(to_remote)) // NOTE: only matters if `extern_url_takes_precedence` is false
            .unwrap_or(Unknown) // Well, at least we tried.
    }

    pub(crate) fn keywords(&self, tcx: TyCtxt<'_>) -> ThinVec<(DefId, Symbol)> {
        let root = self.def_id();

        let as_keyword = |res: Res<!>| {
            if let Res::Def(DefKind::Mod, def_id) = res {
                let mut keyword = None;
                let meta_items = tcx
                    .get_attrs(def_id, sym::doc)
                    .flat_map(|attr| attr.meta_item_list().unwrap_or_default());
                for meta in meta_items {
                    if meta.has_name(sym::keyword)
                        && let Some(v) = meta.value_str()
                    {
                        keyword = Some(v);
                        break;
                    }
                }
                return keyword.map(|p| (def_id, p));
            }
            None
        };
        if root.is_local() {
            tcx.hir_root_module()
                .item_ids
                .iter()
                .filter_map(|&id| {
                    let item = tcx.hir_item(id);
                    match item.kind {
                        hir::ItemKind::Mod(..) => {
                            as_keyword(Res::Def(DefKind::Mod, id.owner_id.to_def_id()))
                        }
                        _ => None,
                    }
                })
                .collect()
        } else {
            tcx.module_children(root).iter().map(|item| item.res).filter_map(as_keyword).collect()
        }
    }

    pub(crate) fn primitives(&self, tcx: TyCtxt<'_>) -> ThinVec<(DefId, PrimitiveType)> {
        let root = self.def_id();

        // Collect all inner modules which are tagged as implementations of
        // primitives.
        //
        // Note that this loop only searches the top-level items of the crate,
        // and this is intentional. If we were to search the entire crate for an
        // item tagged with `#[rustc_doc_primitive]` then we would also have to
        // search the entirety of external modules for items tagged
        // `#[rustc_doc_primitive]`, which is a pretty inefficient process (decoding
        // all that metadata unconditionally).
        //
        // In order to keep the metadata load under control, the
        // `#[rustc_doc_primitive]` feature is explicitly designed to only allow the
        // primitive tags to show up as the top level items in a crate.
        //
        // Also note that this does not attempt to deal with modules tagged
        // duplicately for the same primitive. This is handled later on when
        // rendering by delegating everything to a hash map.
        let as_primitive = |res: Res<!>| {
            let Res::Def(DefKind::Mod, def_id) = res else { return None };
            tcx.get_attrs(def_id, sym::rustc_doc_primitive)
                .map(|attr| {
                    let attr_value = attr.value_str().expect("syntax should already be validated");
                    let Some(prim) = PrimitiveType::from_symbol(attr_value) else {
                        span_bug!(
                            attr.span(),
                            "primitive `{attr_value}` is not a member of `PrimitiveType`"
                        );
                    };

                    (def_id, prim)
                })
                .next()
        };

        if root.is_local() {
            tcx.hir_root_module()
                .item_ids
                .iter()
                .filter_map(|&id| {
                    let item = tcx.hir_item(id);
                    match item.kind {
                        hir::ItemKind::Mod(..) => {
                            as_primitive(Res::Def(DefKind::Mod, id.owner_id.to_def_id()))
                        }
                        _ => None,
                    }
                })
                .collect()
        } else {
            tcx.module_children(root).iter().map(|item| item.res).filter_map(as_primitive).collect()
        }
    }
}

/// Indicates where an external crate can be found.
#[derive(Debug)]
pub(crate) enum ExternalLocation {
    /// Remote URL root of the external crate
    Remote(String),
    /// This external crate can be found in the local doc/ folder
    Local,
    /// The external crate could not be found.
    Unknown,
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[derive(Clone)]
pub(crate) struct Item {
    pub(crate) inner: Box<ItemInner>,
}

// Why does the `Item`/`ItemInner` split exist? `Vec<Item>`s are common, and
// without the split `Item` would be a large type (100+ bytes) which results in
// lots of wasted space in the unused parts of a `Vec<Item>`. With the split,
// `Item` is just 8 bytes, and the wasted space is avoided, at the cost of an
// extra allocation per item. This is a performance win.
#[derive(Clone)]
pub(crate) struct ItemInner {
    /// The name of this item.
    /// Optional because not every item has a name, e.g. impls.
    pub(crate) name: Option<Symbol>,
    /// Information about this item that is specific to what kind of item it is.
    /// E.g., struct vs enum vs function.
    pub(crate) kind: ItemKind,
    pub(crate) attrs: Attributes,
    /// The effective stability, filled out by the `propagate-stability` pass.
    pub(crate) stability: Option<Stability>,
    pub(crate) item_id: ItemId,
    /// This is the `LocalDefId` of the `use` statement if the item was inlined.
    /// The crate metadata doesn't hold this information, so the `use` statement
    /// always belongs to the current crate.
    pub(crate) inline_stmt_id: Option<LocalDefId>,
    pub(crate) cfg: Option<Arc<Cfg>>,
}

impl std::ops::Deref for Item {
    type Target = ItemInner;
    fn deref(&self) -> &ItemInner {
        &self.inner
    }
}

/// NOTE: this does NOT unconditionally print every item, to avoid thousands of lines of logs.
/// If you want to see the debug output for attributes and the `kind` as well, use `{:#?}` instead of `{:?}`.
impl fmt::Debug for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let alternate = f.alternate();
        // hand-picked fields that don't bloat the logs too much
        let mut fmt = f.debug_struct("Item");
        fmt.field("name", &self.name).field("item_id", &self.item_id);
        // allow printing the full item if someone really wants to
        if alternate {
            fmt.field("attrs", &self.attrs).field("kind", &self.kind).field("cfg", &self.cfg);
        } else {
            fmt.field("kind", &self.type_());
            fmt.field("docs", &self.doc_value());
        }
        fmt.finish()
    }
}

pub(crate) fn rustc_span(def_id: DefId, tcx: TyCtxt<'_>) -> Span {
    Span::new(def_id.as_local().map_or_else(
        || tcx.def_span(def_id),
        |local| tcx.hir_span_with_body(tcx.local_def_id_to_hir_id(local)),
    ))
}

fn is_field_vis_inherited(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let parent = tcx.parent(def_id);
    match tcx.def_kind(parent) {
        DefKind::Struct | DefKind::Union => false,
        DefKind::Variant => true,
        parent_kind => panic!("unexpected parent kind: {parent_kind:?}"),
    }
}

impl Item {
    /// Returns the effective stability of the item.
    ///
    /// This method should only be called after the `propagate-stability` pass has been run.
    pub(crate) fn stability(&self, tcx: TyCtxt<'_>) -> Option<Stability> {
        let stability = self.inner.stability;
        debug_assert!(
            stability.is_some()
                || self.def_id().is_none_or(|did| tcx.lookup_stability(did).is_none()),
            "missing stability for cleaned item: {self:?}",
        );
        stability
    }

    pub(crate) fn const_stability(&self, tcx: TyCtxt<'_>) -> Option<ConstStability> {
        self.def_id().and_then(|did| tcx.lookup_const_stability(did))
    }

    pub(crate) fn deprecation(&self, tcx: TyCtxt<'_>) -> Option<Deprecation> {
        self.def_id().and_then(|did| tcx.lookup_deprecation(did)).or_else(|| {
            // `allowed_through_unstable_modules` is a bug-compatibility hack for old rustc
            // versions; the paths that are exposed through it are "deprecated" because they
            // were never supposed to work at all.
            let stab = self.stability(tcx)?;
            if let rustc_attr_parsing::StabilityLevel::Stable {
                allowed_through_unstable_modules: Some(note),
                ..
            } = stab.level
            {
                Some(Deprecation {
                    since: rustc_attr_parsing::DeprecatedSince::Unspecified,
                    note: Some(note),
                    suggestion: None,
                })
            } else {
                None
            }
        })
    }

    pub(crate) fn inner_docs(&self, tcx: TyCtxt<'_>) -> bool {
        self.item_id
            .as_def_id()
            .map(|did| inner_docs(tcx.get_attrs_unchecked(did)))
            .unwrap_or(false)
    }

    pub(crate) fn span(&self, tcx: TyCtxt<'_>) -> Option<Span> {
        let kind = match &self.kind {
            ItemKind::StrippedItem(k) => k,
            _ => &self.kind,
        };
        match kind {
            ItemKind::ModuleItem(Module { span, .. }) => Some(*span),
            ItemKind::ImplItem(box Impl { kind: ImplKind::Auto, .. }) => None,
            ItemKind::ImplItem(box Impl { kind: ImplKind::Blanket(_), .. }) => {
                if let ItemId::Blanket { impl_id, .. } = self.item_id {
                    Some(rustc_span(impl_id, tcx))
                } else {
                    panic!("blanket impl item has non-blanket ID")
                }
            }
            _ => self.def_id().map(|did| rustc_span(did, tcx)),
        }
    }

    pub(crate) fn attr_span(&self, tcx: TyCtxt<'_>) -> rustc_span::Span {
        span_of_fragments(&self.attrs.doc_strings)
            .unwrap_or_else(|| self.span(tcx).map_or(DUMMY_SP, |span| span.inner()))
    }

    /// Combine all doc strings into a single value handling indentation and newlines as needed.
    pub(crate) fn doc_value(&self) -> String {
        self.attrs.doc_value()
    }

    /// Combine all doc strings into a single value handling indentation and newlines as needed.
    /// Returns `None` is there's no documentation at all, and `Some("")` if there is some
    /// documentation but it is empty (e.g. `#[doc = ""]`).
    pub(crate) fn opt_doc_value(&self) -> Option<String> {
        self.attrs.opt_doc_value()
    }

    pub(crate) fn from_def_id_and_parts(
        def_id: DefId,
        name: Option<Symbol>,
        kind: ItemKind,
        cx: &mut DocContext<'_>,
    ) -> Item {
        let hir_attrs = cx.tcx.get_attrs_unchecked(def_id);

        Self::from_def_id_and_attrs_and_parts(
            def_id,
            name,
            kind,
            Attributes::from_hir(hir_attrs),
            extract_cfg_from_attrs(hir_attrs.iter(), cx.tcx, &cx.cache.hidden_cfg),
        )
    }

    pub(crate) fn from_def_id_and_attrs_and_parts(
        def_id: DefId,
        name: Option<Symbol>,
        kind: ItemKind,
        attrs: Attributes,
        cfg: Option<Arc<Cfg>>,
    ) -> Item {
        trace!("name={name:?}, def_id={def_id:?} cfg={cfg:?}");

        Item {
            inner: Box::new(ItemInner {
                item_id: def_id.into(),
                kind,
                attrs,
                stability: None,
                name,
                cfg,
                inline_stmt_id: None,
            }),
        }
    }

    pub(crate) fn links(&self, cx: &Context<'_>) -> Vec<RenderedLink> {
        use crate::html::format::{href, link_tooltip};

        let Some(links) = cx.cache().intra_doc_links.get(&self.item_id) else { return vec![] };
        links
            .iter()
            .filter_map(|ItemLink { link: s, link_text, page_id: id, fragment }| {
                debug!(?id);
                if let Ok((mut href, ..)) = href(*id, cx) {
                    debug!(?href);
                    if let Some(ref fragment) = *fragment {
                        fragment.render(&mut href, cx.tcx())
                    }
                    Some(RenderedLink {
                        original_text: s.clone(),
                        new_text: link_text.clone(),
                        tooltip: link_tooltip(*id, fragment, cx).to_string(),
                        href,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find a list of all link names, without finding their href.
    ///
    /// This is used for generating summary text, which does not include
    /// the link text, but does need to know which `[]`-bracketed names
    /// are actually links.
    pub(crate) fn link_names(&self, cache: &Cache) -> Vec<RenderedLink> {
        let Some(links) = cache.intra_doc_links.get(&self.item_id) else {
            return vec![];
        };
        links
            .iter()
            .map(|ItemLink { link: s, link_text, .. }| RenderedLink {
                original_text: s.clone(),
                new_text: link_text.clone(),
                href: String::new(),
                tooltip: String::new(),
            })
            .collect()
    }

    pub(crate) fn is_crate(&self) -> bool {
        self.is_mod() && self.def_id().is_some_and(|did| did.is_crate_root())
    }
    pub(crate) fn is_mod(&self) -> bool {
        self.type_() == ItemType::Module
    }
    pub(crate) fn is_struct(&self) -> bool {
        self.type_() == ItemType::Struct
    }
    pub(crate) fn is_enum(&self) -> bool {
        self.type_() == ItemType::Enum
    }
    pub(crate) fn is_variant(&self) -> bool {
        self.type_() == ItemType::Variant
    }
    pub(crate) fn is_associated_type(&self) -> bool {
        matches!(self.kind, AssocTypeItem(..) | StrippedItem(box AssocTypeItem(..)))
    }
    pub(crate) fn is_required_associated_type(&self) -> bool {
        matches!(self.kind, RequiredAssocTypeItem(..) | StrippedItem(box RequiredAssocTypeItem(..)))
    }
    pub(crate) fn is_associated_const(&self) -> bool {
        matches!(self.kind, ProvidedAssocConstItem(..) | ImplAssocConstItem(..) | StrippedItem(box (ProvidedAssocConstItem(..) | ImplAssocConstItem(..))))
    }
    pub(crate) fn is_required_associated_const(&self) -> bool {
        matches!(self.kind, RequiredAssocConstItem(..) | StrippedItem(box RequiredAssocConstItem(..)))
    }
    pub(crate) fn is_method(&self) -> bool {
        self.type_() == ItemType::Method
    }
    pub(crate) fn is_ty_method(&self) -> bool {
        self.type_() == ItemType::TyMethod
    }
    pub(crate) fn is_primitive(&self) -> bool {
        self.type_() == ItemType::Primitive
    }
    pub(crate) fn is_union(&self) -> bool {
        self.type_() == ItemType::Union
    }
    pub(crate) fn is_import(&self) -> bool {
        self.type_() == ItemType::Import
    }
    pub(crate) fn is_extern_crate(&self) -> bool {
        self.type_() == ItemType::ExternCrate
    }
    pub(crate) fn is_keyword(&self) -> bool {
        self.type_() == ItemType::Keyword
    }
    pub(crate) fn is_stripped(&self) -> bool {
        match self.kind {
            StrippedItem(..) => true,
            ImportItem(ref i) => !i.should_be_displayed,
            _ => false,
        }
    }
    pub(crate) fn has_stripped_entries(&self) -> Option<bool> {
        match self.kind {
            StructItem(ref struct_) => Some(struct_.has_stripped_entries()),
            UnionItem(ref union_) => Some(union_.has_stripped_entries()),
            EnumItem(ref enum_) => Some(enum_.has_stripped_entries()),
            VariantItem(ref v) => v.has_stripped_entries(),
            _ => None,
        }
    }

    pub(crate) fn stability_class(&self, tcx: TyCtxt<'_>) -> Option<String> {
        self.stability(tcx).as_ref().and_then(|s| {
            let mut classes = Vec::with_capacity(2);

            if s.is_unstable() {
                classes.push("unstable");
            }

            // FIXME: what about non-staged API items that are deprecated?
            if self.deprecation(tcx).is_some() {
                classes.push("deprecated");
            }

            if !classes.is_empty() { Some(classes.join(" ")) } else { None }
        })
    }

    pub(crate) fn stable_since(&self, tcx: TyCtxt<'_>) -> Option<StableSince> {
        self.stability(tcx).and_then(|stability| stability.stable_since())
    }

    pub(crate) fn is_non_exhaustive(&self) -> bool {
        self.attrs.other_attrs.iter().any(|a| a.has_name(sym::non_exhaustive))
    }

    /// Returns a documentation-level item type from the item.
    pub(crate) fn type_(&self) -> ItemType {
        ItemType::from(self)
    }

    pub(crate) fn is_default(&self) -> bool {
        match self.kind {
            ItemKind::MethodItem(_, Some(defaultness)) => {
                defaultness.has_value() && !defaultness.is_final()
            }
            _ => false,
        }
    }

    /// Returns a `FnHeader` if `self` is a function item, otherwise returns `None`.
    pub(crate) fn fn_header(&self, tcx: TyCtxt<'_>) -> Option<hir::FnHeader> {
        fn build_fn_header(
            def_id: DefId,
            tcx: TyCtxt<'_>,
            asyncness: ty::Asyncness,
        ) -> hir::FnHeader {
            let sig = tcx.fn_sig(def_id).skip_binder();
            let constness = if tcx.is_const_fn(def_id) {
                hir::Constness::Const
            } else {
                hir::Constness::NotConst
            };
            let asyncness = match asyncness {
                ty::Asyncness::Yes => hir::IsAsync::Async(DUMMY_SP),
                ty::Asyncness::No => hir::IsAsync::NotAsync,
            };
            hir::FnHeader {
                safety: if tcx.codegen_fn_attrs(def_id).safe_target_features {
                    hir::HeaderSafety::SafeTargetFeatures
                } else {
                    sig.safety().into()
                },
                abi: sig.abi(),
                constness,
                asyncness,
            }
        }
        let header = match self.kind {
            ItemKind::ForeignFunctionItem(_, safety) => {
                let def_id = self.def_id().unwrap();
                let abi = tcx.fn_sig(def_id).skip_binder().abi();
                hir::FnHeader {
                    safety: if tcx.codegen_fn_attrs(def_id).safe_target_features {
                        hir::HeaderSafety::SafeTargetFeatures
                    } else {
                        safety.into()
                    },
                    abi,
                    constness: if tcx.is_const_fn(def_id) {
                        hir::Constness::Const
                    } else {
                        hir::Constness::NotConst
                    },
                    asyncness: hir::IsAsync::NotAsync,
                }
            }
            ItemKind::FunctionItem(_)
            | ItemKind::MethodItem(_, _)
            | ItemKind::RequiredMethodItem(_) => {
                let def_id = self.def_id().unwrap();
                build_fn_header(def_id, tcx, tcx.asyncness(def_id))
            }
            _ => return None,
        };
        Some(header)
    }

    /// Returns the visibility of the current item. If the visibility is "inherited", then `None`
    /// is returned.
    pub(crate) fn visibility(&self, tcx: TyCtxt<'_>) -> Option<Visibility<DefId>> {
        let def_id = match self.item_id {
            // Anything but DefId *shouldn't* matter, but return a reasonable value anyway.
            ItemId::Auto { .. } | ItemId::Blanket { .. } => return None,
            ItemId::DefId(def_id) => def_id,
        };

        match self.kind {
            // Primitives and Keywords are written in the source code as private modules.
            // The modules need to be private so that nobody actually uses them, but the
            // keywords and primitives that they are documenting are public.
            ItemKind::KeywordItem | ItemKind::PrimitiveItem(_) => return Some(Visibility::Public),
            // Variant fields inherit their enum's visibility.
            StructFieldItem(..) if is_field_vis_inherited(tcx, def_id) => {
                return None;
            }
            // Variants always inherit visibility
            VariantItem(..) | ImplItem(..) => return None,
            // Trait items inherit the trait's visibility
            RequiredAssocConstItem(..)
            | ProvidedAssocConstItem(..)
            | ImplAssocConstItem(..)
            | AssocTypeItem(..)
            | RequiredAssocTypeItem(..)
            | RequiredMethodItem(..)
            | MethodItem(..) => {
                let assoc_item = tcx.associated_item(def_id);
                let is_trait_item = match assoc_item.container {
                    ty::AssocItemContainer::Trait => true,
                    ty::AssocItemContainer::Impl => {
                        // Trait impl items always inherit the impl's visibility --
                        // we don't want to show `pub`.
                        tcx.impl_trait_ref(tcx.parent(assoc_item.def_id)).is_some()
                    }
                };
                if is_trait_item {
                    return None;
                }
            }
            _ => {}
        }
        let def_id = match self.inline_stmt_id {
            Some(inlined) => inlined.to_def_id(),
            None => def_id,
        };
        Some(tcx.visibility(def_id))
    }

    pub(crate) fn attributes(&self, tcx: TyCtxt<'_>, cache: &Cache, is_json: bool) -> Vec<String> {
        const ALLOWED_ATTRIBUTES: &[Symbol] =
            &[sym::export_name, sym::link_section, sym::no_mangle, sym::non_exhaustive];

        use rustc_abi::IntegerType;

        let mut attrs: Vec<String> = self
            .attrs
            .other_attrs
            .iter()
            .filter_map(|attr| {
                if is_json {
                    match attr {
                        hir::Attribute::Parsed(AttributeKind::Deprecation { .. }) => {
                            // rustdoc-json stores this in `Item::deprecation`, so we
                            // don't want it it `Item::attrs`.
                            None
                        }
                        rustc_hir::Attribute::Parsed(rustc_attr_parsing::AttributeKind::Repr(
                            ..,
                        )) => {
                            // We have separate pretty-printing logic for `#[repr(..)]` attributes.
                            // For example, there are circumstances where `#[repr(transparent)]`
                            // is applied but should not be publicly shown in rustdoc
                            // because it isn't public API.
                            None
                        }
                        _ => Some(rustc_hir_pretty::attribute_to_string(&tcx, attr)),
                    }
                } else if attr.has_any_name(ALLOWED_ATTRIBUTES) {
                    Some(
                        rustc_hir_pretty::attribute_to_string(&tcx, attr)
                            .replace("\\\n", "")
                            .replace('\n', "")
                            .replace("  ", " "),
                    )
                } else {
                    None
                }
            })
            .collect();

        // Add #[repr(...)]
        if let Some(def_id) = self.def_id()
            && let ItemType::Struct | ItemType::Enum | ItemType::Union = self.type_()
        {
            let adt = tcx.adt_def(def_id);
            let repr = adt.repr();
            let mut out = Vec::new();
            if repr.c() {
                out.push("C");
            }
            if repr.transparent() {
                // Render `repr(transparent)` iff the non-1-ZST field is public or at least one
                // field is public in case all fields are 1-ZST fields.
                let render_transparent = cache.document_private
                    || adt
                        .all_fields()
                        .find(|field| {
                            let ty =
                                field.ty(tcx, ty::GenericArgs::identity_for_item(tcx, field.did));
                            tcx.layout_of(
                                ty::TypingEnv::post_analysis(tcx, field.did).as_query_input(ty),
                            )
                            .is_ok_and(|layout| !layout.is_1zst())
                        })
                        .map_or_else(
                            || adt.all_fields().any(|field| field.vis.is_public()),
                            |field| field.vis.is_public(),
                        );

                if render_transparent {
                    out.push("transparent");
                }
            }
            if repr.simd() {
                out.push("simd");
            }
            let pack_s;
            if let Some(pack) = repr.pack {
                pack_s = format!("packed({})", pack.bytes());
                out.push(&pack_s);
            }
            let align_s;
            if let Some(align) = repr.align {
                align_s = format!("align({})", align.bytes());
                out.push(&align_s);
            }
            let int_s;
            if let Some(int) = repr.int {
                int_s = match int {
                    IntegerType::Pointer(is_signed) => {
                        format!("{}size", if is_signed { 'i' } else { 'u' })
                    }
                    IntegerType::Fixed(size, is_signed) => {
                        format!("{}{}", if is_signed { 'i' } else { 'u' }, size.size().bytes() * 8)
                    }
                };
                out.push(&int_s);
            }
            if !out.is_empty() {
                attrs.push(format!("#[repr({})]", out.join(", ")));
            }
        }
        attrs
    }

    pub fn is_doc_hidden(&self) -> bool {
        self.attrs.is_doc_hidden()
    }

    pub fn def_id(&self) -> Option<DefId> {
        self.item_id.as_def_id()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ItemKind {
    ExternCrateItem {
        /// The crate's name, *not* the name it's imported as.
        src: Option<Symbol>,
    },
    ImportItem(Import),
    StructItem(Struct),
    UnionItem(Union),
    EnumItem(Enum),
    FunctionItem(Box<Function>),
    ModuleItem(Module),
    TypeAliasItem(Box<TypeAlias>),
    StaticItem(Static),
    TraitItem(Box<Trait>),
    TraitAliasItem(TraitAlias),
    ImplItem(Box<Impl>),
    /// A required method in a trait declaration meaning it's only a function signature.
    RequiredMethodItem(Box<Function>),
    /// A method in a trait impl or a provided method in a trait declaration.
    ///
    /// Compared to [RequiredMethodItem], it also contains a method body.
    MethodItem(Box<Function>, Option<hir::Defaultness>),
    StructFieldItem(Type),
    VariantItem(Variant),
    /// `fn`s from an extern block
    ForeignFunctionItem(Box<Function>, hir::Safety),
    /// `static`s from an extern block
    ForeignStaticItem(Static, hir::Safety),
    /// `type`s from an extern block
    ForeignTypeItem,
    MacroItem(Macro),
    ProcMacroItem(ProcMacro),
    PrimitiveItem(PrimitiveType),
    /// A required associated constant in a trait declaration.
    RequiredAssocConstItem(Generics, Box<Type>),
    ConstantItem(Box<Constant>),
    /// An associated constant in a trait declaration with provided default value.
    ProvidedAssocConstItem(Box<Constant>),
    /// An associated constant in an inherent impl or trait impl.
    ImplAssocConstItem(Box<Constant>),
    /// A required associated type in a trait declaration.
    ///
    /// The bounds may be non-empty if there is a `where` clause.
    RequiredAssocTypeItem(Generics, Vec<GenericBound>),
    /// An associated type in a trait impl or a provided one in a trait declaration.
    AssocTypeItem(Box<TypeAlias>, Vec<GenericBound>),
    /// An item that has been stripped by a rustdoc pass
    StrippedItem(Box<ItemKind>),
    KeywordItem,
}

impl ItemKind {
    /// Some items contain others such as structs (for their fields) and Enums
    /// (for their variants). This method returns those contained items.
    pub(crate) fn inner_items(&self) -> impl Iterator<Item = &Item> {
        match self {
            StructItem(s) => s.fields.iter(),
            UnionItem(u) => u.fields.iter(),
            VariantItem(v) => match &v.kind {
                VariantKind::CLike => [].iter(),
                VariantKind::Tuple(t) => t.iter(),
                VariantKind::Struct(s) => s.fields.iter(),
            },
            EnumItem(e) => e.variants.iter(),
            TraitItem(t) => t.items.iter(),
            ImplItem(i) => i.items.iter(),
            ModuleItem(m) => m.items.iter(),
            ExternCrateItem { .. }
            | ImportItem(_)
            | FunctionItem(_)
            | TypeAliasItem(_)
            | StaticItem(_)
            | ConstantItem(_)
            | TraitAliasItem(_)
            | RequiredMethodItem(_)
            | MethodItem(_, _)
            | StructFieldItem(_)
            | ForeignFunctionItem(_, _)
            | ForeignStaticItem(_, _)
            | ForeignTypeItem
            | MacroItem(_)
            | ProcMacroItem(_)
            | PrimitiveItem(_)
            | RequiredAssocConstItem(..)
            | ProvidedAssocConstItem(..)
            | ImplAssocConstItem(..)
            | RequiredAssocTypeItem(..)
            | AssocTypeItem(..)
            | StrippedItem(_)
            | KeywordItem => [].iter(),
        }
    }

    /// Returns `true` if this item does not appear inside an impl block.
    pub(crate) fn is_non_assoc(&self) -> bool {
        matches!(
            self,
            StructItem(_)
                | UnionItem(_)
                | EnumItem(_)
                | TraitItem(_)
                | ModuleItem(_)
                | ExternCrateItem { .. }
                | FunctionItem(_)
                | TypeAliasItem(_)
                | StaticItem(_)
                | ConstantItem(_)
                | TraitAliasItem(_)
                | ForeignFunctionItem(_, _)
                | ForeignStaticItem(_, _)
                | ForeignTypeItem
                | MacroItem(_)
                | ProcMacroItem(_)
                | PrimitiveItem(_)
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Module {
    pub(crate) items: Vec<Item>,
    pub(crate) span: Span,
}

pub(crate) fn hir_attr_lists<'a, I: IntoIterator<Item = &'a hir::Attribute>>(
    attrs: I,
    name: Symbol,
) -> impl Iterator<Item = ast::MetaItemInner> + use<'a, I> {
    attrs
        .into_iter()
        .filter(move |attr| attr.has_name(name))
        .filter_map(ast::attr::AttributeExt::meta_item_list)
        .flatten()
}

pub(crate) fn extract_cfg_from_attrs<'a, I: Iterator<Item = &'a hir::Attribute> + Clone>(
    attrs: I,
    tcx: TyCtxt<'_>,
    hidden_cfg: &FxHashSet<Cfg>,
) -> Option<Arc<Cfg>> {
    let doc_cfg_active = tcx.features().doc_cfg();
    let doc_auto_cfg_active = tcx.features().doc_auto_cfg();

    fn single<T: IntoIterator>(it: T) -> Option<T::Item> {
        let mut iter = it.into_iter();
        let item = iter.next()?;
        if iter.next().is_some() {
            return None;
        }
        Some(item)
    }

    let mut cfg = if doc_cfg_active || doc_auto_cfg_active {
        let mut doc_cfg = attrs
            .clone()
            .filter(|attr| attr.has_name(sym::doc))
            .flat_map(|attr| attr.meta_item_list().unwrap_or_default())
            .filter(|attr| attr.has_name(sym::cfg))
            .peekable();
        if doc_cfg.peek().is_some() && doc_cfg_active {
            let sess = tcx.sess;
            doc_cfg.fold(Cfg::True, |mut cfg, item| {
                if let Some(cfg_mi) =
                    item.meta_item().and_then(|item| rustc_expand::config::parse_cfg(item, sess))
                {
                    // The result is unused here but we can gate unstable predicates
                    rustc_attr_parsing::cfg_matches(
                        cfg_mi,
                        tcx.sess,
                        rustc_ast::CRATE_NODE_ID,
                        Some(tcx.features()),
                    );
                    match Cfg::parse(cfg_mi) {
                        Ok(new_cfg) => cfg &= new_cfg,
                        Err(e) => {
                            sess.dcx().span_err(e.span, e.msg);
                        }
                    }
                }
                cfg
            })
        } else if doc_auto_cfg_active {
            // If there is no `doc(cfg())`, then we retrieve the `cfg()` attributes (because
            // `doc(cfg())` overrides `cfg()`).
            attrs
                .clone()
                .filter(|attr| attr.has_name(sym::cfg_trace))
                .filter_map(|attr| single(attr.meta_item_list()?))
                .filter_map(|attr| Cfg::parse_without(attr.meta_item()?, hidden_cfg).ok().flatten())
                .fold(Cfg::True, |cfg, new_cfg| cfg & new_cfg)
        } else {
            Cfg::True
        }
    } else {
        Cfg::True
    };

    // treat #[target_feature(enable = "feat")] attributes as if they were
    // #[doc(cfg(target_feature = "feat"))] attributes as well
    for attr in hir_attr_lists(attrs, sym::target_feature) {
        if attr.has_name(sym::enable) && attr.value_str().is_some() {
            // Clone `enable = "feat"`, change to `target_feature = "feat"`.
            // Unwrap is safe because `value_str` succeeded above.
            let mut meta = attr.meta_item().unwrap().clone();
            meta.path = ast::Path::from_ident(Ident::with_dummy_span(sym::target_feature));

            if let Ok(feat_cfg) = Cfg::parse(&ast::MetaItemInner::MetaItem(meta)) {
                cfg &= feat_cfg;
            }
        }
    }

    if cfg == Cfg::True { None } else { Some(Arc::new(cfg)) }
}

pub(crate) trait NestedAttributesExt {
    /// Returns `true` if the attribute list contains a specific `word`
    fn has_word(self, word: Symbol) -> bool
    where
        Self: Sized,
    {
        <Self as NestedAttributesExt>::get_word_attr(self, word).is_some()
    }

    /// Returns `Some(attr)` if the attribute list contains 'attr'
    /// corresponding to a specific `word`
    fn get_word_attr(self, word: Symbol) -> Option<ast::MetaItemInner>;
}

impl<I: Iterator<Item = ast::MetaItemInner>> NestedAttributesExt for I {
    fn get_word_attr(mut self, word: Symbol) -> Option<ast::MetaItemInner> {
        self.find(|attr| attr.is_word() && attr.has_name(word))
    }
}

/// A link that has not yet been rendered.
///
/// This link will be turned into a rendered link by [`Item::links`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ItemLink {
    /// The original link written in the markdown
    pub(crate) link: Box<str>,
    /// The link text displayed in the HTML.
    ///
    /// This may not be the same as `link` if there was a disambiguator
    /// in an intra-doc link (e.g. \[`fn@f`\])
    pub(crate) link_text: Box<str>,
    /// The `DefId` of the Item whose **HTML Page** contains the item being
    /// linked to. This will be different to `item_id` on item's that don't
    /// have their own page, such as struct fields and enum variants.
    pub(crate) page_id: DefId,
    /// The url fragment to append to the link
    pub(crate) fragment: Option<UrlFragment>,
}

pub struct RenderedLink {
    /// The text the link was original written as.
    ///
    /// This could potentially include disambiguators and backticks.
    pub(crate) original_text: Box<str>,
    /// The text to display in the HTML
    pub(crate) new_text: Box<str>,
    /// The URL to put in the `href`
    pub(crate) href: String,
    /// The tooltip.
    pub(crate) tooltip: String,
}

/// The attributes on an [`Item`], including attributes like `#[derive(...)]` and `#[inline]`,
/// as well as doc comments.
#[derive(Clone, Debug, Default)]
pub(crate) struct Attributes {
    pub(crate) doc_strings: Vec<DocFragment>,
    pub(crate) other_attrs: ThinVec<hir::Attribute>,
}

impl Attributes {
    pub(crate) fn lists(&self, name: Symbol) -> impl Iterator<Item = ast::MetaItemInner> {
        hir_attr_lists(&self.other_attrs[..], name)
    }

    pub(crate) fn has_doc_flag(&self, flag: Symbol) -> bool {
        for attr in &self.other_attrs {
            if !attr.has_name(sym::doc) {
                continue;
            }

            if let Some(items) = attr.meta_item_list()
                && items.iter().filter_map(|i| i.meta_item()).any(|it| it.has_name(flag))
            {
                return true;
            }
        }

        false
    }

    pub(crate) fn is_doc_hidden(&self) -> bool {
        self.has_doc_flag(sym::hidden)
    }

    pub(crate) fn from_hir(attrs: &[hir::Attribute]) -> Attributes {
        Attributes::from_hir_iter(attrs.iter().map(|attr| (attr, None)), false)
    }

    pub(crate) fn from_hir_with_additional(
        attrs: &[hir::Attribute],
        (additional_attrs, def_id): (&[hir::Attribute], DefId),
    ) -> Attributes {
        // Additional documentation should be shown before the original documentation.
        let attrs1 = additional_attrs.iter().map(|attr| (attr, Some(def_id)));
        let attrs2 = attrs.iter().map(|attr| (attr, None));
        Attributes::from_hir_iter(attrs1.chain(attrs2), false)
    }

    pub(crate) fn from_hir_iter<'a>(
        attrs: impl Iterator<Item = (&'a hir::Attribute, Option<DefId>)>,
        doc_only: bool,
    ) -> Attributes {
        let (doc_strings, other_attrs) = attrs_to_doc_fragments(attrs, doc_only);
        Attributes { doc_strings, other_attrs }
    }

    /// Combine all doc strings into a single value handling indentation and newlines as needed.
    pub(crate) fn doc_value(&self) -> String {
        self.opt_doc_value().unwrap_or_default()
    }

    /// Combine all doc strings into a single value handling indentation and newlines as needed.
    /// Returns `None` is there's no documentation at all, and `Some("")` if there is some
    /// documentation but it is empty (e.g. `#[doc = ""]`).
    pub(crate) fn opt_doc_value(&self) -> Option<String> {
        (!self.doc_strings.is_empty()).then(|| {
            let mut res = String::new();
            for frag in &self.doc_strings {
                add_doc_fragment(&mut res, frag);
            }
            res.pop();
            res
        })
    }

    pub(crate) fn get_doc_aliases(&self) -> Box<[Symbol]> {
        let mut aliases = FxIndexSet::default();

        for attr in
            hir_attr_lists(&self.other_attrs[..], sym::doc).filter(|a| a.has_name(sym::alias))
        {
            if let Some(values) = attr.meta_item_list() {
                for l in values {
                    if let Some(lit) = l.lit()
                        && let ast::LitKind::Str(s, _) = lit.kind
                    {
                        aliases.insert(s);
                    }
                }
            } else if let Some(value) = attr.value_str() {
                aliases.insert(value);
            }
        }
        aliases.into_iter().collect::<Vec<_>>().into()
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum GenericBound {
    TraitBound(PolyTrait, hir::TraitBoundModifiers),
    Outlives(Lifetime),
    /// `use<'a, T>` precise-capturing bound syntax
    Use(Vec<PreciseCapturingArg>),
}

impl GenericBound {
    pub(crate) fn sized(cx: &mut DocContext<'_>) -> GenericBound {
        Self::sized_with(cx, hir::TraitBoundModifiers::NONE)
    }

    pub(crate) fn maybe_sized(cx: &mut DocContext<'_>) -> GenericBound {
        Self::sized_with(
            cx,
            hir::TraitBoundModifiers {
                polarity: hir::BoundPolarity::Maybe(DUMMY_SP),
                constness: hir::BoundConstness::Never,
            },
        )
    }

    fn sized_with(cx: &mut DocContext<'_>, modifiers: hir::TraitBoundModifiers) -> GenericBound {
        let did = cx.tcx.require_lang_item(LangItem::Sized, None);
        let empty = ty::Binder::dummy(ty::GenericArgs::empty());
        let path = clean_middle_path(cx, did, false, ThinVec::new(), empty);
        inline::record_extern_fqn(cx, did, ItemType::Trait);
        GenericBound::TraitBound(PolyTrait { trait_: path, generic_params: Vec::new() }, modifiers)
    }

    pub(crate) fn is_trait_bound(&self) -> bool {
        matches!(self, Self::TraitBound(..))
    }

    pub(crate) fn is_sized_bound(&self, cx: &DocContext<'_>) -> bool {
        if let GenericBound::TraitBound(
            PolyTrait { ref trait_, .. },
            rustc_hir::TraitBoundModifiers::NONE,
        ) = *self
            && Some(trait_.def_id()) == cx.tcx.lang_items().sized_trait()
        {
            return true;
        }
        false
    }

    pub(crate) fn get_trait_path(&self) -> Option<Path> {
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, _) = *self {
            Some(trait_.clone())
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub(crate) struct Lifetime(pub Symbol);

impl Lifetime {
    pub(crate) fn statik() -> Lifetime {
        Lifetime(kw::StaticLifetime)
    }

    pub(crate) fn elided() -> Lifetime {
        Lifetime(kw::UnderscoreLifetime)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub(crate) enum PreciseCapturingArg {
    Lifetime(Lifetime),
    Param(Symbol),
}

impl PreciseCapturingArg {
    pub(crate) fn name(self) -> Symbol {
        match self {
            PreciseCapturingArg::Lifetime(lt) => lt.0,
            PreciseCapturingArg::Param(param) => param,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum WherePredicate {
    BoundPredicate { ty: Type, bounds: Vec<GenericBound>, bound_params: Vec<GenericParamDef> },
    RegionPredicate { lifetime: Lifetime, bounds: Vec<GenericBound> },
    EqPredicate { lhs: Type, rhs: Term },
}

impl WherePredicate {
    pub(crate) fn get_bounds(&self) -> Option<&[GenericBound]> {
        match *self {
            WherePredicate::BoundPredicate { ref bounds, .. } => Some(bounds),
            WherePredicate::RegionPredicate { ref bounds, .. } => Some(bounds),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum GenericParamDefKind {
    Lifetime { outlives: ThinVec<Lifetime> },
    Type { bounds: ThinVec<GenericBound>, default: Option<Box<Type>>, synthetic: bool },
    // Option<Box<String>> makes this type smaller than `Option<String>` would.
    Const { ty: Box<Type>, default: Option<Box<String>>, synthetic: bool },
}

impl GenericParamDefKind {
    pub(crate) fn is_type(&self) -> bool {
        matches!(self, GenericParamDefKind::Type { .. })
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct GenericParamDef {
    pub(crate) name: Symbol,
    pub(crate) def_id: DefId,
    pub(crate) kind: GenericParamDefKind,
}

impl GenericParamDef {
    pub(crate) fn lifetime(def_id: DefId, name: Symbol) -> Self {
        Self { name, def_id, kind: GenericParamDefKind::Lifetime { outlives: ThinVec::new() } }
    }

    pub(crate) fn is_synthetic_param(&self) -> bool {
        match self.kind {
            GenericParamDefKind::Lifetime { .. } | GenericParamDefKind::Const { .. } => false,
            GenericParamDefKind::Type { synthetic, .. } => synthetic,
        }
    }

    pub(crate) fn is_type(&self) -> bool {
        self.kind.is_type()
    }

    pub(crate) fn get_bounds(&self) -> Option<&[GenericBound]> {
        match self.kind {
            GenericParamDefKind::Type { ref bounds, .. } => Some(bounds),
            _ => None,
        }
    }
}

// maybe use a Generic enum and use Vec<Generic>?
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub(crate) struct Generics {
    pub(crate) params: ThinVec<GenericParamDef>,
    pub(crate) where_predicates: ThinVec<WherePredicate>,
}

impl Generics {
    pub(crate) fn is_empty(&self) -> bool {
        self.params.is_empty() && self.where_predicates.is_empty()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Function {
    pub(crate) decl: FnDecl,
    pub(crate) generics: Generics,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct FnDecl {
    pub(crate) inputs: Vec<Parameter>,
    pub(crate) output: Type,
    pub(crate) c_variadic: bool,
}

impl FnDecl {
    pub(crate) fn receiver_type(&self) -> Option<&Type> {
        self.inputs.first().and_then(|v| v.to_receiver())
    }
}

/// A function parameter.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct Parameter {
    pub(crate) name: Option<Symbol>,
    pub(crate) type_: Type,
    /// This field is used to represent "const" arguments from the `rustc_legacy_const_generics`
    /// feature. More information in <https://github.com/rust-lang/rust/issues/83167>.
    pub(crate) is_const: bool,
}

impl Parameter {
    pub(crate) fn to_receiver(&self) -> Option<&Type> {
        if self.name == Some(kw::SelfLower) { Some(&self.type_) } else { None }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Trait {
    pub(crate) def_id: DefId,
    pub(crate) items: Vec<Item>,
    pub(crate) generics: Generics,
    pub(crate) bounds: Vec<GenericBound>,
}

impl Trait {
    pub(crate) fn is_auto(&self, tcx: TyCtxt<'_>) -> bool {
        tcx.trait_is_auto(self.def_id)
    }
    pub(crate) fn is_notable_trait(&self, tcx: TyCtxt<'_>) -> bool {
        tcx.is_doc_notable_trait(self.def_id)
    }
    pub(crate) fn safety(&self, tcx: TyCtxt<'_>) -> hir::Safety {
        tcx.trait_def(self.def_id).safety
    }
    pub(crate) fn is_dyn_compatible(&self, tcx: TyCtxt<'_>) -> bool {
        tcx.is_dyn_compatible(self.def_id)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TraitAlias {
    pub(crate) generics: Generics,
    pub(crate) bounds: Vec<GenericBound>,
}

/// A trait reference, which may have higher ranked lifetimes.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct PolyTrait {
    pub(crate) trait_: Path,
    pub(crate) generic_params: Vec<GenericParamDef>,
}

/// Rustdoc's representation of types, mostly based on the [`hir::Ty`].
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum Type {
    /// A named type, which could be a trait.
    ///
    /// This is mostly Rustdoc's version of [`hir::Path`].
    /// It has to be different because Rustdoc's [`PathSegment`] can contain cleaned generics.
    Path {
        path: Path,
    },
    /// A `dyn Trait` object: `dyn for<'a> Trait<'a> + Send + 'static`
    DynTrait(Vec<PolyTrait>, Option<Lifetime>),
    /// A type parameter.
    Generic(Symbol),
    /// The `Self` type.
    SelfTy,
    /// A primitive (aka, builtin) type.
    Primitive(PrimitiveType),
    /// A function pointer: `extern "ABI" fn(...) -> ...`
    BareFunction(Box<BareFunctionDecl>),
    /// A tuple type: `(i32, &str)`.
    Tuple(Vec<Type>),
    /// A slice type (does *not* include the `&`): `[i32]`
    Slice(Box<Type>),
    /// An array type.
    ///
    /// The `String` field is a stringified version of the array's length parameter.
    Array(Box<Type>, Box<str>),
    Pat(Box<Type>, Box<str>),
    /// A raw pointer type: `*const i32`, `*mut i32`
    RawPointer(Mutability, Box<Type>),
    /// A reference type: `&i32`, `&'a mut Foo`
    BorrowedRef {
        lifetime: Option<Lifetime>,
        mutability: Mutability,
        type_: Box<Type>,
    },

    /// A qualified path to an associated item: `<Type as Trait>::Name`
    QPath(Box<QPathData>),

    /// A type that is inferred: `_`
    Infer,

    /// An `impl Trait`: `impl TraitA + TraitB + ...`
    ImplTrait(Vec<GenericBound>),

    UnsafeBinder(Box<UnsafeBinderTy>),
}

impl Type {
    /// When comparing types for equality, it can help to ignore `&` wrapping.
    pub(crate) fn without_borrowed_ref(&self) -> &Type {
        let mut result = self;
        while let Type::BorrowedRef { type_, .. } = result {
            result = type_;
        }
        result
    }

    pub(crate) fn is_borrowed_ref(&self) -> bool {
        matches!(self, Type::BorrowedRef { .. })
    }

    fn is_type_alias(&self) -> bool {
        matches!(self, Type::Path { path: Path { res: Res::Def(DefKind::TyAlias, _), .. } })
    }

    /// Check if two types are "the same" for documentation purposes.
    ///
    /// This is different from `Eq`, because it knows that things like
    /// `Placeholder` are possible matches for everything.
    ///
    /// This relation is not commutative when generics are involved:
    ///
    /// ```ignore(private)
    /// # // see types/tests.rs:is_same_generic for the real test
    /// use rustdoc::format::cache::Cache;
    /// use rustdoc::clean::types::{Type, PrimitiveType};
    /// let cache = Cache::new(false);
    /// let generic = Type::Generic(rustc_span::symbol::sym::Any);
    /// let unit = Type::Primitive(PrimitiveType::Unit);
    /// assert!(!generic.is_same(&unit, &cache));
    /// assert!(unit.is_same(&generic, &cache));
    /// ```
    ///
    /// An owned type is also the same as its borrowed variants (this is commutative),
    /// but `&T` is not the same as `&mut T`.
    pub(crate) fn is_doc_subtype_of(&self, other: &Self, cache: &Cache) -> bool {
        // Strip the references so that it can compare the actual types, unless both are references.
        // If both are references, leave them alone and compare the mutabilities later.
        let (self_cleared, other_cleared) = if !self.is_borrowed_ref() || !other.is_borrowed_ref() {
            (self.without_borrowed_ref(), other.without_borrowed_ref())
        } else {
            (self, other)
        };

        // FIXME: `Cache` does not have the data required to unwrap type aliases,
        // so we just assume they are equal.
        // This is only remotely acceptable because we were previously
        // assuming all types were equal when used
        // as a generic parameter of a type in `Deref::Target`.
        if self_cleared.is_type_alias() || other_cleared.is_type_alias() {
            return true;
        }

        match (self_cleared, other_cleared) {
            // Recursive cases.
            (Type::Tuple(a), Type::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.is_doc_subtype_of(b, cache))
            }
            (Type::Slice(a), Type::Slice(b)) => a.is_doc_subtype_of(b, cache),
            (Type::Array(a, al), Type::Array(b, bl)) => al == bl && a.is_doc_subtype_of(b, cache),
            (Type::RawPointer(mutability, type_), Type::RawPointer(b_mutability, b_type_)) => {
                mutability == b_mutability && type_.is_doc_subtype_of(b_type_, cache)
            }
            (
                Type::BorrowedRef { mutability, type_, .. },
                Type::BorrowedRef { mutability: b_mutability, type_: b_type_, .. },
            ) => mutability == b_mutability && type_.is_doc_subtype_of(b_type_, cache),
            // Placeholders are equal to all other types.
            (Type::Infer, _) | (_, Type::Infer) => true,
            // Generics match everything on the right, but not on the left.
            // If both sides are generic, this returns true.
            (_, Type::Generic(_)) => true,
            (Type::Generic(_), _) => false,
            // `Self` only matches itself.
            (Type::SelfTy, Type::SelfTy) => true,
            // Paths account for both the path itself and its generics.
            (Type::Path { path: a }, Type::Path { path: b }) => {
                a.def_id() == b.def_id()
                    && a.generics()
                        .zip(b.generics())
                        .map(|(ag, bg)| {
                            ag.iter().zip(bg.iter()).all(|(at, bt)| at.is_doc_subtype_of(bt, cache))
                        })
                        .unwrap_or(true)
            }
            // Other cases, such as primitives, just use recursion.
            (a, b) => a
                .def_id(cache)
                .and_then(|a| Some((a, b.def_id(cache)?)))
                .map(|(a, b)| a == b)
                .unwrap_or(false),
        }
    }

    pub(crate) fn primitive_type(&self) -> Option<PrimitiveType> {
        match *self {
            Primitive(p) | BorrowedRef { type_: box Primitive(p), .. } => Some(p),
            Slice(..) | BorrowedRef { type_: box Slice(..), .. } => Some(PrimitiveType::Slice),
            Array(..) | BorrowedRef { type_: box Array(..), .. } => Some(PrimitiveType::Array),
            Tuple(ref tys) => {
                if tys.is_empty() {
                    Some(PrimitiveType::Unit)
                } else {
                    Some(PrimitiveType::Tuple)
                }
            }
            RawPointer(..) => Some(PrimitiveType::RawPointer),
            BareFunction(..) => Some(PrimitiveType::Fn),
            _ => None,
        }
    }

    /// Returns the sugared return type for an async function.
    ///
    /// For example, if the return type is `impl std::future::Future<Output = i32>`, this function
    /// will return `i32`.
    ///
    /// # Panics
    ///
    /// This function will panic if the return type does not match the expected sugaring for async
    /// functions.
    pub(crate) fn sugared_async_return_type(self) -> Type {
        if let Type::ImplTrait(mut v) = self
            && let Some(GenericBound::TraitBound(PolyTrait { mut trait_, .. }, _)) = v.pop()
            && let Some(segment) = trait_.segments.pop()
            && let GenericArgs::AngleBracketed { mut constraints, .. } = segment.args
            && let Some(constraint) = constraints.pop()
            && let AssocItemConstraintKind::Equality { term } = constraint.kind
            && let Term::Type(ty) = term
        {
            ty
        } else {
            panic!("unexpected async fn return type")
        }
    }

    /// Checks if this is a `T::Name` path for an associated type.
    pub(crate) fn is_assoc_ty(&self) -> bool {
        match self {
            Type::Path { path, .. } => path.is_assoc_ty(),
            _ => false,
        }
    }

    pub(crate) fn is_self_type(&self) -> bool {
        matches!(*self, Type::SelfTy)
    }

    pub(crate) fn generic_args(&self) -> Option<&GenericArgs> {
        match self {
            Type::Path { path, .. } => path.generic_args(),
            _ => None,
        }
    }

    pub(crate) fn generics(&self) -> Option<Vec<&Type>> {
        match self {
            Type::Path { path, .. } => path.generics(),
            _ => None,
        }
    }

    pub(crate) fn is_full_generic(&self) -> bool {
        matches!(self, Type::Generic(_))
    }

    pub(crate) fn is_unit(&self) -> bool {
        matches!(self, Type::Tuple(v) if v.is_empty())
    }

    pub(crate) fn projection(&self) -> Option<(&Type, DefId, PathSegment)> {
        if let QPath(box QPathData { self_type, trait_, assoc, .. }) = self {
            Some((self_type, trait_.as_ref()?.def_id(), assoc.clone()))
        } else {
            None
        }
    }

    /// Use this method to get the [DefId] of a [clean] AST node, including [PrimitiveType]s.
    ///
    /// [clean]: crate::clean
    pub(crate) fn def_id(&self, cache: &Cache) -> Option<DefId> {
        let t: PrimitiveType = match *self {
            Type::Path { ref path } => return Some(path.def_id()),
            DynTrait(ref bounds, _) => return bounds.first().map(|b| b.trait_.def_id()),
            Primitive(p) => return cache.primitive_locations.get(&p).cloned(),
            BorrowedRef { type_: box Generic(..), .. } => PrimitiveType::Reference,
            BorrowedRef { ref type_, .. } => return type_.def_id(cache),
            Tuple(ref tys) => {
                if tys.is_empty() {
                    PrimitiveType::Unit
                } else {
                    PrimitiveType::Tuple
                }
            }
            BareFunction(..) => PrimitiveType::Fn,
            Slice(..) => PrimitiveType::Slice,
            Array(..) => PrimitiveType::Array,
            Type::Pat(..) => PrimitiveType::Pat,
            RawPointer(..) => PrimitiveType::RawPointer,
            QPath(box QPathData { ref self_type, .. }) => return self_type.def_id(cache),
            Generic(_) | SelfTy | Infer | ImplTrait(_) | UnsafeBinder(_) => return None,
        };
        Primitive(t).def_id(cache)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct QPathData {
    pub assoc: PathSegment,
    pub self_type: Type,
    /// FIXME: compute this field on demand.
    pub should_show_cast: bool,
    pub trait_: Option<Path>,
}

/// A primitive (aka, builtin) type.
///
/// This represents things like `i32`, `str`, etc.
///
/// N.B. This has to be different from [`hir::PrimTy`] because it also includes types that aren't
/// paths, like [`Self::Unit`].
#[derive(Clone, PartialEq, Eq, Hash, Copy, Debug)]
pub(crate) enum PrimitiveType {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
    F16,
    F32,
    F64,
    F128,
    Char,
    Bool,
    Str,
    Slice,
    Array,
    Pat,
    Tuple,
    Unit,
    RawPointer,
    Reference,
    Fn,
    Never,
}

type SimplifiedTypes = FxIndexMap<PrimitiveType, ArrayVec<SimplifiedType, 3>>;
impl PrimitiveType {
    pub(crate) fn from_hir(prim: hir::PrimTy) -> PrimitiveType {
        use ast::{FloatTy, IntTy, UintTy};
        match prim {
            hir::PrimTy::Int(IntTy::Isize) => PrimitiveType::Isize,
            hir::PrimTy::Int(IntTy::I8) => PrimitiveType::I8,
            hir::PrimTy::Int(IntTy::I16) => PrimitiveType::I16,
            hir::PrimTy::Int(IntTy::I32) => PrimitiveType::I32,
            hir::PrimTy::Int(IntTy::I64) => PrimitiveType::I64,
            hir::PrimTy::Int(IntTy::I128) => PrimitiveType::I128,
            hir::PrimTy::Uint(UintTy::Usize) => PrimitiveType::Usize,
            hir::PrimTy::Uint(UintTy::U8) => PrimitiveType::U8,
            hir::PrimTy::Uint(UintTy::U16) => PrimitiveType::U16,
            hir::PrimTy::Uint(UintTy::U32) => PrimitiveType::U32,
            hir::PrimTy::Uint(UintTy::U64) => PrimitiveType::U64,
            hir::PrimTy::Uint(UintTy::U128) => PrimitiveType::U128,
            hir::PrimTy::Float(FloatTy::F16) => PrimitiveType::F16,
            hir::PrimTy::Float(FloatTy::F32) => PrimitiveType::F32,
            hir::PrimTy::Float(FloatTy::F64) => PrimitiveType::F64,
            hir::PrimTy::Float(FloatTy::F128) => PrimitiveType::F128,
            hir::PrimTy::Str => PrimitiveType::Str,
            hir::PrimTy::Bool => PrimitiveType::Bool,
            hir::PrimTy::Char => PrimitiveType::Char,
        }
    }

    pub(crate) fn from_symbol(s: Symbol) -> Option<PrimitiveType> {
        match s {
            sym::isize => Some(PrimitiveType::Isize),
            sym::i8 => Some(PrimitiveType::I8),
            sym::i16 => Some(PrimitiveType::I16),
            sym::i32 => Some(PrimitiveType::I32),
            sym::i64 => Some(PrimitiveType::I64),
            sym::i128 => Some(PrimitiveType::I128),
            sym::usize => Some(PrimitiveType::Usize),
            sym::u8 => Some(PrimitiveType::U8),
            sym::u16 => Some(PrimitiveType::U16),
            sym::u32 => Some(PrimitiveType::U32),
            sym::u64 => Some(PrimitiveType::U64),
            sym::u128 => Some(PrimitiveType::U128),
            sym::bool => Some(PrimitiveType::Bool),
            sym::char => Some(PrimitiveType::Char),
            sym::str => Some(PrimitiveType::Str),
            sym::f16 => Some(PrimitiveType::F16),
            sym::f32 => Some(PrimitiveType::F32),
            sym::f64 => Some(PrimitiveType::F64),
            sym::f128 => Some(PrimitiveType::F128),
            sym::array => Some(PrimitiveType::Array),
            sym::slice => Some(PrimitiveType::Slice),
            sym::tuple => Some(PrimitiveType::Tuple),
            sym::unit => Some(PrimitiveType::Unit),
            sym::pointer => Some(PrimitiveType::RawPointer),
            sym::reference => Some(PrimitiveType::Reference),
            kw::Fn => Some(PrimitiveType::Fn),
            sym::never => Some(PrimitiveType::Never),
            _ => None,
        }
    }

    pub(crate) fn simplified_types() -> &'static SimplifiedTypes {
        use PrimitiveType::*;
        use ty::{FloatTy, IntTy, UintTy};
        static CELL: OnceCell<SimplifiedTypes> = OnceCell::new();

        let single = |x| iter::once(x).collect();
        CELL.get_or_init(move || {
            map! {
                Isize => single(SimplifiedType::Int(IntTy::Isize)),
                I8 => single(SimplifiedType::Int(IntTy::I8)),
                I16 => single(SimplifiedType::Int(IntTy::I16)),
                I32 => single(SimplifiedType::Int(IntTy::I32)),
                I64 => single(SimplifiedType::Int(IntTy::I64)),
                I128 => single(SimplifiedType::Int(IntTy::I128)),
                Usize => single(SimplifiedType::Uint(UintTy::Usize)),
                U8 => single(SimplifiedType::Uint(UintTy::U8)),
                U16 => single(SimplifiedType::Uint(UintTy::U16)),
                U32 => single(SimplifiedType::Uint(UintTy::U32)),
                U64 => single(SimplifiedType::Uint(UintTy::U64)),
                U128 => single(SimplifiedType::Uint(UintTy::U128)),
                F16 => single(SimplifiedType::Float(FloatTy::F16)),
                F32 => single(SimplifiedType::Float(FloatTy::F32)),
                F64 => single(SimplifiedType::Float(FloatTy::F64)),
                F128 => single(SimplifiedType::Float(FloatTy::F128)),
                Str => single(SimplifiedType::Str),
                Bool => single(SimplifiedType::Bool),
                Char => single(SimplifiedType::Char),
                Array => single(SimplifiedType::Array),
                Slice => single(SimplifiedType::Slice),
                // FIXME: If we ever add an inherent impl for tuples
                // with different lengths, they won't show in rustdoc.
                //
                // Either manually update this arrayvec at this point
                // or start with a more complex refactoring.
                Tuple => [SimplifiedType::Tuple(1), SimplifiedType::Tuple(2), SimplifiedType::Tuple(3)].into(),
                Unit => single(SimplifiedType::Tuple(0)),
                RawPointer => [SimplifiedType::Ptr(Mutability::Not), SimplifiedType::Ptr(Mutability::Mut)].into_iter().collect(),
                Reference => [SimplifiedType::Ref(Mutability::Not), SimplifiedType::Ref(Mutability::Mut)].into_iter().collect(),
                // FIXME: This will be wrong if we ever add inherent impls
                // for function pointers.
                Fn => single(SimplifiedType::Function(1)),
                Never => single(SimplifiedType::Never),
            }
        })
    }

    pub(crate) fn impls<'tcx>(&self, tcx: TyCtxt<'tcx>) -> impl Iterator<Item = DefId> + 'tcx {
        Self::simplified_types()
            .get(self)
            .into_iter()
            .flatten()
            .flat_map(move |&simp| tcx.incoherent_impls(simp).iter())
            .copied()
    }

    pub(crate) fn all_impls(tcx: TyCtxt<'_>) -> impl Iterator<Item = DefId> {
        Self::simplified_types()
            .values()
            .flatten()
            .flat_map(move |&simp| tcx.incoherent_impls(simp).iter())
            .copied()
    }

    pub(crate) fn as_sym(&self) -> Symbol {
        use PrimitiveType::*;
        match self {
            Isize => sym::isize,
            I8 => sym::i8,
            I16 => sym::i16,
            I32 => sym::i32,
            I64 => sym::i64,
            I128 => sym::i128,
            Usize => sym::usize,
            U8 => sym::u8,
            U16 => sym::u16,
            U32 => sym::u32,
            U64 => sym::u64,
            U128 => sym::u128,
            F16 => sym::f16,
            F32 => sym::f32,
            F64 => sym::f64,
            F128 => sym::f128,
            Str => sym::str,
            Bool => sym::bool,
            Char => sym::char,
            Array => sym::array,
            Pat => sym::pat,
            Slice => sym::slice,
            Tuple => sym::tuple,
            Unit => sym::unit,
            RawPointer => sym::pointer,
            Reference => sym::reference,
            Fn => kw::Fn,
            Never => sym::never,
        }
    }

    /// Returns the DefId of the module with `rustc_doc_primitive` for this primitive type.
    /// Panics if there is no such module.
    ///
    /// This gives precedence to primitives defined in the current crate, and deprioritizes
    /// primitives defined in `core`,
    /// but otherwise, if multiple crates define the same primitive, there is no guarantee of which
    /// will be picked.
    ///
    /// In particular, if a crate depends on both `std` and another crate that also defines
    /// `rustc_doc_primitive`, then it's entirely random whether `std` or the other crate is picked.
    /// (no_std crates are usually fine unless multiple dependencies define a primitive.)
    pub(crate) fn primitive_locations(tcx: TyCtxt<'_>) -> &FxIndexMap<PrimitiveType, DefId> {
        static PRIMITIVE_LOCATIONS: OnceCell<FxIndexMap<PrimitiveType, DefId>> = OnceCell::new();
        PRIMITIVE_LOCATIONS.get_or_init(|| {
            let mut primitive_locations = FxIndexMap::default();
            // NOTE: technically this misses crates that are only passed with `--extern` and not loaded when checking the crate.
            // This is a degenerate case that I don't plan to support.
            for &crate_num in tcx.crates(()) {
                let e = ExternalCrate { crate_num };
                let crate_name = e.name(tcx);
                debug!(?crate_num, ?crate_name);
                for &(def_id, prim) in &e.primitives(tcx) {
                    // HACK: try to link to std instead where possible
                    if crate_name == sym::core && primitive_locations.contains_key(&prim) {
                        continue;
                    }
                    primitive_locations.insert(prim, def_id);
                }
            }
            let local_primitives = ExternalCrate { crate_num: LOCAL_CRATE }.primitives(tcx);
            for (def_id, prim) in local_primitives {
                primitive_locations.insert(prim, def_id);
            }
            primitive_locations
        })
    }
}

impl From<ast::IntTy> for PrimitiveType {
    fn from(int_ty: ast::IntTy) -> PrimitiveType {
        match int_ty {
            ast::IntTy::Isize => PrimitiveType::Isize,
            ast::IntTy::I8 => PrimitiveType::I8,
            ast::IntTy::I16 => PrimitiveType::I16,
            ast::IntTy::I32 => PrimitiveType::I32,
            ast::IntTy::I64 => PrimitiveType::I64,
            ast::IntTy::I128 => PrimitiveType::I128,
        }
    }
}

impl From<ast::UintTy> for PrimitiveType {
    fn from(uint_ty: ast::UintTy) -> PrimitiveType {
        match uint_ty {
            ast::UintTy::Usize => PrimitiveType::Usize,
            ast::UintTy::U8 => PrimitiveType::U8,
            ast::UintTy::U16 => PrimitiveType::U16,
            ast::UintTy::U32 => PrimitiveType::U32,
            ast::UintTy::U64 => PrimitiveType::U64,
            ast::UintTy::U128 => PrimitiveType::U128,
        }
    }
}

impl From<ast::FloatTy> for PrimitiveType {
    fn from(float_ty: ast::FloatTy) -> PrimitiveType {
        match float_ty {
            ast::FloatTy::F16 => PrimitiveType::F16,
            ast::FloatTy::F32 => PrimitiveType::F32,
            ast::FloatTy::F64 => PrimitiveType::F64,
            ast::FloatTy::F128 => PrimitiveType::F128,
        }
    }
}

impl From<ty::IntTy> for PrimitiveType {
    fn from(int_ty: ty::IntTy) -> PrimitiveType {
        match int_ty {
            ty::IntTy::Isize => PrimitiveType::Isize,
            ty::IntTy::I8 => PrimitiveType::I8,
            ty::IntTy::I16 => PrimitiveType::I16,
            ty::IntTy::I32 => PrimitiveType::I32,
            ty::IntTy::I64 => PrimitiveType::I64,
            ty::IntTy::I128 => PrimitiveType::I128,
        }
    }
}

impl From<ty::UintTy> for PrimitiveType {
    fn from(uint_ty: ty::UintTy) -> PrimitiveType {
        match uint_ty {
            ty::UintTy::Usize => PrimitiveType::Usize,
            ty::UintTy::U8 => PrimitiveType::U8,
            ty::UintTy::U16 => PrimitiveType::U16,
            ty::UintTy::U32 => PrimitiveType::U32,
            ty::UintTy::U64 => PrimitiveType::U64,
            ty::UintTy::U128 => PrimitiveType::U128,
        }
    }
}

impl From<ty::FloatTy> for PrimitiveType {
    fn from(float_ty: ty::FloatTy) -> PrimitiveType {
        match float_ty {
            ty::FloatTy::F16 => PrimitiveType::F16,
            ty::FloatTy::F32 => PrimitiveType::F32,
            ty::FloatTy::F64 => PrimitiveType::F64,
            ty::FloatTy::F128 => PrimitiveType::F128,
        }
    }
}

impl From<hir::PrimTy> for PrimitiveType {
    fn from(prim_ty: hir::PrimTy) -> PrimitiveType {
        match prim_ty {
            hir::PrimTy::Int(int_ty) => int_ty.into(),
            hir::PrimTy::Uint(uint_ty) => uint_ty.into(),
            hir::PrimTy::Float(float_ty) => float_ty.into(),
            hir::PrimTy::Str => PrimitiveType::Str,
            hir::PrimTy::Bool => PrimitiveType::Bool,
            hir::PrimTy::Char => PrimitiveType::Char,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Struct {
    pub(crate) ctor_kind: Option<CtorKind>,
    pub(crate) generics: Generics,
    pub(crate) fields: ThinVec<Item>,
}

impl Struct {
    pub(crate) fn has_stripped_entries(&self) -> bool {
        self.fields.iter().any(|f| f.is_stripped())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Union {
    pub(crate) generics: Generics,
    pub(crate) fields: Vec<Item>,
}

impl Union {
    pub(crate) fn has_stripped_entries(&self) -> bool {
        self.fields.iter().any(|f| f.is_stripped())
    }
}

/// This is a more limited form of the standard Struct, different in that
/// it lacks the things most items have (name, id, parameterization). Found
/// only as a variant in an enum.
#[derive(Clone, Debug)]
pub(crate) struct VariantStruct {
    pub(crate) fields: ThinVec<Item>,
}

impl VariantStruct {
    pub(crate) fn has_stripped_entries(&self) -> bool {
        self.fields.iter().any(|f| f.is_stripped())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Enum {
    pub(crate) variants: IndexVec<VariantIdx, Item>,
    pub(crate) generics: Generics,
}

impl Enum {
    pub(crate) fn has_stripped_entries(&self) -> bool {
        self.variants.iter().any(|f| f.is_stripped())
    }

    pub(crate) fn variants(&self) -> impl Iterator<Item = &Item> {
        self.variants.iter().filter(|v| !v.is_stripped())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Variant {
    pub kind: VariantKind,
    pub discriminant: Option<Discriminant>,
}

#[derive(Clone, Debug)]
pub(crate) enum VariantKind {
    CLike,
    Tuple(ThinVec<Item>),
    Struct(VariantStruct),
}

impl Variant {
    pub(crate) fn has_stripped_entries(&self) -> Option<bool> {
        match &self.kind {
            VariantKind::Struct(struct_) => Some(struct_.has_stripped_entries()),
            VariantKind::CLike | VariantKind::Tuple(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Discriminant {
    // In the case of cross crate re-exports, we don't have the necessary information
    // to reconstruct the expression of the discriminant, only the value.
    pub(super) expr: Option<BodyId>,
    pub(super) value: DefId,
}

impl Discriminant {
    /// Will be `None` in the case of cross-crate reexports, and may be
    /// simplified
    pub(crate) fn expr(&self, tcx: TyCtxt<'_>) -> Option<String> {
        self.expr
            .map(|body| rendered_const(tcx, tcx.hir_body(body), tcx.hir_body_owner_def_id(body)))
    }
    pub(crate) fn value(&self, tcx: TyCtxt<'_>, with_underscores: bool) -> String {
        print_evaluated_const(tcx, self.value, with_underscores, false).unwrap()
    }
}

/// Small wrapper around [`rustc_span::Span`] that adds helper methods
/// and enforces calling [`rustc_span::Span::source_callsite()`].
#[derive(Copy, Clone, Debug)]
pub(crate) struct Span(rustc_span::Span);

impl Span {
    /// Wraps a [`rustc_span::Span`]. In case this span is the result of a macro expansion, the
    /// span will be updated to point to the macro invocation instead of the macro definition.
    ///
    /// (See rust-lang/rust#39726)
    pub(crate) fn new(sp: rustc_span::Span) -> Self {
        Self(sp.source_callsite())
    }

    pub(crate) fn inner(&self) -> rustc_span::Span {
        self.0
    }

    pub(crate) fn filename(&self, sess: &Session) -> FileName {
        sess.source_map().span_to_filename(self.0)
    }

    pub(crate) fn lo(&self, sess: &Session) -> Loc {
        sess.source_map().lookup_char_pos(self.0.lo())
    }

    pub(crate) fn hi(&self, sess: &Session) -> Loc {
        sess.source_map().lookup_char_pos(self.0.hi())
    }

    pub(crate) fn cnum(&self, sess: &Session) -> CrateNum {
        // FIXME: is there a time when the lo and hi crate would be different?
        self.lo(sess).file.cnum
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct Path {
    pub(crate) res: Res,
    pub(crate) segments: ThinVec<PathSegment>,
}

impl Path {
    pub(crate) fn def_id(&self) -> DefId {
        self.res.def_id()
    }

    pub(crate) fn last_opt(&self) -> Option<Symbol> {
        self.segments.last().map(|s| s.name)
    }

    pub(crate) fn last(&self) -> Symbol {
        self.last_opt().expect("segments were empty")
    }

    pub(crate) fn whole_name(&self) -> String {
        self.segments
            .iter()
            .map(|s| if s.name == kw::PathRoot { "" } else { s.name.as_str() })
            .intersperse("::")
            .collect()
    }

    /// Checks if this is a `T::Name` path for an associated type.
    pub(crate) fn is_assoc_ty(&self) -> bool {
        match self.res {
            Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } | Res::Def(DefKind::TyParam, _)
                if self.segments.len() != 1 =>
            {
                true
            }
            Res::Def(DefKind::AssocTy, _) => true,
            _ => false,
        }
    }

    pub(crate) fn generic_args(&self) -> Option<&GenericArgs> {
        self.segments.last().map(|seg| &seg.args)
    }

    pub(crate) fn generics(&self) -> Option<Vec<&Type>> {
        self.segments.last().and_then(|seg| {
            if let GenericArgs::AngleBracketed { ref args, .. } = seg.args {
                Some(
                    args.iter()
                        .filter_map(|arg| match arg {
                            GenericArg::Type(ty) => Some(ty),
                            _ => None,
                        })
                        .collect(),
                )
            } else {
                None
            }
        })
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum GenericArg {
    Lifetime(Lifetime),
    Type(Type),
    Const(Box<ConstantKind>),
    Infer,
}

impl GenericArg {
    pub(crate) fn as_lt(&self) -> Option<&Lifetime> {
        if let Self::Lifetime(lt) = self { Some(lt) } else { None }
    }

    pub(crate) fn as_ty(&self) -> Option<&Type> {
        if let Self::Type(ty) = self { Some(ty) } else { None }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum GenericArgs {
    /// `<args, constraints = ..>`
    AngleBracketed { args: ThinVec<GenericArg>, constraints: ThinVec<AssocItemConstraint> },
    /// `(inputs) -> output`
    Parenthesized { inputs: ThinVec<Type>, output: Option<Box<Type>> },
    /// `(..)`
    ReturnTypeNotation,
}

impl GenericArgs {
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            GenericArgs::AngleBracketed { args, constraints } => {
                args.is_empty() && constraints.is_empty()
            }
            GenericArgs::Parenthesized { inputs, output } => inputs.is_empty() && output.is_none(),
            GenericArgs::ReturnTypeNotation => false,
        }
    }
    pub(crate) fn constraints(&self) -> Box<dyn Iterator<Item = AssocItemConstraint> + '_> {
        match self {
            GenericArgs::AngleBracketed { constraints, .. } => {
                Box::new(constraints.iter().cloned())
            }
            GenericArgs::Parenthesized { output, .. } => Box::new(
                output
                    .as_ref()
                    .map(|ty| AssocItemConstraint {
                        assoc: PathSegment {
                            name: sym::Output,
                            args: GenericArgs::AngleBracketed {
                                args: ThinVec::new(),
                                constraints: ThinVec::new(),
                            },
                        },
                        kind: AssocItemConstraintKind::Equality {
                            term: Term::Type((**ty).clone()),
                        },
                    })
                    .into_iter(),
            ),
            GenericArgs::ReturnTypeNotation => Box::new([].into_iter()),
        }
    }
}

impl<'a> IntoIterator for &'a GenericArgs {
    type IntoIter = Box<dyn Iterator<Item = GenericArg> + 'a>;
    type Item = GenericArg;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            GenericArgs::AngleBracketed { args, .. } => Box::new(args.iter().cloned()),
            GenericArgs::Parenthesized { inputs, .. } => {
                // FIXME: This isn't really right, since `Fn(A, B)` is `Fn<(A, B)>`
                Box::new(inputs.iter().cloned().map(GenericArg::Type))
            }
            GenericArgs::ReturnTypeNotation => Box::new([].into_iter()),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct PathSegment {
    pub(crate) name: Symbol,
    pub(crate) args: GenericArgs,
}

#[derive(Clone, Debug)]
pub(crate) enum TypeAliasInnerType {
    Enum { variants: IndexVec<VariantIdx, Item>, is_non_exhaustive: bool },
    Union { fields: Vec<Item> },
    Struct { ctor_kind: Option<CtorKind>, fields: Vec<Item> },
}

#[derive(Clone, Debug)]
pub(crate) struct TypeAlias {
    pub(crate) type_: Type,
    pub(crate) generics: Generics,
    /// Inner `AdtDef` type, ie `type TyKind = IrTyKind<Adt, Ty>`,
    /// to be shown directly on the typedef page.
    pub(crate) inner_type: Option<TypeAliasInnerType>,
    /// `type_` can come from either the HIR or from metadata. If it comes from HIR, it may be a type
    /// alias instead of the final type. This will always have the final type, regardless of whether
    /// `type_` came from HIR or from metadata.
    ///
    /// If `item_type.is_none()`, `type_` is guaranteed to come from metadata (and therefore hold the
    /// final type).
    pub(crate) item_type: Option<Type>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct BareFunctionDecl {
    pub(crate) safety: hir::Safety,
    pub(crate) generic_params: Vec<GenericParamDef>,
    pub(crate) decl: FnDecl,
    pub(crate) abi: ExternAbi,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct UnsafeBinderTy {
    pub(crate) generic_params: Vec<GenericParamDef>,
    pub(crate) ty: Type,
}

#[derive(Clone, Debug)]
pub(crate) struct Static {
    pub(crate) type_: Box<Type>,
    pub(crate) mutability: Mutability,
    pub(crate) expr: Option<BodyId>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Constant {
    pub(crate) generics: Generics,
    pub(crate) kind: ConstantKind,
    pub(crate) type_: Type,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Term {
    Type(Type),
    Constant(ConstantKind),
}

impl Term {
    pub(crate) fn ty(&self) -> Option<&Type> {
        if let Term::Type(ty) = self { Some(ty) } else { None }
    }
}

impl From<Type> for Term {
    fn from(ty: Type) -> Self {
        Term::Type(ty)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ConstantKind {
    /// This is the wrapper around `ty::Const` for a non-local constant. Because it doesn't have a
    /// `BodyId`, we need to handle it on its own.
    ///
    /// Note that `ty::Const` includes generic parameters, and may not always be uniquely identified
    /// by a DefId. So this field must be different from `Extern`.
    TyConst { expr: Box<str> },
    /// A constant that is just a path (i.e., referring to a const param, free const, etc.).
    // FIXME: this is an unfortunate representation. rustdoc's logic around consts needs to be improved.
    Path { path: Box<str> },
    /// A constant (expression) that's not an item or associated item. These are usually found
    /// nested inside types (e.g., array lengths) or expressions (e.g., repeat counts), and also
    /// used to define explicit discriminant values for enum variants.
    Anonymous { body: BodyId },
    /// A constant from a different crate.
    Extern { def_id: DefId },
    /// `const FOO: u32 = ...;`
    Local { def_id: DefId, body: BodyId },
    /// An inferred constant as in `[10u8; _]`.
    Infer,
}

impl Constant {
    pub(crate) fn expr(&self, tcx: TyCtxt<'_>) -> String {
        self.kind.expr(tcx)
    }

    pub(crate) fn value(&self, tcx: TyCtxt<'_>) -> Option<String> {
        self.kind.value(tcx)
    }

    pub(crate) fn is_literal(&self, tcx: TyCtxt<'_>) -> bool {
        self.kind.is_literal(tcx)
    }
}

impl ConstantKind {
    pub(crate) fn expr(&self, tcx: TyCtxt<'_>) -> String {
        match *self {
            ConstantKind::TyConst { ref expr } => expr.to_string(),
            ConstantKind::Path { ref path } => path.to_string(),
            ConstantKind::Extern { def_id } => print_inlined_const(tcx, def_id),
            ConstantKind::Local { body, .. } | ConstantKind::Anonymous { body } => {
                rendered_const(tcx, tcx.hir_body(body), tcx.hir_body_owner_def_id(body))
            }
            ConstantKind::Infer => "_".to_string(),
        }
    }

    pub(crate) fn value(&self, tcx: TyCtxt<'_>) -> Option<String> {
        match *self {
            ConstantKind::TyConst { .. }
            | ConstantKind::Path { .. }
            | ConstantKind::Anonymous { .. }
            | ConstantKind::Infer => None,
            ConstantKind::Extern { def_id } | ConstantKind::Local { def_id, .. } => {
                print_evaluated_const(tcx, def_id, true, true)
            }
        }
    }

    pub(crate) fn is_literal(&self, tcx: TyCtxt<'_>) -> bool {
        match *self {
            ConstantKind::TyConst { .. }
            | ConstantKind::Extern { .. }
            | ConstantKind::Path { .. }
            | ConstantKind::Infer => false,
            ConstantKind::Local { body, .. } | ConstantKind::Anonymous { body } => {
                is_literal_expr(tcx, body.hir_id)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Impl {
    pub(crate) safety: hir::Safety,
    pub(crate) generics: Generics,
    pub(crate) trait_: Option<Path>,
    pub(crate) for_: Type,
    pub(crate) items: Vec<Item>,
    pub(crate) polarity: ty::ImplPolarity,
    pub(crate) kind: ImplKind,
}

impl Impl {
    pub(crate) fn provided_trait_methods(&self, tcx: TyCtxt<'_>) -> FxIndexSet<Symbol> {
        self.trait_
            .as_ref()
            .map(|t| t.def_id())
            .map(|did| tcx.provided_trait_methods(did).map(|meth| meth.name()).collect())
            .unwrap_or_default()
    }

    pub(crate) fn is_negative_trait_impl(&self) -> bool {
        matches!(self.polarity, ty::ImplPolarity::Negative)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ImplKind {
    Normal,
    Auto,
    FakeVariadic,
    Blanket(Box<Type>),
}

impl ImplKind {
    pub(crate) fn is_auto(&self) -> bool {
        matches!(self, ImplKind::Auto)
    }

    pub(crate) fn is_blanket(&self) -> bool {
        matches!(self, ImplKind::Blanket(_))
    }

    pub(crate) fn is_fake_variadic(&self) -> bool {
        matches!(self, ImplKind::FakeVariadic)
    }

    pub(crate) fn as_blanket_ty(&self) -> Option<&Type> {
        match self {
            ImplKind::Blanket(ty) => Some(ty),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Import {
    pub(crate) kind: ImportKind,
    /// The item being re-exported.
    pub(crate) source: ImportSource,
    pub(crate) should_be_displayed: bool,
}

impl Import {
    pub(crate) fn new_simple(
        name: Symbol,
        source: ImportSource,
        should_be_displayed: bool,
    ) -> Self {
        Self { kind: ImportKind::Simple(name), source, should_be_displayed }
    }

    pub(crate) fn new_glob(source: ImportSource, should_be_displayed: bool) -> Self {
        Self { kind: ImportKind::Glob, source, should_be_displayed }
    }

    pub(crate) fn imported_item_is_doc_hidden(&self, tcx: TyCtxt<'_>) -> bool {
        self.source.did.is_some_and(|did| tcx.is_doc_hidden(did))
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ImportKind {
    // use source as str;
    Simple(Symbol),
    // use source::*;
    Glob,
}

#[derive(Clone, Debug)]
pub(crate) struct ImportSource {
    pub(crate) path: Path,
    pub(crate) did: Option<DefId>,
}

#[derive(Clone, Debug)]
pub(crate) struct Macro {
    pub(crate) source: String,
    /// Whether the macro was defined via `macro_rules!` as opposed to `macro`.
    pub(crate) macro_rules: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ProcMacro {
    pub(crate) kind: MacroKind,
    pub(crate) helpers: Vec<Symbol>,
}

/// A constraint on an associated item.
///
/// ### Examples
///
/// * the `A = Ty` and `B = Ty` in `Trait<A = Ty, B = Ty>`
/// * the `G<Ty> = Ty` in `Trait<G<Ty> = Ty>`
/// * the `A: Bound` in `Trait<A: Bound>`
/// * the `RetTy` in `Trait(ArgTy, ArgTy) -> RetTy`
/// * the `C = { Ct }` in `Trait<C = { Ct }>` (feature `associated_const_equality`)
/// * the `f(..): Bound` in `Trait<f(..): Bound>` (feature `return_type_notation`)
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct AssocItemConstraint {
    pub(crate) assoc: PathSegment,
    pub(crate) kind: AssocItemConstraintKind,
}

/// The kind of [associated item constraint][AssocItemConstraint].
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum AssocItemConstraintKind {
    Equality { term: Term },
    Bound { bounds: Vec<GenericBound> },
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(Crate, 16); // frequently moved by-value
    static_assert_size!(DocFragment, 32);
    static_assert_size!(GenericArg, 32);
    static_assert_size!(GenericArgs, 24);
    static_assert_size!(GenericParamDef, 40);
    static_assert_size!(Generics, 16);
    static_assert_size!(Item, 8);
    static_assert_size!(ItemInner, 136);
    static_assert_size!(ItemKind, 48);
    static_assert_size!(PathSegment, 32);
    static_assert_size!(Type, 32);
    // tidy-alphabetical-end
}
