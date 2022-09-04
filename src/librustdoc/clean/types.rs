use std::cell::RefCell;
use std::default::Default;
use std::hash::Hash;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::OnceLock as OnceCell;
use std::{cmp, fmt, iter};

use arrayvec::ArrayVec;
use thin_vec::ThinVec;

use rustc_ast::attr;
use rustc_ast::util::comments::beautify_doc_string;
use rustc_ast::{self as ast, AttrStyle};
use rustc_attr::{ConstStability, Deprecation, Stability, StabilityLevel};
use rustc_const_eval::const_eval::is_unstable_const_fn;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{BodyId, Mutability};
use rustc_index::vec::IndexVec;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::Session;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{self, FileName, Loc};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi::Abi;
use rustc_typeck::check::intrinsic::intrinsic_operation_unsafety;

use crate::clean::cfg::Cfg;
use crate::clean::clean_visibility;
use crate::clean::external_path;
use crate::clean::inline::{self, print_inlined_const};
use crate::clean::utils::{is_literal_expr, print_const_expr, print_evaluated_const};
use crate::core::DocContext;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::render::Context;
use crate::passes::collect_intra_doc_links::UrlFragment;

pub(crate) use self::FnRetTy::*;
pub(crate) use self::ItemKind::*;
pub(crate) use self::SelfTy::*;
pub(crate) use self::Type::{
    Array, BareFunction, BorrowedRef, DynTrait, Generic, ImplTrait, Infer, Primitive, QPath,
    RawPointer, Slice, Tuple,
};
pub(crate) use self::Visibility::{Inherited, Public};

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
    /// Identifier for primitive types.
    Primitive(PrimitiveType, CrateNum),
}

impl ItemId {
    #[inline]
    pub(crate) fn is_local(self) -> bool {
        match self {
            ItemId::Auto { for_: id, .. }
            | ItemId::Blanket { for_: id, .. }
            | ItemId::DefId(id) => id.is_local(),
            ItemId::Primitive(_, krate) => krate == LOCAL_CRATE,
        }
    }

    #[inline]
    #[track_caller]
    pub(crate) fn expect_def_id(self) -> DefId {
        self.as_def_id()
            .unwrap_or_else(|| panic!("ItemId::expect_def_id: `{:?}` isn't a DefId", self))
    }

    #[inline]
    pub(crate) fn as_def_id(self) -> Option<DefId> {
        match self {
            ItemId::DefId(id) => Some(id),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn krate(self) -> CrateNum {
        match self {
            ItemId::Auto { for_: id, .. }
            | ItemId::Blanket { for_: id, .. }
            | ItemId::DefId(id) => id.krate,
            ItemId::Primitive(_, krate) => krate,
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
    pub(crate) primitives: ThinVec<(DefId, PrimitiveType)>,
    /// Only here so that they can be filtered through the rustdoc passes.
    pub(crate) external_traits: Rc<RefCell<FxHashMap<DefId, TraitWithExtraInfo>>>,
}

impl Crate {
    pub(crate) fn name(&self, tcx: TyCtxt<'_>) -> Symbol {
        ExternalCrate::LOCAL.name(tcx)
    }

    pub(crate) fn src(&self, tcx: TyCtxt<'_>) -> FileName {
        ExternalCrate::LOCAL.src(tcx)
    }
}

/// This struct is used to wrap additional information added by rustdoc on a `trait` item.
#[derive(Clone, Debug)]
pub(crate) struct TraitWithExtraInfo {
    pub(crate) trait_: Trait,
    pub(crate) is_notable: bool,
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
    /// rendering in to the specified source destination.
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

        if extern_url_takes_precedence {
            if let Some(url) = extern_url {
                return to_remote(url);
            }
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
                    if meta.has_name(sym::keyword) {
                        if let Some(v) = meta.value_str() {
                            keyword = Some(v);
                            break;
                        }
                    }
                }
                return keyword.map(|p| (def_id, p));
            }
            None
        };
        if root.is_local() {
            tcx.hir()
                .root_module()
                .item_ids
                .iter()
                .filter_map(|&id| {
                    let item = tcx.hir().item(id);
                    match item.kind {
                        hir::ItemKind::Mod(_) => {
                            as_keyword(Res::Def(DefKind::Mod, id.def_id.to_def_id()))
                        }
                        hir::ItemKind::Use(path, hir::UseKind::Single)
                            if tcx.visibility(id.def_id).is_public() =>
                        {
                            as_keyword(path.res.expect_non_local())
                                .map(|(_, prim)| (id.def_id.to_def_id(), prim))
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
        // item tagged with `#[doc(primitive)]` then we would also have to
        // search the entirety of external modules for items tagged
        // `#[doc(primitive)]`, which is a pretty inefficient process (decoding
        // all that metadata unconditionally).
        //
        // In order to keep the metadata load under control, the
        // `#[doc(primitive)]` feature is explicitly designed to only allow the
        // primitive tags to show up as the top level items in a crate.
        //
        // Also note that this does not attempt to deal with modules tagged
        // duplicately for the same primitive. This is handled later on when
        // rendering by delegating everything to a hash map.
        let as_primitive = |res: Res<!>| {
            if let Res::Def(DefKind::Mod, def_id) = res {
                let mut prim = None;
                let meta_items = tcx
                    .get_attrs(def_id, sym::doc)
                    .flat_map(|attr| attr.meta_item_list().unwrap_or_default());
                for meta in meta_items {
                    if let Some(v) = meta.value_str() {
                        if meta.has_name(sym::primitive) {
                            prim = PrimitiveType::from_symbol(v);
                            if prim.is_some() {
                                break;
                            }
                            // FIXME: should warn on unknown primitives?
                        }
                    }
                }
                return prim.map(|p| (def_id, p));
            }
            None
        };

        if root.is_local() {
            tcx.hir()
                .root_module()
                .item_ids
                .iter()
                .filter_map(|&id| {
                    let item = tcx.hir().item(id);
                    match item.kind {
                        hir::ItemKind::Mod(_) => {
                            as_primitive(Res::Def(DefKind::Mod, id.def_id.to_def_id()))
                        }
                        hir::ItemKind::Use(path, hir::UseKind::Single)
                            if tcx.visibility(id.def_id).is_public() =>
                        {
                            as_primitive(path.res.expect_non_local()).map(|(_, prim)| {
                                // Pretend the primitive is local.
                                (id.def_id.to_def_id(), prim)
                            })
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
    /// The name of this item.
    /// Optional because not every item has a name, e.g. impls.
    pub(crate) name: Option<Symbol>,
    pub(crate) attrs: Box<Attributes>,
    pub(crate) visibility: Visibility,
    /// Information about this item that is specific to what kind of item it is.
    /// E.g., struct vs enum vs function.
    pub(crate) kind: Box<ItemKind>,
    pub(crate) item_id: ItemId,

    pub(crate) cfg: Option<Arc<Cfg>>,
}

/// NOTE: this does NOT unconditionally print every item, to avoid thousands of lines of logs.
/// If you want to see the debug output for attributes and the `kind` as well, use `{:#?}` instead of `{:?}`.
impl fmt::Debug for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let alternate = f.alternate();
        // hand-picked fields that don't bloat the logs too much
        let mut fmt = f.debug_struct("Item");
        fmt.field("name", &self.name)
            .field("visibility", &self.visibility)
            .field("item_id", &self.item_id);
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
        |local| {
            let hir = tcx.hir();
            hir.span_with_body(hir.local_def_id_to_hir_id(local))
        },
    ))
}

impl Item {
    pub(crate) fn stability<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Option<Stability> {
        self.item_id.as_def_id().and_then(|did| tcx.lookup_stability(did))
    }

    pub(crate) fn const_stability<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Option<ConstStability> {
        self.item_id.as_def_id().and_then(|did| tcx.lookup_const_stability(did))
    }

    pub(crate) fn deprecation(&self, tcx: TyCtxt<'_>) -> Option<Deprecation> {
        self.item_id.as_def_id().and_then(|did| tcx.lookup_deprecation(did))
    }

    pub(crate) fn inner_docs(&self, tcx: TyCtxt<'_>) -> bool {
        self.item_id
            .as_def_id()
            .map(|did| tcx.get_attrs_unchecked(did).inner_docs())
            .unwrap_or(false)
    }

    pub(crate) fn span(&self, tcx: TyCtxt<'_>) -> Option<Span> {
        let kind = match &*self.kind {
            ItemKind::StrippedItem(k) => k,
            _ => &*self.kind,
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
            _ => self.item_id.as_def_id().map(|did| rustc_span(did, tcx)),
        }
    }

    pub(crate) fn attr_span(&self, tcx: TyCtxt<'_>) -> rustc_span::Span {
        crate::passes::span_of_attrs(&self.attrs)
            .unwrap_or_else(|| self.span(tcx).map_or(rustc_span::DUMMY_SP, |span| span.inner()))
    }

    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    pub(crate) fn doc_value(&self) -> Option<String> {
        self.attrs.doc_value()
    }

    /// Convenience wrapper around [`Self::from_def_id_and_parts`] which converts
    /// `hir_id` to a [`DefId`]
    pub(crate) fn from_hir_id_and_parts(
        hir_id: hir::HirId,
        name: Option<Symbol>,
        kind: ItemKind,
        cx: &mut DocContext<'_>,
    ) -> Item {
        Item::from_def_id_and_parts(cx.tcx.hir().local_def_id(hir_id).to_def_id(), name, kind, cx)
    }

    pub(crate) fn from_def_id_and_parts(
        def_id: DefId,
        name: Option<Symbol>,
        kind: ItemKind,
        cx: &mut DocContext<'_>,
    ) -> Item {
        let ast_attrs = cx.tcx.get_attrs_unchecked(def_id);

        Self::from_def_id_and_attrs_and_parts(
            def_id,
            name,
            kind,
            Box::new(Attributes::from_ast(ast_attrs)),
            cx,
            ast_attrs.cfg(cx.tcx, &cx.cache.hidden_cfg),
        )
    }

    pub(crate) fn from_def_id_and_attrs_and_parts(
        def_id: DefId,
        name: Option<Symbol>,
        kind: ItemKind,
        attrs: Box<Attributes>,
        cx: &mut DocContext<'_>,
        cfg: Option<Arc<Cfg>>,
    ) -> Item {
        trace!("name={:?}, def_id={:?} cfg={:?}", name, def_id, cfg);

        // Primitives and Keywords are written in the source code as private modules.
        // The modules need to be private so that nobody actually uses them, but the
        // keywords and primitives that they are documenting are public.
        let visibility = if matches!(&kind, ItemKind::KeywordItem | ItemKind::PrimitiveItem(..)) {
            Visibility::Public
        } else {
            clean_visibility(cx.tcx.visibility(def_id))
        };

        Item { item_id: def_id.into(), kind: Box::new(kind), name, attrs, visibility, cfg }
    }

    /// Finds all `doc` attributes as NameValues and returns their corresponding values, joined
    /// with newlines.
    pub(crate) fn collapsed_doc_value(&self) -> Option<String> {
        self.attrs.collapsed_doc_value()
    }

    pub(crate) fn links(&self, cx: &Context<'_>) -> Vec<RenderedLink> {
        use crate::html::format::href;

        cx.cache()
            .intra_doc_links
            .get(&self.item_id)
            .map_or(&[][..], |v| v.as_slice())
            .iter()
            .filter_map(|ItemLink { link: s, link_text, did, ref fragment }| {
                debug!(?did);
                if let Ok((mut href, ..)) = href(*did, cx) {
                    debug!(?href);
                    if let Some(ref fragment) = *fragment {
                        fragment.render(&mut href, cx.tcx())
                    }
                    Some(RenderedLink {
                        original_text: s.clone(),
                        new_text: link_text.clone(),
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
        cache
            .intra_doc_links
            .get(&self.item_id)
            .map_or(&[][..], |v| v.as_slice())
            .iter()
            .map(|ItemLink { link: s, link_text, .. }| RenderedLink {
                original_text: s.clone(),
                new_text: link_text.clone(),
                href: String::new(),
            })
            .collect()
    }

    pub(crate) fn is_crate(&self) -> bool {
        self.is_mod() && self.item_id.as_def_id().map_or(false, |did| did.is_crate_root())
    }
    pub(crate) fn is_mod(&self) -> bool {
        self.type_() == ItemType::Module
    }
    pub(crate) fn is_trait(&self) -> bool {
        self.type_() == ItemType::Trait
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
        matches!(&*self.kind, AssocTypeItem(..) | StrippedItem(box AssocTypeItem(..)))
    }
    pub(crate) fn is_ty_associated_type(&self) -> bool {
        matches!(&*self.kind, TyAssocTypeItem(..) | StrippedItem(box TyAssocTypeItem(..)))
    }
    pub(crate) fn is_associated_const(&self) -> bool {
        matches!(&*self.kind, AssocConstItem(..) | StrippedItem(box AssocConstItem(..)))
    }
    pub(crate) fn is_ty_associated_const(&self) -> bool {
        matches!(&*self.kind, TyAssocConstItem(..) | StrippedItem(box TyAssocConstItem(..)))
    }
    pub(crate) fn is_method(&self) -> bool {
        self.type_() == ItemType::Method
    }
    pub(crate) fn is_ty_method(&self) -> bool {
        self.type_() == ItemType::TyMethod
    }
    pub(crate) fn is_typedef(&self) -> bool {
        self.type_() == ItemType::Typedef
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
        match *self.kind {
            StrippedItem(..) => true,
            ImportItem(ref i) => !i.should_be_displayed,
            _ => false,
        }
    }
    pub(crate) fn has_stripped_entries(&self) -> Option<bool> {
        match *self.kind {
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

    pub(crate) fn stable_since(&self, tcx: TyCtxt<'_>) -> Option<Symbol> {
        match self.stability(tcx)?.level {
            StabilityLevel::Stable { since, .. } => Some(since),
            StabilityLevel::Unstable { .. } => None,
        }
    }

    pub(crate) fn const_stable_since(&self, tcx: TyCtxt<'_>) -> Option<Symbol> {
        match self.const_stability(tcx)?.level {
            StabilityLevel::Stable { since, .. } => Some(since),
            StabilityLevel::Unstable { .. } => None,
        }
    }

    pub(crate) fn is_non_exhaustive(&self) -> bool {
        self.attrs.other_attrs.iter().any(|a| a.has_name(sym::non_exhaustive))
    }

    /// Returns a documentation-level item type from the item.
    pub(crate) fn type_(&self) -> ItemType {
        ItemType::from(self)
    }

    pub(crate) fn is_default(&self) -> bool {
        match *self.kind {
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
            asyncness: hir::IsAsync,
        ) -> hir::FnHeader {
            let sig = tcx.fn_sig(def_id);
            let constness =
                if tcx.is_const_fn(def_id) && is_unstable_const_fn(tcx, def_id).is_none() {
                    hir::Constness::Const
                } else {
                    hir::Constness::NotConst
                };
            hir::FnHeader { unsafety: sig.unsafety(), abi: sig.abi(), constness, asyncness }
        }
        let header = match *self.kind {
            ItemKind::ForeignFunctionItem(_) => {
                let abi = tcx.fn_sig(self.item_id.as_def_id().unwrap()).abi();
                hir::FnHeader {
                    unsafety: if abi == Abi::RustIntrinsic {
                        intrinsic_operation_unsafety(self.name.unwrap())
                    } else {
                        hir::Unsafety::Unsafe
                    },
                    abi,
                    constness: hir::Constness::NotConst,
                    asyncness: hir::IsAsync::NotAsync,
                }
            }
            ItemKind::FunctionItem(_) | ItemKind::MethodItem(_, _) => {
                let def_id = self.item_id.as_def_id().unwrap();
                build_fn_header(def_id, tcx, tcx.asyncness(def_id))
            }
            ItemKind::TyMethodItem(_) => {
                build_fn_header(self.item_id.as_def_id().unwrap(), tcx, hir::IsAsync::NotAsync)
            }
            _ => return None,
        };
        Some(header)
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
    TypedefItem(Box<Typedef>),
    OpaqueTyItem(OpaqueTy),
    StaticItem(Static),
    ConstantItem(Constant),
    TraitItem(Box<Trait>),
    TraitAliasItem(TraitAlias),
    ImplItem(Box<Impl>),
    /// A required method in a trait declaration meaning it's only a function signature.
    TyMethodItem(Box<Function>),
    /// A method in a trait impl or a provided method in a trait declaration.
    ///
    /// Compared to [TyMethodItem], it also contains a method body.
    MethodItem(Box<Function>, Option<hir::Defaultness>),
    StructFieldItem(Type),
    VariantItem(Variant),
    /// `fn`s from an extern block
    ForeignFunctionItem(Box<Function>),
    /// `static`s from an extern block
    ForeignStaticItem(Static),
    /// `type`s from an extern block
    ForeignTypeItem,
    MacroItem(Macro),
    ProcMacroItem(ProcMacro),
    PrimitiveItem(PrimitiveType),
    /// A required associated constant in a trait declaration.
    TyAssocConstItem(Type),
    /// An associated associated constant in a trait impl or a provided one in a trait declaration.
    AssocConstItem(Type, ConstantKind),
    /// A required associated type in a trait declaration.
    ///
    /// The bounds may be non-empty if there is a `where` clause.
    TyAssocTypeItem(Box<Generics>, Vec<GenericBound>),
    /// An associated type in a trait impl or a provided one in a trait declaration.
    AssocTypeItem(Box<Typedef>, Vec<GenericBound>),
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
            VariantItem(Variant::Struct(v)) => v.fields.iter(),
            VariantItem(Variant::Tuple(v)) => v.iter(),
            EnumItem(e) => e.variants.iter(),
            TraitItem(t) => t.items.iter(),
            ImplItem(i) => i.items.iter(),
            ModuleItem(m) => m.items.iter(),
            ExternCrateItem { .. }
            | ImportItem(_)
            | FunctionItem(_)
            | TypedefItem(_)
            | OpaqueTyItem(_)
            | StaticItem(_)
            | ConstantItem(_)
            | TraitAliasItem(_)
            | TyMethodItem(_)
            | MethodItem(_, _)
            | StructFieldItem(_)
            | VariantItem(_)
            | ForeignFunctionItem(_)
            | ForeignStaticItem(_)
            | ForeignTypeItem
            | MacroItem(_)
            | ProcMacroItem(_)
            | PrimitiveItem(_)
            | TyAssocConstItem(_)
            | AssocConstItem(_, _)
            | TyAssocTypeItem(..)
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
                | TypedefItem(_)
                | OpaqueTyItem(_)
                | StaticItem(_)
                | ConstantItem(_)
                | TraitAliasItem(_)
                | ForeignFunctionItem(_)
                | ForeignStaticItem(_)
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

pub(crate) trait AttributesExt {
    type AttributeIterator<'a>: Iterator<Item = ast::NestedMetaItem>
    where
        Self: 'a;

    fn lists<'a>(&'a self, name: Symbol) -> Self::AttributeIterator<'a>;

    fn span(&self) -> Option<rustc_span::Span>;

    fn inner_docs(&self) -> bool;

    fn cfg(&self, tcx: TyCtxt<'_>, hidden_cfg: &FxHashSet<Cfg>) -> Option<Arc<Cfg>>;
}

impl AttributesExt for [ast::Attribute] {
    type AttributeIterator<'a> = impl Iterator<Item = ast::NestedMetaItem> + 'a;

    fn lists<'a>(&'a self, name: Symbol) -> Self::AttributeIterator<'a> {
        self.iter()
            .filter(move |attr| attr.has_name(name))
            .filter_map(ast::Attribute::meta_item_list)
            .flatten()
    }

    /// Return the span of the first doc-comment, if it exists.
    fn span(&self) -> Option<rustc_span::Span> {
        self.iter().find(|attr| attr.doc_str().is_some()).map(|attr| attr.span)
    }

    /// Returns whether the first doc-comment is an inner attribute.
    ///
    //// If there are no doc-comments, return true.
    /// FIXME(#78591): Support both inner and outer attributes on the same item.
    fn inner_docs(&self) -> bool {
        self.iter().find(|a| a.doc_str().is_some()).map_or(true, |a| a.style == AttrStyle::Inner)
    }

    fn cfg(&self, tcx: TyCtxt<'_>, hidden_cfg: &FxHashSet<Cfg>) -> Option<Arc<Cfg>> {
        let sess = tcx.sess;
        let doc_cfg_active = tcx.features().doc_cfg;
        let doc_auto_cfg_active = tcx.features().doc_auto_cfg;

        fn single<T: IntoIterator>(it: T) -> Option<T::Item> {
            let mut iter = it.into_iter();
            let item = iter.next()?;
            if iter.next().is_some() {
                return None;
            }
            Some(item)
        }

        let mut cfg = if doc_cfg_active || doc_auto_cfg_active {
            let mut doc_cfg = self
                .iter()
                .filter(|attr| attr.has_name(sym::doc))
                .flat_map(|attr| attr.meta_item_list().unwrap_or_default())
                .filter(|attr| attr.has_name(sym::cfg))
                .peekable();
            if doc_cfg.peek().is_some() && doc_cfg_active {
                doc_cfg
                    .filter_map(|attr| Cfg::parse(attr.meta_item()?).ok())
                    .fold(Cfg::True, |cfg, new_cfg| cfg & new_cfg)
            } else if doc_auto_cfg_active {
                self.iter()
                    .filter(|attr| attr.has_name(sym::cfg))
                    .filter_map(|attr| single(attr.meta_item_list()?))
                    .filter_map(|attr| {
                        Cfg::parse_without(attr.meta_item()?, hidden_cfg).ok().flatten()
                    })
                    .fold(Cfg::True, |cfg, new_cfg| cfg & new_cfg)
            } else {
                Cfg::True
            }
        } else {
            Cfg::True
        };

        for attr in self.iter() {
            // #[doc]
            if attr.doc_str().is_none() && attr.has_name(sym::doc) {
                // #[doc(...)]
                if let Some(list) = attr.meta().as_ref().and_then(|mi| mi.meta_item_list()) {
                    for item in list {
                        // #[doc(hidden)]
                        if !item.has_name(sym::cfg) {
                            continue;
                        }
                        // #[doc(cfg(...))]
                        if let Some(cfg_mi) = item
                            .meta_item()
                            .and_then(|item| rustc_expand::config::parse_cfg(item, sess))
                        {
                            match Cfg::parse(cfg_mi) {
                                Ok(new_cfg) => cfg &= new_cfg,
                                Err(e) => {
                                    sess.span_err(e.span, e.msg);
                                }
                            }
                        }
                    }
                }
            }
        }

        // treat #[target_feature(enable = "feat")] attributes as if they were
        // #[doc(cfg(target_feature = "feat"))] attributes as well
        for attr in self.lists(sym::target_feature) {
            if attr.has_name(sym::enable) {
                if let Some(feat) = attr.value_str() {
                    let meta = attr::mk_name_value_item_str(
                        Ident::with_dummy_span(sym::target_feature),
                        feat,
                        DUMMY_SP,
                    );
                    if let Ok(feat_cfg) = Cfg::parse(&meta) {
                        cfg &= feat_cfg;
                    }
                }
            }
        }

        if cfg == Cfg::True { None } else { Some(Arc::new(cfg)) }
    }
}

pub(crate) trait NestedAttributesExt {
    /// Returns `true` if the attribute list contains a specific `word`
    fn has_word(self, word: Symbol) -> bool
    where
        Self: std::marker::Sized,
    {
        <Self as NestedAttributesExt>::get_word_attr(self, word).is_some()
    }

    /// Returns `Some(attr)` if the attribute list contains 'attr'
    /// corresponding to a specific `word`
    fn get_word_attr(self, word: Symbol) -> Option<ast::NestedMetaItem>;
}

impl<I: Iterator<Item = ast::NestedMetaItem>> NestedAttributesExt for I {
    fn get_word_attr(mut self, word: Symbol) -> Option<ast::NestedMetaItem> {
        self.find(|attr| attr.is_word() && attr.has_name(word))
    }
}

/// A portion of documentation, extracted from a `#[doc]` attribute.
///
/// Each variant contains the line number within the complete doc-comment where the fragment
/// starts, as well as the Span where the corresponding doc comment or attribute is located.
///
/// Included files are kept separate from inline doc comments so that proper line-number
/// information can be given when a doctest fails. Sugared doc comments and "raw" doc comments are
/// kept separate because of issue #42760.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct DocFragment {
    pub(crate) span: rustc_span::Span,
    /// The module this doc-comment came from.
    ///
    /// This allows distinguishing between the original documentation and a pub re-export.
    /// If it is `None`, the item was not re-exported.
    pub(crate) parent_module: Option<DefId>,
    pub(crate) doc: Symbol,
    pub(crate) kind: DocFragmentKind,
    pub(crate) indent: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DocFragmentKind {
    /// A doc fragment created from a `///` or `//!` doc comment.
    SugaredDoc,
    /// A doc fragment created from a "raw" `#[doc=""]` attribute.
    RawDoc,
}

/// The goal of this function is to apply the `DocFragment` transformation that is required when
/// transforming into the final Markdown, which is applying the computed indent to each line in
/// each doc fragment (a `DocFragment` can contain multiple lines in case of `#[doc = ""]`).
///
/// Note: remove the trailing newline where appropriate
fn add_doc_fragment(out: &mut String, frag: &DocFragment) {
    let s = frag.doc.as_str();
    let mut iter = s.lines();
    if s.is_empty() {
        out.push('\n');
        return;
    }
    while let Some(line) = iter.next() {
        if line.chars().any(|c| !c.is_whitespace()) {
            assert!(line.len() >= frag.indent);
            out.push_str(&line[frag.indent..]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
}

/// Collapse a collection of [`DocFragment`]s into one string,
/// handling indentation and newlines as needed.
pub(crate) fn collapse_doc_fragments(doc_strings: &[DocFragment]) -> String {
    let mut acc = String::new();
    for frag in doc_strings {
        add_doc_fragment(&mut acc, frag);
    }
    acc.pop();
    acc
}

/// Removes excess indentation on comments in order for the Markdown
/// to be parsed correctly. This is necessary because the convention for
/// writing documentation is to provide a space between the /// or //! marker
/// and the doc text, but Markdown is whitespace-sensitive. For example,
/// a block of text with four-space indentation is parsed as a code block,
/// so if we didn't unindent comments, these list items
///
/// /// A list:
/// ///
/// ///    - Foo
/// ///    - Bar
///
/// would be parsed as if they were in a code block, which is likely not what the user intended.
fn unindent_doc_fragments(docs: &mut Vec<DocFragment>) {
    // `add` is used in case the most common sugared doc syntax is used ("/// "). The other
    // fragments kind's lines are never starting with a whitespace unless they are using some
    // markdown formatting requiring it. Therefore, if the doc block have a mix between the two,
    // we need to take into account the fact that the minimum indent minus one (to take this
    // whitespace into account).
    //
    // For example:
    //
    // /// hello!
    // #[doc = "another"]
    //
    // In this case, you want "hello! another" and not "hello!  another".
    let add = if docs.windows(2).any(|arr| arr[0].kind != arr[1].kind)
        && docs.iter().any(|d| d.kind == DocFragmentKind::SugaredDoc)
    {
        // In case we have a mix of sugared doc comments and "raw" ones, we want the sugared one to
        // "decide" how much the minimum indent will be.
        1
    } else {
        0
    };

    // `min_indent` is used to know how much whitespaces from the start of each lines must be
    // removed. Example:
    //
    // ///     hello!
    // #[doc = "another"]
    //
    // In here, the `min_indent` is 1 (because non-sugared fragment are always counted with minimum
    // 1 whitespace), meaning that "hello!" will be considered a codeblock because it starts with 4
    // (5 - 1) whitespaces.
    let Some(min_indent) = docs
        .iter()
        .map(|fragment| {
            fragment.doc.as_str().lines().fold(usize::MAX, |min_indent, line| {
                if line.chars().all(|c| c.is_whitespace()) {
                    min_indent
                } else {
                    // Compare against either space or tab, ignoring whether they are
                    // mixed or not.
                    let whitespace = line.chars().take_while(|c| *c == ' ' || *c == '\t').count();
                    cmp::min(min_indent, whitespace)
                        + if fragment.kind == DocFragmentKind::SugaredDoc { 0 } else { add }
                }
            })
        })
        .min()
    else {
        return;
    };

    for fragment in docs {
        if fragment.doc == kw::Empty {
            continue;
        }

        let min_indent = if fragment.kind != DocFragmentKind::SugaredDoc && min_indent > 0 {
            min_indent - add
        } else {
            min_indent
        };

        fragment.indent = min_indent;
    }
}

/// A link that has not yet been rendered.
///
/// This link will be turned into a rendered link by [`Item::links`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ItemLink {
    /// The original link written in the markdown
    pub(crate) link: String,
    /// The link text displayed in the HTML.
    ///
    /// This may not be the same as `link` if there was a disambiguator
    /// in an intra-doc link (e.g. \[`fn@f`\])
    pub(crate) link_text: String,
    pub(crate) did: DefId,
    /// The url fragment to append to the link
    pub(crate) fragment: Option<UrlFragment>,
}

pub struct RenderedLink {
    /// The text the link was original written as.
    ///
    /// This could potentially include disambiguators and backticks.
    pub(crate) original_text: String,
    /// The text to display in the HTML
    pub(crate) new_text: String,
    /// The URL to put in the `href`
    pub(crate) href: String,
}

/// The attributes on an [`Item`], including attributes like `#[derive(...)]` and `#[inline]`,
/// as well as doc comments.
#[derive(Clone, Debug, Default)]
pub(crate) struct Attributes {
    pub(crate) doc_strings: Vec<DocFragment>,
    pub(crate) other_attrs: ast::AttrVec,
}

impl Attributes {
    pub(crate) fn lists(&self, name: Symbol) -> impl Iterator<Item = ast::NestedMetaItem> + '_ {
        self.other_attrs.lists(name)
    }

    pub(crate) fn has_doc_flag(&self, flag: Symbol) -> bool {
        for attr in &self.other_attrs {
            if !attr.has_name(sym::doc) {
                continue;
            }

            if let Some(items) = attr.meta_item_list() {
                if items.iter().filter_map(|i| i.meta_item()).any(|it| it.has_name(flag)) {
                    return true;
                }
            }
        }

        false
    }

    pub(crate) fn from_ast(attrs: &[ast::Attribute]) -> Attributes {
        Attributes::from_ast_iter(attrs.iter().map(|attr| (attr, None)), false)
    }

    pub(crate) fn from_ast_with_additional(
        attrs: &[ast::Attribute],
        (additional_attrs, def_id): (&[ast::Attribute], DefId),
    ) -> Attributes {
        // Additional documentation should be shown before the original documentation.
        let attrs1 = additional_attrs.iter().map(|attr| (attr, Some(def_id)));
        let attrs2 = attrs.iter().map(|attr| (attr, None));
        Attributes::from_ast_iter(attrs1.chain(attrs2), false)
    }

    pub(crate) fn from_ast_iter<'a>(
        attrs: impl Iterator<Item = (&'a ast::Attribute, Option<DefId>)>,
        doc_only: bool,
    ) -> Attributes {
        let mut doc_strings = Vec::new();
        let mut other_attrs = ast::AttrVec::new();
        for (attr, parent_module) in attrs {
            if let Some((doc_str, comment_kind)) = attr.doc_str_and_comment_kind() {
                trace!("got doc_str={doc_str:?}");
                let doc = beautify_doc_string(doc_str, comment_kind);
                let kind = if attr.is_doc_comment() {
                    DocFragmentKind::SugaredDoc
                } else {
                    DocFragmentKind::RawDoc
                };
                let fragment = DocFragment { span: attr.span, doc, kind, parent_module, indent: 0 };
                doc_strings.push(fragment);
            } else if !doc_only {
                other_attrs.push(attr.clone());
            }
        }

        unindent_doc_fragments(&mut doc_strings);

        Attributes { doc_strings, other_attrs }
    }

    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    pub(crate) fn doc_value(&self) -> Option<String> {
        let mut iter = self.doc_strings.iter();

        let ori = iter.next()?;
        let mut out = String::new();
        add_doc_fragment(&mut out, ori);
        for new_frag in iter {
            add_doc_fragment(&mut out, new_frag);
        }
        out.pop();
        if out.is_empty() { None } else { Some(out) }
    }

    /// Return the doc-comments on this item, grouped by the module they came from.
    /// The module can be different if this is a re-export with added documentation.
    ///
    /// The last newline is not trimmed so the produced strings are reusable between
    /// early and late doc link resolution regardless of their position.
    pub(crate) fn prepare_to_doc_link_resolution(&self) -> FxHashMap<Option<DefId>, String> {
        let mut res = FxHashMap::default();
        for fragment in &self.doc_strings {
            let out_str = res.entry(fragment.parent_module).or_default();
            add_doc_fragment(out_str, fragment);
        }
        res
    }

    /// Finds all `doc` attributes as NameValues and returns their corresponding values, joined
    /// with newlines.
    pub(crate) fn collapsed_doc_value(&self) -> Option<String> {
        if self.doc_strings.is_empty() {
            None
        } else {
            Some(collapse_doc_fragments(&self.doc_strings))
        }
    }

    pub(crate) fn get_doc_aliases(&self) -> Box<[Symbol]> {
        let mut aliases = FxHashSet::default();

        for attr in self.other_attrs.lists(sym::doc).filter(|a| a.has_name(sym::alias)) {
            if let Some(values) = attr.meta_item_list() {
                for l in values {
                    match l.literal().unwrap().kind {
                        ast::LitKind::Str(s, _) => {
                            aliases.insert(s);
                        }
                        _ => unreachable!(),
                    }
                }
            } else {
                aliases.insert(attr.value_str().unwrap());
            }
        }
        aliases.into_iter().collect::<Vec<_>>().into()
    }
}

impl PartialEq for Attributes {
    fn eq(&self, rhs: &Self) -> bool {
        self.doc_strings == rhs.doc_strings
            && self
                .other_attrs
                .iter()
                .map(|attr| attr.id)
                .eq(rhs.other_attrs.iter().map(|attr| attr.id))
    }
}

impl Eq for Attributes {}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum GenericBound {
    TraitBound(PolyTrait, hir::TraitBoundModifier),
    Outlives(Lifetime),
}

impl GenericBound {
    pub(crate) fn maybe_sized(cx: &mut DocContext<'_>) -> GenericBound {
        let did = cx.tcx.require_lang_item(LangItem::Sized, None);
        let empty = cx.tcx.intern_substs(&[]);
        let path = external_path(cx, did, false, ThinVec::new(), empty);
        inline::record_extern_fqn(cx, did, ItemType::Trait);
        GenericBound::TraitBound(
            PolyTrait { trait_: path, generic_params: Vec::new() },
            hir::TraitBoundModifier::Maybe,
        )
    }

    pub(crate) fn is_sized_bound(&self, cx: &DocContext<'_>) -> bool {
        use rustc_hir::TraitBoundModifier as TBM;
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, TBM::None) = *self {
            if Some(trait_.def_id()) == cx.tcx.lang_items().sized_trait() {
                return true;
            }
        }
        false
    }

    pub(crate) fn get_poly_trait(&self) -> Option<PolyTrait> {
        if let GenericBound::TraitBound(ref p, _) = *self {
            return Some(p.clone());
        }
        None
    }

    pub(crate) fn get_trait_path(&self) -> Option<Path> {
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, _) = *self {
            Some(trait_.clone())
        } else {
            None
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct Lifetime(pub Symbol);

impl Lifetime {
    pub(crate) fn statik() -> Lifetime {
        Lifetime(kw::StaticLifetime)
    }

    pub(crate) fn elided() -> Lifetime {
        Lifetime(kw::UnderscoreLifetime)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum WherePredicate {
    BoundPredicate { ty: Type, bounds: Vec<GenericBound>, bound_params: Vec<Lifetime> },
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
    Lifetime { outlives: Vec<Lifetime> },
    Type { did: DefId, bounds: Vec<GenericBound>, default: Option<Box<Type>>, synthetic: bool },
    Const { did: DefId, ty: Box<Type>, default: Option<Box<String>> },
}

impl GenericParamDefKind {
    pub(crate) fn is_type(&self) -> bool {
        matches!(self, GenericParamDefKind::Type { .. })
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct GenericParamDef {
    pub(crate) name: Symbol,
    pub(crate) kind: GenericParamDefKind,
}

impl GenericParamDef {
    pub(crate) fn is_synthetic_type_param(&self) -> bool {
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
#[derive(Clone, Debug, Default)]
pub(crate) struct Generics {
    pub(crate) params: Vec<GenericParamDef>,
    pub(crate) where_predicates: Vec<WherePredicate>,
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
    pub(crate) inputs: Arguments,
    pub(crate) output: FnRetTy,
    pub(crate) c_variadic: bool,
}

impl FnDecl {
    pub(crate) fn self_type(&self) -> Option<SelfTy> {
        self.inputs.values.get(0).and_then(|v| v.to_self())
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
    pub(crate) fn sugared_async_return_type(&self) -> FnRetTy {
        match &self.output {
            FnRetTy::Return(Type::ImplTrait(bounds)) => match &bounds[0] {
                GenericBound::TraitBound(PolyTrait { trait_, .. }, ..) => {
                    let bindings = trait_.bindings().unwrap();
                    let ret_ty = bindings[0].term();
                    let ty = ret_ty.ty().expect("Unexpected constant return term");
                    FnRetTy::Return(ty.clone())
                }
                _ => panic!("unexpected desugaring of async function"),
            },
            _ => panic!("unexpected desugaring of async function"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct Arguments {
    pub(crate) values: Vec<Argument>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct Argument {
    pub(crate) type_: Type,
    pub(crate) name: Symbol,
    /// This field is used to represent "const" arguments from the `rustc_legacy_const_generics`
    /// feature. More information in <https://github.com/rust-lang/rust/issues/83167>.
    pub(crate) is_const: bool,
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum SelfTy {
    SelfValue,
    SelfBorrowed(Option<Lifetime>, Mutability),
    SelfExplicit(Type),
}

impl Argument {
    pub(crate) fn to_self(&self) -> Option<SelfTy> {
        if self.name != kw::SelfLower {
            return None;
        }
        if self.type_.is_self_type() {
            return Some(SelfValue);
        }
        match self.type_ {
            BorrowedRef { ref lifetime, mutability, ref type_ } if type_.is_self_type() => {
                Some(SelfBorrowed(lifetime.clone(), mutability))
            }
            _ => Some(SelfExplicit(self.type_.clone())),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum FnRetTy {
    Return(Type),
    DefaultReturn,
}

impl FnRetTy {
    pub(crate) fn as_return(&self) -> Option<&Type> {
        match self {
            Return(ret) => Some(ret),
            DefaultReturn => None,
        }
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
    pub(crate) fn unsafety(&self, tcx: TyCtxt<'_>) -> hir::Unsafety {
        tcx.trait_def(self.def_id).unsafety
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
    Path { path: Path },
    /// A `dyn Trait` object: `dyn for<'a> Trait<'a> + Send + 'static`
    DynTrait(Vec<PolyTrait>, Option<Lifetime>),
    /// A type parameter.
    Generic(Symbol),
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
    Array(Box<Type>, String),
    /// A raw pointer type: `*const i32`, `*mut i32`
    RawPointer(Mutability, Box<Type>),
    /// A reference type: `&i32`, `&'a mut Foo`
    BorrowedRef { lifetime: Option<Lifetime>, mutability: Mutability, type_: Box<Type> },

    /// A qualified path to an associated item: `<Type as Trait>::Name`
    QPath(Box<QPathData>),

    /// A type that is inferred: `_`
    Infer,

    /// An `impl Trait`: `impl TraitA + TraitB + ...`
    ImplTrait(Vec<GenericBound>),
}

impl Type {
    /// When comparing types for equality, it can help to ignore `&` wrapping.
    pub(crate) fn without_borrowed_ref(&self) -> &Type {
        let mut result = self;
        while let Type::BorrowedRef { type_, .. } = result {
            result = &*type_;
        }
        result
    }

    /// Check if two types are "potentially the same".
    /// This is different from `Eq`, because it knows that things like
    /// `Placeholder` are possible matches for everything.
    pub(crate) fn is_same(&self, other: &Self, cache: &Cache) -> bool {
        match (self, other) {
            // Recursive cases.
            (Type::Tuple(a), Type::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.is_same(b, cache))
            }
            (Type::Slice(a), Type::Slice(b)) => a.is_same(b, cache),
            (Type::Array(a, al), Type::Array(b, bl)) => al == bl && a.is_same(b, cache),
            (Type::RawPointer(mutability, type_), Type::RawPointer(b_mutability, b_type_)) => {
                mutability == b_mutability && type_.is_same(b_type_, cache)
            }
            (
                Type::BorrowedRef { mutability, type_, .. },
                Type::BorrowedRef { mutability: b_mutability, type_: b_type_, .. },
            ) => mutability == b_mutability && type_.is_same(b_type_, cache),
            // Placeholders and generics are equal to all other types.
            (Type::Infer, _) | (_, Type::Infer) => true,
            (Type::Generic(_), _) | (_, Type::Generic(_)) => true,
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

    /// Checks if this is a `T::Name` path for an associated type.
    pub(crate) fn is_assoc_ty(&self) -> bool {
        match self {
            Type::Path { path, .. } => path.is_assoc_ty(),
            _ => false,
        }
    }

    pub(crate) fn is_self_type(&self) -> bool {
        match *self {
            Generic(name) => name == kw::SelfUpper,
            _ => false,
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

    pub(crate) fn is_impl_trait(&self) -> bool {
        matches!(self, Type::ImplTrait(_))
    }

    pub(crate) fn projection(&self) -> Option<(&Type, DefId, PathSegment)> {
        if let QPath(box QPathData { self_type, trait_, assoc, .. }) = self {
            Some((self_type, trait_.def_id(), assoc.clone()))
        } else {
            None
        }
    }

    fn inner_def_id(&self, cache: Option<&Cache>) -> Option<DefId> {
        let t: PrimitiveType = match *self {
            Type::Path { ref path } => return Some(path.def_id()),
            DynTrait(ref bounds, _) => return Some(bounds[0].trait_.def_id()),
            Primitive(p) => return cache.and_then(|c| c.primitive_locations.get(&p).cloned()),
            BorrowedRef { type_: box Generic(..), .. } => PrimitiveType::Reference,
            BorrowedRef { ref type_, .. } => return type_.inner_def_id(cache),
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
            RawPointer(..) => PrimitiveType::RawPointer,
            QPath(box QPathData { ref self_type, .. }) => return self_type.inner_def_id(cache),
            Generic(_) | Infer | ImplTrait(_) => return None,
        };
        cache.and_then(|c| Primitive(t).def_id(c))
    }

    /// Use this method to get the [DefId] of a [clean] AST node, including [PrimitiveType]s.
    ///
    /// [clean]: crate::clean
    pub(crate) fn def_id(&self, cache: &Cache) -> Option<DefId> {
        self.inner_def_id(Some(cache))
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct QPathData {
    pub assoc: PathSegment,
    pub self_type: Type,
    /// FIXME: compute this field on demand.
    pub should_show_cast: bool,
    pub trait_: Path,
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
    F32,
    F64,
    Char,
    Bool,
    Str,
    Slice,
    Array,
    Tuple,
    Unit,
    RawPointer,
    Reference,
    Fn,
    Never,
}

type SimplifiedTypes = FxHashMap<PrimitiveType, ArrayVec<SimplifiedType, 3>>;
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
            hir::PrimTy::Float(FloatTy::F32) => PrimitiveType::F32,
            hir::PrimTy::Float(FloatTy::F64) => PrimitiveType::F64,
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
            sym::f32 => Some(PrimitiveType::F32),
            sym::f64 => Some(PrimitiveType::F64),
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
        use ty::fast_reject::SimplifiedTypeGen::*;
        use ty::{FloatTy, IntTy, UintTy};
        use PrimitiveType::*;
        static CELL: OnceCell<SimplifiedTypes> = OnceCell::new();

        let single = |x| iter::once(x).collect();
        CELL.get_or_init(move || {
            map! {
                Isize => single(IntSimplifiedType(IntTy::Isize)),
                I8 => single(IntSimplifiedType(IntTy::I8)),
                I16 => single(IntSimplifiedType(IntTy::I16)),
                I32 => single(IntSimplifiedType(IntTy::I32)),
                I64 => single(IntSimplifiedType(IntTy::I64)),
                I128 => single(IntSimplifiedType(IntTy::I128)),
                Usize => single(UintSimplifiedType(UintTy::Usize)),
                U8 => single(UintSimplifiedType(UintTy::U8)),
                U16 => single(UintSimplifiedType(UintTy::U16)),
                U32 => single(UintSimplifiedType(UintTy::U32)),
                U64 => single(UintSimplifiedType(UintTy::U64)),
                U128 => single(UintSimplifiedType(UintTy::U128)),
                F32 => single(FloatSimplifiedType(FloatTy::F32)),
                F64 => single(FloatSimplifiedType(FloatTy::F64)),
                Str => single(StrSimplifiedType),
                Bool => single(BoolSimplifiedType),
                Char => single(CharSimplifiedType),
                Array => single(ArraySimplifiedType),
                Slice => single(SliceSimplifiedType),
                // FIXME: If we ever add an inherent impl for tuples
                // with different lengths, they won't show in rustdoc.
                //
                // Either manually update this arrayvec at this point
                // or start with a more complex refactoring.
                Tuple => [TupleSimplifiedType(1), TupleSimplifiedType(2), TupleSimplifiedType(3)].into(),
                Unit => single(TupleSimplifiedType(0)),
                RawPointer => [PtrSimplifiedType(Mutability::Not), PtrSimplifiedType(Mutability::Mut)].into_iter().collect(),
                Reference => [RefSimplifiedType(Mutability::Not), RefSimplifiedType(Mutability::Mut)].into_iter().collect(),
                // FIXME: This will be wrong if we ever add inherent impls
                // for function pointers.
                Fn => single(FunctionSimplifiedType(1)),
                Never => single(NeverSimplifiedType),
            }
        })
    }

    pub(crate) fn impls<'tcx>(&self, tcx: TyCtxt<'tcx>) -> impl Iterator<Item = DefId> + 'tcx {
        Self::simplified_types()
            .get(self)
            .into_iter()
            .flatten()
            .flat_map(move |&simp| tcx.incoherent_impls(simp))
            .copied()
    }

    pub(crate) fn all_impls(tcx: TyCtxt<'_>) -> impl Iterator<Item = DefId> + '_ {
        Self::simplified_types()
            .values()
            .flatten()
            .flat_map(move |&simp| tcx.incoherent_impls(simp))
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
            F32 => sym::f32,
            F64 => sym::f64,
            Str => sym::str,
            Bool => sym::bool,
            Char => sym::char,
            Array => sym::array,
            Slice => sym::slice,
            Tuple => sym::tuple,
            Unit => sym::unit,
            RawPointer => sym::pointer,
            Reference => sym::reference,
            Fn => kw::Fn,
            Never => sym::never,
        }
    }

    /// Returns the DefId of the module with `doc(primitive)` for this primitive type.
    /// Panics if there is no such module.
    ///
    /// This gives precedence to primitives defined in the current crate, and deprioritizes primitives defined in `core`,
    /// but otherwise, if multiple crates define the same primitive, there is no guarantee of which will be picked.
    /// In particular, if a crate depends on both `std` and another crate that also defines `doc(primitive)`, then
    /// it's entirely random whether `std` or the other crate is picked. (no_std crates are usually fine unless multiple dependencies define a primitive.)
    pub(crate) fn primitive_locations(tcx: TyCtxt<'_>) -> &FxHashMap<PrimitiveType, DefId> {
        static PRIMITIVE_LOCATIONS: OnceCell<FxHashMap<PrimitiveType, DefId>> = OnceCell::new();
        PRIMITIVE_LOCATIONS.get_or_init(|| {
            let mut primitive_locations = FxHashMap::default();
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
            ast::FloatTy::F32 => PrimitiveType::F32,
            ast::FloatTy::F64 => PrimitiveType::F64,
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
            ty::FloatTy::F32 => PrimitiveType::F32,
            ty::FloatTy::F64 => PrimitiveType::F64,
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

#[derive(Copy, Clone, Debug)]
pub(crate) enum Visibility {
    /// `pub`
    Public,
    /// Visibility inherited from parent.
    ///
    /// For example, this is the visibility of private items and of enum variants.
    Inherited,
    /// `pub(crate)`, `pub(super)`, or `pub(in path::to::somewhere)`
    Restricted(DefId),
}

impl Visibility {
    pub(crate) fn is_public(&self) -> bool {
        matches!(self, Visibility::Public)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Struct {
    pub(crate) struct_type: CtorKind,
    pub(crate) generics: Generics,
    pub(crate) fields: Vec<Item>,
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
    pub(crate) struct_type: CtorKind,
    pub(crate) fields: Vec<Item>,
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
pub(crate) enum Variant {
    CLike(Option<Discriminant>),
    Tuple(Vec<Item>),
    Struct(VariantStruct),
}

impl Variant {
    pub(crate) fn has_stripped_entries(&self) -> Option<bool> {
        match *self {
            Self::Struct(ref struct_) => Some(struct_.has_stripped_entries()),
            Self::CLike(..) | Self::Tuple(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Discriminant {
    // In the case of cross crate re-exports, we don't have the nessesary information
    // to reconstruct the expression of the discriminant, only the value.
    pub(super) expr: Option<BodyId>,
    pub(super) value: DefId,
}

impl Discriminant {
    /// Will be `None` in the case of cross-crate reexports, and may be
    /// simplified
    pub(crate) fn expr(&self, tcx: TyCtxt<'_>) -> Option<String> {
        self.expr.map(|body| print_const_expr(tcx, body))
    }
    /// Will always be a machine readable number, without underscores or suffixes.
    pub(crate) fn value(&self, tcx: TyCtxt<'_>) -> String {
        print_evaluated_const(tcx, self.value, false).unwrap()
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
    pub(crate) segments: Vec<PathSegment>,
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
            Res::SelfTy { .. } if self.segments.len() != 1 => true,
            Res::Def(DefKind::TyParam, _) if self.segments.len() != 1 => true,
            Res::Def(DefKind::AssocTy, _) => true,
            _ => false,
        }
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

    pub(crate) fn bindings(&self) -> Option<&[TypeBinding]> {
        self.segments.last().and_then(|seg| {
            if let GenericArgs::AngleBracketed { ref bindings, .. } = seg.args {
                Some(&**bindings)
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
    Const(Box<Constant>),
    Infer,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum GenericArgs {
    AngleBracketed { args: Box<[GenericArg]>, bindings: ThinVec<TypeBinding> },
    Parenthesized { inputs: Box<[Type]>, output: Option<Box<Type>> },
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct PathSegment {
    pub(crate) name: Symbol,
    pub(crate) args: GenericArgs,
}

#[derive(Clone, Debug)]
pub(crate) struct Typedef {
    pub(crate) type_: Type,
    pub(crate) generics: Generics,
    /// `type_` can come from either the HIR or from metadata. If it comes from HIR, it may be a type
    /// alias instead of the final type. This will always have the final type, regardless of whether
    /// `type_` came from HIR or from metadata.
    ///
    /// If `item_type.is_none()`, `type_` is guaranteed to come from metadata (and therefore hold the
    /// final type).
    pub(crate) item_type: Option<Type>,
}

#[derive(Clone, Debug)]
pub(crate) struct OpaqueTy {
    pub(crate) bounds: Vec<GenericBound>,
    pub(crate) generics: Generics,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct BareFunctionDecl {
    pub(crate) unsafety: hir::Unsafety,
    pub(crate) generic_params: Vec<GenericParamDef>,
    pub(crate) decl: FnDecl,
    pub(crate) abi: Abi,
}

#[derive(Clone, Debug)]
pub(crate) struct Static {
    pub(crate) type_: Type,
    pub(crate) mutability: Mutability,
    pub(crate) expr: Option<BodyId>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Constant {
    pub(crate) type_: Type,
    pub(crate) kind: ConstantKind,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Term {
    Type(Type),
    Constant(Constant),
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
    TyConst { expr: String },
    /// A constant (expression) that's not an item or associated item. These are usually found
    /// nested inside types (e.g., array lengths) or expressions (e.g., repeat counts), and also
    /// used to define explicit discriminant values for enum variants.
    Anonymous { body: BodyId },
    /// A constant from a different crate.
    Extern { def_id: DefId },
    /// `const FOO: u32 = ...;`
    Local { def_id: DefId, body: BodyId },
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
            ConstantKind::TyConst { ref expr } => expr.clone(),
            ConstantKind::Extern { def_id } => print_inlined_const(tcx, def_id),
            ConstantKind::Local { body, .. } | ConstantKind::Anonymous { body } => {
                print_const_expr(tcx, body)
            }
        }
    }

    pub(crate) fn value(&self, tcx: TyCtxt<'_>) -> Option<String> {
        match *self {
            ConstantKind::TyConst { .. } | ConstantKind::Anonymous { .. } => None,
            ConstantKind::Extern { def_id } | ConstantKind::Local { def_id, .. } => {
                print_evaluated_const(tcx, def_id, true)
            }
        }
    }

    pub(crate) fn is_literal(&self, tcx: TyCtxt<'_>) -> bool {
        match *self {
            ConstantKind::TyConst { .. } => false,
            ConstantKind::Extern { def_id } => def_id.as_local().map_or(false, |def_id| {
                is_literal_expr(tcx, tcx.hir().local_def_id_to_hir_id(def_id))
            }),
            ConstantKind::Local { body, .. } | ConstantKind::Anonymous { body } => {
                is_literal_expr(tcx, body.hir_id)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Impl {
    pub(crate) unsafety: hir::Unsafety,
    pub(crate) generics: Generics,
    pub(crate) trait_: Option<Path>,
    pub(crate) for_: Type,
    pub(crate) items: Vec<Item>,
    pub(crate) polarity: ty::ImplPolarity,
    pub(crate) kind: ImplKind,
}

impl Impl {
    pub(crate) fn provided_trait_methods(&self, tcx: TyCtxt<'_>) -> FxHashSet<Symbol> {
        self.trait_
            .as_ref()
            .map(|t| t.def_id())
            .map(|did| tcx.provided_trait_methods(did).map(|meth| meth.name).collect())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ImplKind {
    Normal,
    Auto,
    FakeVaradic,
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
        matches!(self, ImplKind::FakeVaradic)
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
}

#[derive(Clone, Debug)]
pub(crate) struct ProcMacro {
    pub(crate) kind: MacroKind,
    pub(crate) helpers: Vec<Symbol>,
}

/// An type binding on an associated type (e.g., `A = Bar` in `Foo<A = Bar>` or
/// `A: Send + Sync` in `Foo<A: Send + Sync>`).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct TypeBinding {
    pub(crate) assoc: PathSegment,
    pub(crate) kind: TypeBindingKind,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) enum TypeBindingKind {
    Equality { term: Term },
    Constraint { bounds: Vec<GenericBound> },
}

impl TypeBinding {
    pub(crate) fn term(&self) -> &Term {
        match self.kind {
            TypeBindingKind::Equality { ref term } => term,
            _ => panic!("expected equality type binding for parenthesized generic args"),
        }
    }
}

/// The type, lifetime, or constant that a private type alias's parameter should be
/// replaced with when expanding a use of that type alias.
///
/// For example:
///
/// ```
/// type PrivAlias<T> = Vec<T>;
///
/// pub fn public_fn() -> PrivAlias<i32> { vec![] }
/// ```
///
/// `public_fn`'s docs will show it as returning `Vec<i32>`, since `PrivAlias` is private.
/// [`SubstParam`] is used to record that `T` should be mapped to `i32`.
pub(crate) enum SubstParam {
    Type(Type),
    Lifetime(Lifetime),
    Constant(Constant),
}

impl SubstParam {
    pub(crate) fn as_ty(&self) -> Option<&Type> {
        if let Self::Type(ty) = self { Some(ty) } else { None }
    }

    pub(crate) fn as_lt(&self) -> Option<&Lifetime> {
        if let Self::Lifetime(lt) = self { Some(lt) } else { None }
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // These are in alphabetical order, which is easy to maintain.
    static_assert_size!(Crate, 72); // frequently moved by-value
    static_assert_size!(DocFragment, 32);
    static_assert_size!(GenericArg, 64);
    static_assert_size!(GenericArgs, 32);
    static_assert_size!(GenericParamDef, 56);
    static_assert_size!(Item, 56);
    static_assert_size!(ItemKind, 96);
    static_assert_size!(PathSegment, 40);
    static_assert_size!(Type, 56);
}
