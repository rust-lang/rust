use std::cell::RefCell;
use std::default::Default;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::lazy::SyncOnceCell as OnceCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::{slice, vec};

use arrayvec::ArrayVec;

use rustc_ast::attr;
use rustc_ast::util::comments::beautify_doc_string;
use rustc_ast::{self as ast, AttrStyle};
use rustc_attr::{ConstStability, Deprecation, Stability, StabilityLevel};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{BodyId, Mutability};
use rustc_index::vec::IndexVec;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::Session;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::{kw, sym, Ident, Symbol, SymbolStr};
use rustc_span::{self, FileName, Loc};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi::Abi;

use crate::clean::cfg::Cfg;
use crate::clean::external_path;
use crate::clean::inline::{self, print_inlined_const};
use crate::clean::types::Type::{QPath, ResolvedPath};
use crate::clean::utils::{is_literal_expr, print_const_expr, print_evaluated_const};
use crate::clean::Clean;
use crate::core::DocContext;
use crate::formats::cache::Cache;
use crate::formats::item_type::ItemType;
use crate::html::render::cache::ExternalLocation;
use crate::html::render::Context;

use self::FnRetTy::*;
use self::ItemKind::*;
use self::SelfTy::*;
use self::Type::*;

crate type ItemIdSet = FxHashSet<ItemId>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
crate enum ItemId {
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
    crate fn is_local(self) -> bool {
        match self {
            ItemId::Auto { for_: id, .. }
            | ItemId::Blanket { for_: id, .. }
            | ItemId::DefId(id) => id.is_local(),
            ItemId::Primitive(_, krate) => krate == LOCAL_CRATE,
        }
    }

    #[inline]
    #[track_caller]
    crate fn expect_def_id(self) -> DefId {
        self.as_def_id()
            .unwrap_or_else(|| panic!("ItemId::expect_def_id: `{:?}` isn't a DefId", self))
    }

    #[inline]
    crate fn as_def_id(self) -> Option<DefId> {
        match self {
            ItemId::DefId(id) => Some(id),
            _ => None,
        }
    }

    #[inline]
    crate fn krate(self) -> CrateNum {
        match self {
            ItemId::Auto { for_: id, .. }
            | ItemId::Blanket { for_: id, .. }
            | ItemId::DefId(id) => id.krate,
            ItemId::Primitive(_, krate) => krate,
        }
    }

    #[inline]
    crate fn index(self) -> Option<DefIndex> {
        match self {
            ItemId::DefId(id) => Some(id.index),
            _ => None,
        }
    }
}

impl From<DefId> for ItemId {
    fn from(id: DefId) -> Self {
        Self::DefId(id)
    }
}

#[derive(Clone, Debug)]
crate struct Crate {
    crate name: Symbol,
    crate src: FileName,
    crate module: Item,
    crate externs: Vec<ExternalCrate>,
    crate primitives: ThinVec<(DefId, PrimitiveType)>,
    /// Only here so that they can be filtered through the rustdoc passes.
    crate external_traits: Rc<RefCell<FxHashMap<DefId, TraitWithExtraInfo>>>,
    crate collapsed: bool,
}

/// This struct is used to wrap additional information added by rustdoc on a `trait` item.
#[derive(Clone, Debug)]
crate struct TraitWithExtraInfo {
    crate trait_: Trait,
    crate is_notable: bool,
}

#[derive(Copy, Clone, Debug)]
crate struct ExternalCrate {
    crate crate_num: CrateNum,
}

impl ExternalCrate {
    #[inline]
    crate fn def_id(&self) -> DefId {
        DefId { krate: self.crate_num, index: CRATE_DEF_INDEX }
    }

    crate fn src(&self, tcx: TyCtxt<'_>) -> FileName {
        let krate_span = tcx.def_span(self.def_id());
        tcx.sess.source_map().span_to_filename(krate_span)
    }

    crate fn name(&self, tcx: TyCtxt<'_>) -> Symbol {
        tcx.crate_name(self.crate_num)
    }

    crate fn src_root(&self, tcx: TyCtxt<'_>) -> PathBuf {
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
    crate fn location(
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
        let local_location = dst.join(&*self.name(tcx).as_str());
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
        let did = DefId { krate: self.crate_num, index: CRATE_DEF_INDEX };
        tcx.get_attrs(did)
            .lists(sym::doc)
            .filter(|a| a.has_name(sym::html_root_url))
            .filter_map(|a| a.value_str())
            .map(to_remote)
            .next()
            .or(extern_url.map(to_remote)) // NOTE: only matters if `extern_url_takes_precedence` is false
            .unwrap_or(Unknown) // Well, at least we tried.
    }

    crate fn keywords(&self, tcx: TyCtxt<'_>) -> ThinVec<(DefId, Symbol)> {
        let root = self.def_id();

        let as_keyword = |res: Res<!>| {
            if let Res::Def(DefKind::Mod, def_id) = res {
                let attrs = tcx.get_attrs(def_id);
                let mut keyword = None;
                for attr in attrs.lists(sym::doc) {
                    if attr.has_name(sym::keyword) {
                        if let Some(v) = attr.value_str() {
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
                        hir::ItemKind::Use(ref path, hir::UseKind::Single)
                            if item.vis.node.is_pub() =>
                        {
                            as_keyword(path.res.expect_non_local())
                                .map(|(_, prim)| (id.def_id.to_def_id(), prim))
                        }
                        _ => None,
                    }
                })
                .collect()
        } else {
            tcx.item_children(root).iter().map(|item| item.res).filter_map(as_keyword).collect()
        }
    }

    crate fn primitives(&self, tcx: TyCtxt<'_>) -> ThinVec<(DefId, PrimitiveType)> {
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
                let attrs = tcx.get_attrs(def_id);
                let mut prim = None;
                for attr in attrs.lists(sym::doc) {
                    if let Some(v) = attr.value_str() {
                        if attr.has_name(sym::primitive) {
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
                        hir::ItemKind::Use(ref path, hir::UseKind::Single)
                            if item.vis.node.is_pub() =>
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
            tcx.item_children(root).iter().map(|item| item.res).filter_map(as_primitive).collect()
        }
    }
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[derive(Clone, Debug)]
crate struct Item {
    /// The name of this item.
    /// Optional because not every item has a name, e.g. impls.
    crate name: Option<Symbol>,
    crate attrs: Box<Attributes>,
    crate visibility: Visibility,
    /// Information about this item that is specific to what kind of item it is.
    /// E.g., struct vs enum vs function.
    crate kind: Box<ItemKind>,
    crate def_id: ItemId,

    crate cfg: Option<Arc<Cfg>>,
}

// `Item` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(Item, 56);

crate fn rustc_span(def_id: DefId, tcx: TyCtxt<'_>) -> Span {
    Span::new(def_id.as_local().map_or_else(
        || tcx.def_span(def_id),
        |local| {
            let hir = tcx.hir();
            hir.span_with_body(hir.local_def_id_to_hir_id(local))
        },
    ))
}

impl Item {
    crate fn stability<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Option<&'tcx Stability> {
        self.def_id.as_def_id().and_then(|did| tcx.lookup_stability(did))
    }

    crate fn const_stability<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Option<&'tcx ConstStability> {
        self.def_id.as_def_id().and_then(|did| tcx.lookup_const_stability(did))
    }

    crate fn deprecation(&self, tcx: TyCtxt<'_>) -> Option<Deprecation> {
        self.def_id.as_def_id().and_then(|did| tcx.lookup_deprecation(did))
    }

    crate fn inner_docs(&self, tcx: TyCtxt<'_>) -> bool {
        self.def_id.as_def_id().map(|did| tcx.get_attrs(did).inner_docs()).unwrap_or(false)
    }

    crate fn span(&self, tcx: TyCtxt<'_>) -> Span {
        let kind = match &*self.kind {
            ItemKind::StrippedItem(k) => k,
            _ => &*self.kind,
        };
        if let ItemKind::ModuleItem(Module { span, .. }) | ItemKind::ImplItem(Impl { span, .. }) =
            kind
        {
            *span
        } else {
            self.def_id.as_def_id().map(|did| rustc_span(did, tcx)).unwrap_or_else(|| Span::dummy())
        }
    }

    crate fn attr_span(&self, tcx: TyCtxt<'_>) -> rustc_span::Span {
        crate::passes::span_of_attrs(&self.attrs).unwrap_or_else(|| self.span(tcx).inner())
    }

    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    crate fn doc_value(&self) -> Option<String> {
        self.attrs.doc_value()
    }

    /// Convenience wrapper around [`Self::from_def_id_and_parts`] which converts
    /// `hir_id` to a [`DefId`]
    pub fn from_hir_id_and_parts(
        hir_id: hir::HirId,
        name: Option<Symbol>,
        kind: ItemKind,
        cx: &mut DocContext<'_>,
    ) -> Item {
        Item::from_def_id_and_parts(cx.tcx.hir().local_def_id(hir_id).to_def_id(), name, kind, cx)
    }

    pub fn from_def_id_and_parts(
        def_id: DefId,
        name: Option<Symbol>,
        kind: ItemKind,
        cx: &mut DocContext<'_>,
    ) -> Item {
        let ast_attrs = cx.tcx.get_attrs(def_id);

        Self::from_def_id_and_attrs_and_parts(
            def_id,
            name,
            kind,
            box ast_attrs.clean(cx),
            cx,
            ast_attrs.cfg(cx.tcx, &cx.cache.hidden_cfg),
        )
    }

    pub fn from_def_id_and_attrs_and_parts(
        def_id: DefId,
        name: Option<Symbol>,
        kind: ItemKind,
        attrs: Box<Attributes>,
        cx: &mut DocContext<'_>,
        cfg: Option<Arc<Cfg>>,
    ) -> Item {
        trace!("name={:?}, def_id={:?}", name, def_id);

        Item {
            def_id: def_id.into(),
            kind: box kind,
            name,
            attrs,
            visibility: cx.tcx.visibility(def_id).clean(cx),
            cfg,
        }
    }

    /// Finds all `doc` attributes as NameValues and returns their corresponding values, joined
    /// with newlines.
    crate fn collapsed_doc_value(&self) -> Option<String> {
        self.attrs.collapsed_doc_value()
    }

    crate fn links(&self, cx: &Context<'_>) -> Vec<RenderedLink> {
        use crate::html::format::href;

        cx.cache()
            .intra_doc_links
            .get(&self.def_id)
            .map_or(&[][..], |v| v.as_slice())
            .iter()
            .filter_map(|ItemLink { link: s, link_text, did, ref fragment }| {
                debug!(?did);
                if let Ok((mut href, ..)) = href(*did, cx) {
                    debug!(?href);
                    if let Some(ref fragment) = *fragment {
                        href.push('#');
                        href.push_str(fragment);
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
    crate fn link_names(&self, cache: &Cache) -> Vec<RenderedLink> {
        cache
            .intra_doc_links
            .get(&self.def_id)
            .map_or(&[][..], |v| v.as_slice())
            .iter()
            .map(|ItemLink { link: s, link_text, .. }| RenderedLink {
                original_text: s.clone(),
                new_text: link_text.clone(),
                href: String::new(),
            })
            .collect()
    }

    crate fn is_crate(&self) -> bool {
        self.is_mod() && self.def_id.as_def_id().map_or(false, |did| did.index == CRATE_DEF_INDEX)
    }
    crate fn is_mod(&self) -> bool {
        self.type_() == ItemType::Module
    }
    crate fn is_trait(&self) -> bool {
        self.type_() == ItemType::Trait
    }
    crate fn is_struct(&self) -> bool {
        self.type_() == ItemType::Struct
    }
    crate fn is_enum(&self) -> bool {
        self.type_() == ItemType::Enum
    }
    crate fn is_variant(&self) -> bool {
        self.type_() == ItemType::Variant
    }
    crate fn is_associated_type(&self) -> bool {
        self.type_() == ItemType::AssocType
    }
    crate fn is_associated_const(&self) -> bool {
        self.type_() == ItemType::AssocConst
    }
    crate fn is_method(&self) -> bool {
        self.type_() == ItemType::Method
    }
    crate fn is_ty_method(&self) -> bool {
        self.type_() == ItemType::TyMethod
    }
    crate fn is_typedef(&self) -> bool {
        self.type_() == ItemType::Typedef
    }
    crate fn is_primitive(&self) -> bool {
        self.type_() == ItemType::Primitive
    }
    crate fn is_union(&self) -> bool {
        self.type_() == ItemType::Union
    }
    crate fn is_import(&self) -> bool {
        self.type_() == ItemType::Import
    }
    crate fn is_extern_crate(&self) -> bool {
        self.type_() == ItemType::ExternCrate
    }
    crate fn is_keyword(&self) -> bool {
        self.type_() == ItemType::Keyword
    }
    crate fn is_stripped(&self) -> bool {
        match *self.kind {
            StrippedItem(..) => true,
            ImportItem(ref i) => !i.should_be_displayed,
            _ => false,
        }
    }
    crate fn has_stripped_fields(&self) -> Option<bool> {
        match *self.kind {
            StructItem(ref _struct) => Some(_struct.fields_stripped),
            UnionItem(ref union) => Some(union.fields_stripped),
            VariantItem(Variant::Struct(ref vstruct)) => Some(vstruct.fields_stripped),
            _ => None,
        }
    }

    crate fn stability_class(&self, tcx: TyCtxt<'_>) -> Option<String> {
        self.stability(tcx).as_ref().and_then(|ref s| {
            let mut classes = Vec::with_capacity(2);

            if s.level.is_unstable() {
                classes.push("unstable");
            }

            // FIXME: what about non-staged API items that are deprecated?
            if self.deprecation(tcx).is_some() {
                classes.push("deprecated");
            }

            if !classes.is_empty() { Some(classes.join(" ")) } else { None }
        })
    }

    crate fn stable_since(&self, tcx: TyCtxt<'_>) -> Option<SymbolStr> {
        match self.stability(tcx)?.level {
            StabilityLevel::Stable { since, .. } => Some(since.as_str()),
            StabilityLevel::Unstable { .. } => None,
        }
    }

    crate fn const_stable_since(&self, tcx: TyCtxt<'_>) -> Option<SymbolStr> {
        match self.const_stability(tcx)?.level {
            StabilityLevel::Stable { since, .. } => Some(since.as_str()),
            StabilityLevel::Unstable { .. } => None,
        }
    }

    crate fn is_non_exhaustive(&self) -> bool {
        self.attrs.other_attrs.iter().any(|a| a.has_name(sym::non_exhaustive))
    }

    /// Returns a documentation-level item type from the item.
    crate fn type_(&self) -> ItemType {
        ItemType::from(self)
    }

    crate fn is_default(&self) -> bool {
        match *self.kind {
            ItemKind::MethodItem(_, Some(defaultness)) => {
                defaultness.has_value() && !defaultness.is_final()
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
crate enum ItemKind {
    ExternCrateItem {
        /// The crate's name, *not* the name it's imported as.
        src: Option<Symbol>,
    },
    ImportItem(Import),
    StructItem(Struct),
    UnionItem(Union),
    EnumItem(Enum),
    FunctionItem(Function),
    ModuleItem(Module),
    TypedefItem(Typedef, bool /* is associated type */),
    OpaqueTyItem(OpaqueTy),
    StaticItem(Static),
    ConstantItem(Constant),
    TraitItem(Trait),
    TraitAliasItem(TraitAlias),
    ImplItem(Impl),
    /// A method signature only. Used for required methods in traits (ie,
    /// non-default-methods).
    TyMethodItem(Function),
    /// A method with a body.
    MethodItem(Function, Option<hir::Defaultness>),
    StructFieldItem(Type),
    VariantItem(Variant),
    /// `fn`s from an extern block
    ForeignFunctionItem(Function),
    /// `static`s from an extern block
    ForeignStaticItem(Static),
    /// `type`s from an extern block
    ForeignTypeItem,
    MacroItem(Macro),
    ProcMacroItem(ProcMacro),
    PrimitiveItem(PrimitiveType),
    AssocConstItem(Type, Option<String>),
    /// An associated item in a trait or trait impl.
    ///
    /// The bounds may be non-empty if there is a `where` clause.
    /// The `Option<Type>` is the default concrete type (e.g. `trait Trait { type Target = usize; }`)
    AssocTypeItem(Vec<GenericBound>, Option<Type>),
    /// An item that has been stripped by a rustdoc pass
    StrippedItem(Box<ItemKind>),
    KeywordItem(Symbol),
}

impl ItemKind {
    /// Some items contain others such as structs (for their fields) and Enums
    /// (for their variants). This method returns those contained items.
    crate fn inner_items(&self) -> impl Iterator<Item = &Item> {
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
            | TypedefItem(_, _)
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
            | AssocConstItem(_, _)
            | AssocTypeItem(_, _)
            | StrippedItem(_)
            | KeywordItem(_) => [].iter(),
        }
    }
}

#[derive(Clone, Debug)]
crate struct Module {
    crate items: Vec<Item>,
    crate span: Span,
}

crate struct ListAttributesIter<'a> {
    attrs: slice::Iter<'a, ast::Attribute>,
    current_list: vec::IntoIter<ast::NestedMetaItem>,
    name: Symbol,
}

impl<'a> Iterator for ListAttributesIter<'a> {
    type Item = ast::NestedMetaItem;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(nested) = self.current_list.next() {
            return Some(nested);
        }

        for attr in &mut self.attrs {
            if let Some(list) = attr.meta_item_list() {
                if attr.has_name(self.name) {
                    self.current_list = list.into_iter();
                    if let Some(nested) = self.current_list.next() {
                        return Some(nested);
                    }
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let lower = self.current_list.len();
        (lower, None)
    }
}

crate trait AttributesExt {
    /// Finds an attribute as List and returns the list of attributes nested inside.
    fn lists(&self, name: Symbol) -> ListAttributesIter<'_>;

    fn span(&self) -> Option<rustc_span::Span>;

    fn inner_docs(&self) -> bool;

    fn other_attrs(&self) -> Vec<ast::Attribute>;

    fn cfg(&self, tcx: TyCtxt<'_>, hidden_cfg: &FxHashSet<Cfg>) -> Option<Arc<Cfg>>;
}

impl AttributesExt for [ast::Attribute] {
    fn lists(&self, name: Symbol) -> ListAttributesIter<'_> {
        ListAttributesIter { attrs: self.iter(), current_list: Vec::new().into_iter(), name }
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

    fn other_attrs(&self) -> Vec<ast::Attribute> {
        self.iter().filter(|attr| attr.doc_str().is_none()).cloned().collect()
    }

    fn cfg(&self, tcx: TyCtxt<'_>, hidden_cfg: &FxHashSet<Cfg>) -> Option<Arc<Cfg>> {
        let sess = tcx.sess;
        let doc_cfg_active = tcx.features().doc_cfg;

        fn single<T: IntoIterator>(it: T) -> Option<T::Item> {
            let mut iter = it.into_iter();
            let item = iter.next()?;
            if iter.next().is_some() {
                return None;
            }
            Some(item)
        }

        let mut cfg = if doc_cfg_active {
            let mut doc_cfg = self
                .iter()
                .filter(|attr| attr.has_name(sym::doc))
                .flat_map(|attr| attr.meta_item_list().unwrap_or_else(Vec::new))
                .filter(|attr| attr.has_name(sym::cfg))
                .peekable();
            if doc_cfg.peek().is_some() {
                doc_cfg
                    .filter_map(|attr| Cfg::parse(attr.meta_item()?).ok())
                    .fold(Cfg::True, |cfg, new_cfg| cfg & new_cfg)
            } else {
                self.iter()
                    .filter(|attr| attr.has_name(sym::cfg))
                    .filter_map(|attr| single(attr.meta_item_list()?))
                    .filter_map(|attr| Cfg::parse(attr.meta_item()?).ok())
                    .filter(|cfg| !hidden_cfg.contains(cfg))
                    .fold(Cfg::True, |cfg, new_cfg| cfg & new_cfg)
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
                            .and_then(|item| rustc_expand::config::parse_cfg(&item, sess))
                        {
                            match Cfg::parse(&cfg_mi) {
                                Ok(new_cfg) => cfg &= new_cfg,
                                Err(e) => sess.span_err(e.span, e.msg),
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

crate trait NestedAttributesExt {
    /// Returns `true` if the attribute list contains a specific `Word`
    fn has_word(self, word: Symbol) -> bool;
    fn get_word_attr(self, word: Symbol) -> Option<ast::NestedMetaItem>;
}

impl<I: Iterator<Item = ast::NestedMetaItem> + IntoIterator<Item = ast::NestedMetaItem>>
    NestedAttributesExt for I
{
    fn has_word(self, word: Symbol) -> bool {
        self.into_iter().any(|attr| attr.is_word() && attr.has_name(word))
    }

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
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct DocFragment {
    crate line: usize,
    crate span: rustc_span::Span,
    /// The module this doc-comment came from.
    ///
    /// This allows distinguishing between the original documentation and a pub re-export.
    /// If it is `None`, the item was not re-exported.
    crate parent_module: Option<DefId>,
    crate doc: Symbol,
    crate kind: DocFragmentKind,
    crate need_backline: bool,
    crate indent: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
crate enum DocFragmentKind {
    /// A doc fragment created from a `///` or `//!` doc comment.
    SugaredDoc,
    /// A doc fragment created from a "raw" `#[doc=""]` attribute.
    RawDoc,
}

// The goal of this function is to apply the `DocFragment` transformations that are required when
// transforming into the final markdown. So the transformations in here are:
//
// * Applying the computed indent to each lines in each doc fragment (a `DocFragment` can contain
//   multiple lines in case of `#[doc = ""]`).
// * Adding backlines between `DocFragment`s and adding an extra one if required (stored in the
//   `need_backline` field).
fn add_doc_fragment(out: &mut String, frag: &DocFragment) {
    let s = frag.doc.as_str();
    let mut iter = s.lines().peekable();
    while let Some(line) = iter.next() {
        if line.chars().any(|c| !c.is_whitespace()) {
            assert!(line.len() >= frag.indent);
            out.push_str(&line[frag.indent..]);
        } else {
            out.push_str(line);
        }
        if iter.peek().is_some() {
            out.push('\n');
        }
    }
    if frag.need_backline {
        out.push('\n');
    }
}

impl<'a> FromIterator<&'a DocFragment> for String {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a DocFragment>,
    {
        iter.into_iter().fold(String::new(), |mut acc, frag| {
            add_doc_fragment(&mut acc, &frag);
            acc
        })
    }
}

/// A link that has not yet been rendered.
///
/// This link will be turned into a rendered link by [`Item::links`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
crate struct ItemLink {
    /// The original link written in the markdown
    pub(crate) link: String,
    /// The link text displayed in the HTML.
    ///
    /// This may not be the same as `link` if there was a disambiguator
    /// in an intra-doc link (e.g. \[`fn@f`\])
    pub(crate) link_text: String,
    pub(crate) did: DefId,
    /// The url fragment to append to the link
    pub(crate) fragment: Option<String>,
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
crate struct Attributes {
    crate doc_strings: Vec<DocFragment>,
    crate other_attrs: Vec<ast::Attribute>,
}

impl Attributes {
    crate fn lists(&self, name: Symbol) -> ListAttributesIter<'_> {
        self.other_attrs.lists(name)
    }

    crate fn has_doc_flag(&self, flag: Symbol) -> bool {
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

    crate fn from_ast(
        attrs: &[ast::Attribute],
        additional_attrs: Option<(&[ast::Attribute], DefId)>,
    ) -> Attributes {
        let mut doc_strings: Vec<DocFragment> = vec![];
        let mut doc_line = 0;

        fn update_need_backline(doc_strings: &mut Vec<DocFragment>) {
            if let Some(prev) = doc_strings.last_mut() {
                prev.need_backline = true;
            }
        }

        let clean_attr = |(attr, parent_module): (&ast::Attribute, Option<DefId>)| {
            if let Some(value) = attr.doc_str() {
                trace!("got doc_str={:?}", value);
                let value = beautify_doc_string(value);
                let kind = if attr.is_doc_comment() {
                    DocFragmentKind::SugaredDoc
                } else {
                    DocFragmentKind::RawDoc
                };

                let line = doc_line;
                doc_line += value.as_str().lines().count();
                let frag = DocFragment {
                    line,
                    span: attr.span,
                    doc: value,
                    kind,
                    parent_module,
                    need_backline: false,
                    indent: 0,
                };

                update_need_backline(&mut doc_strings);

                doc_strings.push(frag);

                None
            } else {
                Some(attr.clone())
            }
        };

        // Additional documentation should be shown before the original documentation
        let other_attrs = additional_attrs
            .into_iter()
            .map(|(attrs, id)| attrs.iter().map(move |attr| (attr, Some(id))))
            .flatten()
            .chain(attrs.iter().map(|attr| (attr, None)))
            .filter_map(clean_attr)
            .collect();

        Attributes { doc_strings, other_attrs }
    }

    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    crate fn doc_value(&self) -> Option<String> {
        let mut iter = self.doc_strings.iter();

        let ori = iter.next()?;
        let mut out = String::new();
        add_doc_fragment(&mut out, &ori);
        while let Some(new_frag) = iter.next() {
            if new_frag.kind != ori.kind || new_frag.parent_module != ori.parent_module {
                break;
            }
            add_doc_fragment(&mut out, &new_frag);
        }
        if out.is_empty() { None } else { Some(out) }
    }

    /// Return the doc-comments on this item, grouped by the module they came from.
    ///
    /// The module can be different if this is a re-export with added documentation.
    crate fn collapsed_doc_value_by_module_level(&self) -> FxHashMap<Option<DefId>, String> {
        let mut ret = FxHashMap::default();

        for new_frag in self.doc_strings.iter() {
            let out = ret.entry(new_frag.parent_module).or_default();
            add_doc_fragment(out, &new_frag);
        }
        ret
    }

    /// Finds all `doc` attributes as NameValues and returns their corresponding values, joined
    /// with newlines.
    crate fn collapsed_doc_value(&self) -> Option<String> {
        if self.doc_strings.is_empty() { None } else { Some(self.doc_strings.iter().collect()) }
    }

    crate fn get_doc_aliases(&self) -> Box<[String]> {
        let mut aliases = FxHashSet::default();

        for attr in self.other_attrs.lists(sym::doc).filter(|a| a.has_name(sym::alias)) {
            if let Some(values) = attr.meta_item_list() {
                for l in values {
                    match l.literal().unwrap().kind {
                        ast::LitKind::Str(s, _) => {
                            aliases.insert(s.as_str().to_string());
                        }
                        _ => unreachable!(),
                    }
                }
            } else {
                aliases.insert(attr.value_str().map(|s| s.to_string()).unwrap());
            }
        }
        aliases.into_iter().collect::<Vec<String>>().into()
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

impl Hash for Attributes {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.doc_strings.hash(hasher);
        for attr in &self.other_attrs {
            attr.id.hash(hasher);
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate enum GenericBound {
    TraitBound(PolyTrait, hir::TraitBoundModifier),
    Outlives(Lifetime),
}

impl GenericBound {
    crate fn maybe_sized(cx: &mut DocContext<'_>) -> GenericBound {
        let did = cx.tcx.require_lang_item(LangItem::Sized, None);
        let empty = cx.tcx.intern_substs(&[]);
        let path = external_path(cx, did, false, vec![], empty);
        inline::record_extern_fqn(cx, did, ItemType::Trait);
        GenericBound::TraitBound(
            PolyTrait { trait_: path, generic_params: Vec::new() },
            hir::TraitBoundModifier::Maybe,
        )
    }

    crate fn is_sized_bound(&self, cx: &DocContext<'_>) -> bool {
        use rustc_hir::TraitBoundModifier as TBM;
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, TBM::None) = *self {
            if Some(trait_.def_id()) == cx.tcx.lang_items().sized_trait() {
                return true;
            }
        }
        false
    }

    crate fn get_poly_trait(&self) -> Option<PolyTrait> {
        if let GenericBound::TraitBound(ref p, _) = *self {
            return Some(p.clone());
        }
        None
    }

    crate fn get_trait_path(&self) -> Option<Path> {
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, _) = *self {
            Some(trait_.clone())
        } else {
            None
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct Lifetime(pub Symbol);

impl Lifetime {
    crate fn get_ref(&self) -> SymbolStr {
        self.0.as_str()
    }

    crate fn statik() -> Lifetime {
        Lifetime(kw::StaticLifetime)
    }

    crate fn elided() -> Lifetime {
        Lifetime(kw::UnderscoreLifetime)
    }
}

#[derive(Clone, Debug)]
crate enum WherePredicate {
    BoundPredicate { ty: Type, bounds: Vec<GenericBound>, bound_params: Vec<Lifetime> },
    RegionPredicate { lifetime: Lifetime, bounds: Vec<GenericBound> },
    EqPredicate { lhs: Type, rhs: Type },
}

impl WherePredicate {
    crate fn get_bounds(&self) -> Option<&[GenericBound]> {
        match *self {
            WherePredicate::BoundPredicate { ref bounds, .. } => Some(bounds),
            WherePredicate::RegionPredicate { ref bounds, .. } => Some(bounds),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate enum GenericParamDefKind {
    Lifetime {
        outlives: Vec<Lifetime>,
    },
    Type {
        did: DefId,
        bounds: Vec<GenericBound>,
        default: Option<Type>,
        synthetic: Option<hir::SyntheticTyParamKind>,
    },
    Const {
        did: DefId,
        ty: Type,
        default: Option<String>,
    },
}

impl GenericParamDefKind {
    crate fn is_type(&self) -> bool {
        matches!(self, GenericParamDefKind::Type { .. })
    }

    // FIXME(eddyb) this either returns the default of a type parameter, or the
    // type of a `const` parameter. It seems that the intention is to *visit*
    // any embedded types, but `get_type` seems to be the wrong name for that.
    crate fn get_type(&self) -> Option<Type> {
        match self {
            GenericParamDefKind::Type { default, .. } => default.clone(),
            GenericParamDefKind::Const { ty, .. } => Some(ty.clone()),
            GenericParamDefKind::Lifetime { .. } => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct GenericParamDef {
    crate name: Symbol,
    crate kind: GenericParamDefKind,
}

impl GenericParamDef {
    crate fn is_synthetic_type_param(&self) -> bool {
        match self.kind {
            GenericParamDefKind::Lifetime { .. } | GenericParamDefKind::Const { .. } => false,
            GenericParamDefKind::Type { ref synthetic, .. } => synthetic.is_some(),
        }
    }

    crate fn is_type(&self) -> bool {
        self.kind.is_type()
    }

    crate fn get_type(&self) -> Option<Type> {
        self.kind.get_type()
    }

    crate fn get_bounds(&self) -> Option<&[GenericBound]> {
        match self.kind {
            GenericParamDefKind::Type { ref bounds, .. } => Some(bounds),
            _ => None,
        }
    }
}

// maybe use a Generic enum and use Vec<Generic>?
#[derive(Clone, Debug, Default)]
crate struct Generics {
    crate params: Vec<GenericParamDef>,
    crate where_predicates: Vec<WherePredicate>,
}

#[derive(Clone, Debug)]
crate struct Function {
    crate decl: FnDecl,
    crate generics: Generics,
    crate header: hir::FnHeader,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct FnDecl {
    crate inputs: Arguments,
    crate output: FnRetTy,
    crate c_variadic: bool,
}

impl FnDecl {
    crate fn self_type(&self) -> Option<SelfTy> {
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
    crate fn sugared_async_return_type(&self) -> FnRetTy {
        match &self.output {
            FnRetTy::Return(Type::ImplTrait(bounds)) => match &bounds[0] {
                GenericBound::TraitBound(PolyTrait { trait_, .. }, ..) => {
                    let bindings = trait_.bindings().unwrap();
                    FnRetTy::Return(bindings[0].ty().clone())
                }
                _ => panic!("unexpected desugaring of async function"),
            },
            _ => panic!("unexpected desugaring of async function"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct Arguments {
    crate values: Vec<Argument>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct Argument {
    crate type_: Type,
    crate name: Symbol,
}

#[derive(Clone, PartialEq, Debug)]
crate enum SelfTy {
    SelfValue,
    SelfBorrowed(Option<Lifetime>, Mutability),
    SelfExplicit(Type),
}

impl Argument {
    crate fn to_self(&self) -> Option<SelfTy> {
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
crate enum FnRetTy {
    Return(Type),
    DefaultReturn,
}

impl GetDefId for FnRetTy {
    fn def_id(&self) -> Option<DefId> {
        match *self {
            Return(ref ty) => ty.def_id(),
            DefaultReturn => None,
        }
    }

    fn def_id_full(&self, cache: &Cache) -> Option<DefId> {
        match *self {
            Return(ref ty) => ty.def_id_full(cache),
            DefaultReturn => None,
        }
    }
}

#[derive(Clone, Debug)]
crate struct Trait {
    crate unsafety: hir::Unsafety,
    crate items: Vec<Item>,
    crate generics: Generics,
    crate bounds: Vec<GenericBound>,
    crate is_auto: bool,
}

#[derive(Clone, Debug)]
crate struct TraitAlias {
    crate generics: Generics,
    crate bounds: Vec<GenericBound>,
}

/// A trait reference, which may have higher ranked lifetimes.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct PolyTrait {
    crate trait_: Path,
    crate generic_params: Vec<GenericParamDef>,
}

/// Rustdoc's representation of types, mostly based on the [`hir::Ty`].
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate enum Type {
    /// A named type, which could be a trait.
    ///
    /// This is mostly Rustdoc's version of [`hir::Path`]. It has to be different because Rustdoc's [`PathSegment`] can contain cleaned generics.
    ResolvedPath { path: Path, did: DefId },
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
    QPath {
        name: Symbol,
        self_type: Box<Type>,
        /// FIXME: This is a hack that should be removed; see [this discussion][1].
        ///
        /// [1]: https://github.com/rust-lang/rust/pull/85479#discussion_r635729093
        self_def_id: Option<DefId>,
        trait_: Path,
    },

    /// A type that is inferred: `_`
    Infer,

    /// An `impl Trait`: `impl TraitA + TraitB + ...`
    ImplTrait(Vec<GenericBound>),
}

crate trait GetDefId {
    /// Use this method to get the [`DefId`] of a [`clean`] AST node.
    /// This will return [`None`] when called on a primitive [`clean::Type`].
    /// Use [`Self::def_id_full`] if you want to include primitives.
    ///
    /// [`clean`]: crate::clean
    /// [`clean::Type`]: crate::clean::Type
    // FIXME: get rid of this function and always use `def_id_full`
    fn def_id(&self) -> Option<DefId>;

    /// Use this method to get the [DefId] of a [clean] AST node, including [PrimitiveType]s.
    ///
    /// See [`Self::def_id`] for more.
    ///
    /// [clean]: crate::clean
    fn def_id_full(&self, cache: &Cache) -> Option<DefId>;
}

impl<T: GetDefId> GetDefId for Option<T> {
    fn def_id(&self) -> Option<DefId> {
        self.as_ref().and_then(|d| d.def_id())
    }

    fn def_id_full(&self, cache: &Cache) -> Option<DefId> {
        self.as_ref().and_then(|d| d.def_id_full(cache))
    }
}

impl Type {
    crate fn primitive_type(&self) -> Option<PrimitiveType> {
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
    crate fn is_assoc_ty(&self) -> bool {
        match self {
            ResolvedPath { path, .. } => path.is_assoc_ty(),
            _ => false,
        }
    }

    crate fn is_self_type(&self) -> bool {
        match *self {
            Generic(name) => name == kw::SelfUpper,
            _ => false,
        }
    }

    crate fn generics(&self) -> Option<Vec<&Type>> {
        match self {
            ResolvedPath { path, .. } => path.generics(),
            _ => None,
        }
    }

    crate fn is_full_generic(&self) -> bool {
        matches!(self, Type::Generic(_))
    }

    crate fn is_primitive(&self) -> bool {
        self.primitive_type().is_some()
    }

    crate fn projection(&self) -> Option<(&Type, DefId, Symbol)> {
        let (self_, trait_, name) = match self {
            QPath { self_type, trait_, name, .. } => (self_type, trait_, name),
            _ => return None,
        };
        Some((&self_, trait_.def_id(), *name))
    }

    fn inner_def_id(&self, cache: Option<&Cache>) -> Option<DefId> {
        let t: PrimitiveType = match *self {
            ResolvedPath { did, .. } => return Some(did),
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
            QPath { ref self_type, .. } => return self_type.inner_def_id(cache),
            Generic(_) | Infer | ImplTrait(_) => return None,
        };
        cache.and_then(|c| Primitive(t).def_id_full(c))
    }
}

impl GetDefId for Type {
    fn def_id(&self) -> Option<DefId> {
        self.inner_def_id(None)
    }

    fn def_id_full(&self, cache: &Cache) -> Option<DefId> {
        self.inner_def_id(Some(cache))
    }
}

/// A primitive (aka, builtin) type.
///
/// This represents things like `i32`, `str`, etc.
///
/// N.B. This has to be different from [`hir::PrimTy`] because it also includes types that aren't
/// paths, like [`Self::Unit`].
#[derive(Clone, PartialEq, Eq, Hash, Copy, Debug)]
crate enum PrimitiveType {
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

impl PrimitiveType {
    crate fn from_hir(prim: hir::PrimTy) -> PrimitiveType {
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

    crate fn from_symbol(s: Symbol) -> Option<PrimitiveType> {
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

    crate fn impls(&self, tcx: TyCtxt<'_>) -> &'static ArrayVec<DefId, 4> {
        Self::all_impls(tcx).get(self).expect("missing impl for primitive type")
    }

    crate fn all_impls(tcx: TyCtxt<'_>) -> &'static FxHashMap<PrimitiveType, ArrayVec<DefId, 4>> {
        static CELL: OnceCell<FxHashMap<PrimitiveType, ArrayVec<DefId, 4>>> = OnceCell::new();

        CELL.get_or_init(move || {
            use self::PrimitiveType::*;

            let single = |a: Option<DefId>| a.into_iter().collect();
            let both = |a: Option<DefId>, b: Option<DefId>| -> ArrayVec<_, 4> {
                a.into_iter().chain(b).collect()
            };

            let lang_items = tcx.lang_items();
            map! {
                Isize => single(lang_items.isize_impl()),
                I8 => single(lang_items.i8_impl()),
                I16 => single(lang_items.i16_impl()),
                I32 => single(lang_items.i32_impl()),
                I64 => single(lang_items.i64_impl()),
                I128 => single(lang_items.i128_impl()),
                Usize => single(lang_items.usize_impl()),
                U8 => single(lang_items.u8_impl()),
                U16 => single(lang_items.u16_impl()),
                U32 => single(lang_items.u32_impl()),
                U64 => single(lang_items.u64_impl()),
                U128 => single(lang_items.u128_impl()),
                F32 => both(lang_items.f32_impl(), lang_items.f32_runtime_impl()),
                F64 => both(lang_items.f64_impl(), lang_items.f64_runtime_impl()),
                Char => single(lang_items.char_impl()),
                Bool => single(lang_items.bool_impl()),
                Str => both(lang_items.str_impl(), lang_items.str_alloc_impl()),
                Slice => {
                    lang_items
                        .slice_impl()
                        .into_iter()
                        .chain(lang_items.slice_u8_impl())
                        .chain(lang_items.slice_alloc_impl())
                        .chain(lang_items.slice_u8_alloc_impl())
                        .collect()
                },
                Array => single(lang_items.array_impl()),
                Tuple => ArrayVec::new(),
                Unit => ArrayVec::new(),
                RawPointer => {
                    lang_items
                        .const_ptr_impl()
                        .into_iter()
                        .chain(lang_items.mut_ptr_impl())
                        .chain(lang_items.const_slice_ptr_impl())
                        .chain(lang_items.mut_slice_ptr_impl())
                        .collect()
                },
                Reference => ArrayVec::new(),
                Fn => ArrayVec::new(),
                Never => ArrayVec::new(),
            }
        })
    }

    crate fn as_sym(&self) -> Symbol {
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
    crate fn primitive_locations(tcx: TyCtxt<'_>) -> &FxHashMap<PrimitiveType, DefId> {
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
crate enum Visibility {
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
    crate fn is_public(&self) -> bool {
        matches!(self, Visibility::Public)
    }
}

#[derive(Clone, Debug)]
crate struct Struct {
    crate struct_type: CtorKind,
    crate generics: Generics,
    crate fields: Vec<Item>,
    crate fields_stripped: bool,
}

#[derive(Clone, Debug)]
crate struct Union {
    crate generics: Generics,
    crate fields: Vec<Item>,
    crate fields_stripped: bool,
}

/// This is a more limited form of the standard Struct, different in that
/// it lacks the things most items have (name, id, parameterization). Found
/// only as a variant in an enum.
#[derive(Clone, Debug)]
crate struct VariantStruct {
    crate struct_type: CtorKind,
    crate fields: Vec<Item>,
    crate fields_stripped: bool,
}

#[derive(Clone, Debug)]
crate struct Enum {
    crate variants: IndexVec<VariantIdx, Item>,
    crate generics: Generics,
    crate variants_stripped: bool,
}

#[derive(Clone, Debug)]
crate enum Variant {
    CLike,
    Tuple(Vec<Item>),
    Struct(VariantStruct),
}

/// Small wrapper around [`rustc_span::Span`] that adds helper methods
/// and enforces calling [`rustc_span::Span::source_callsite()`].
#[derive(Copy, Clone, Debug)]
crate struct Span(rustc_span::Span);

impl Span {
    /// Wraps a [`rustc_span::Span`]. In case this span is the result of a macro expansion, the
    /// span will be updated to point to the macro invocation instead of the macro definition.
    ///
    /// (See rust-lang/rust#39726)
    crate fn new(sp: rustc_span::Span) -> Self {
        Self(sp.source_callsite())
    }

    crate fn inner(&self) -> rustc_span::Span {
        self.0
    }

    crate fn dummy() -> Self {
        Self(rustc_span::DUMMY_SP)
    }

    crate fn is_dummy(&self) -> bool {
        self.0.is_dummy()
    }

    crate fn filename(&self, sess: &Session) -> FileName {
        sess.source_map().span_to_filename(self.0)
    }

    crate fn lo(&self, sess: &Session) -> Loc {
        sess.source_map().lookup_char_pos(self.0.lo())
    }

    crate fn hi(&self, sess: &Session) -> Loc {
        sess.source_map().lookup_char_pos(self.0.hi())
    }

    crate fn cnum(&self, sess: &Session) -> CrateNum {
        // FIXME: is there a time when the lo and hi crate would be different?
        self.lo(sess).file.cnum
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct Path {
    crate res: Res,
    crate segments: Vec<PathSegment>,
}

impl Path {
    crate fn def_id(&self) -> DefId {
        self.res.def_id()
    }

    crate fn last(&self) -> Symbol {
        self.segments.last().expect("segments were empty").name
    }

    crate fn last_name(&self) -> SymbolStr {
        self.segments.last().expect("segments were empty").name.as_str()
    }

    crate fn whole_name(&self) -> String {
        self.segments
            .iter()
            .map(|s| if s.name == kw::PathRoot { String::new() } else { s.name.to_string() })
            .intersperse("::".into())
            .collect()
    }

    /// Checks if this is a `T::Name` path for an associated type.
    crate fn is_assoc_ty(&self) -> bool {
        match self.res {
            Res::SelfTy(..) if self.segments.len() != 1 => true,
            Res::Def(DefKind::TyParam, _) if self.segments.len() != 1 => true,
            Res::Def(DefKind::AssocTy, _) => true,
            _ => false,
        }
    }

    crate fn generics(&self) -> Option<Vec<&Type>> {
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

    crate fn bindings(&self) -> Option<&[TypeBinding]> {
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
crate enum GenericArg {
    Lifetime(Lifetime),
    Type(Type),
    Const(Box<Constant>),
    Infer,
}

// `GenericArg` can occur many times in a single `Path`, so make sure it
// doesn't increase in size unexpectedly.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(GenericArg, 80);

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate enum GenericArgs {
    AngleBracketed { args: Vec<GenericArg>, bindings: Vec<TypeBinding> },
    Parenthesized { inputs: Vec<Type>, output: Option<Box<Type>> },
}

// `GenericArgs` is in every `PathSegment`, so its size can significantly
// affect rustdoc's memory usage.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(GenericArgs, 56);

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct PathSegment {
    crate name: Symbol,
    crate args: GenericArgs,
}

// `PathSegment` usually occurs multiple times in every `Path`, so its size can
// significantly affect rustdoc's memory usage.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(PathSegment, 64);

#[derive(Clone, Debug)]
crate struct Typedef {
    crate type_: Type,
    crate generics: Generics,
    /// `type_` can come from either the HIR or from metadata. If it comes from HIR, it may be a type
    /// alias instead of the final type. This will always have the final type, regardless of whether
    /// `type_` came from HIR or from metadata.
    ///
    /// If `item_type.is_none()`, `type_` is guarenteed to come from metadata (and therefore hold the
    /// final type).
    crate item_type: Option<Type>,
}

impl GetDefId for Typedef {
    fn def_id(&self) -> Option<DefId> {
        self.type_.def_id()
    }

    fn def_id_full(&self, cache: &Cache) -> Option<DefId> {
        self.type_.def_id_full(cache)
    }
}

#[derive(Clone, Debug)]
crate struct OpaqueTy {
    crate bounds: Vec<GenericBound>,
    crate generics: Generics,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct BareFunctionDecl {
    crate unsafety: hir::Unsafety,
    crate generic_params: Vec<GenericParamDef>,
    crate decl: FnDecl,
    crate abi: Abi,
}

#[derive(Clone, Debug)]
crate struct Static {
    crate type_: Type,
    crate mutability: Mutability,
    crate expr: Option<BodyId>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
crate struct Constant {
    crate type_: Type,
    crate kind: ConstantKind,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
crate enum ConstantKind {
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
    crate fn expr(&self, tcx: TyCtxt<'_>) -> String {
        match self.kind {
            ConstantKind::TyConst { ref expr } => expr.clone(),
            ConstantKind::Extern { def_id } => print_inlined_const(tcx, def_id),
            ConstantKind::Local { body, .. } | ConstantKind::Anonymous { body } => {
                print_const_expr(tcx, body)
            }
        }
    }

    crate fn value(&self, tcx: TyCtxt<'_>) -> Option<String> {
        match self.kind {
            ConstantKind::TyConst { .. } | ConstantKind::Anonymous { .. } => None,
            ConstantKind::Extern { def_id } | ConstantKind::Local { def_id, .. } => {
                print_evaluated_const(tcx, def_id)
            }
        }
    }

    crate fn is_literal(&self, tcx: TyCtxt<'_>) -> bool {
        match self.kind {
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
crate struct Impl {
    crate span: Span,
    crate unsafety: hir::Unsafety,
    crate generics: Generics,
    crate trait_: Option<Path>,
    crate for_: Type,
    crate items: Vec<Item>,
    crate negative_polarity: bool,
    crate synthetic: bool,
    crate blanket_impl: Option<Box<Type>>,
}

impl Impl {
    crate fn provided_trait_methods(&self, tcx: TyCtxt<'_>) -> FxHashSet<Symbol> {
        self.trait_
            .as_ref()
            .map(|t| t.def_id())
            .map(|did| tcx.provided_trait_methods(did).map(|meth| meth.ident.name).collect())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
crate struct Import {
    crate kind: ImportKind,
    crate source: ImportSource,
    crate should_be_displayed: bool,
}

impl Import {
    crate fn new_simple(name: Symbol, source: ImportSource, should_be_displayed: bool) -> Self {
        Self { kind: ImportKind::Simple(name), source, should_be_displayed }
    }

    crate fn new_glob(source: ImportSource, should_be_displayed: bool) -> Self {
        Self { kind: ImportKind::Glob, source, should_be_displayed }
    }
}

#[derive(Clone, Debug)]
crate enum ImportKind {
    // use source as str;
    Simple(Symbol),
    // use source::*;
    Glob,
}

#[derive(Clone, Debug)]
crate struct ImportSource {
    crate path: Path,
    crate did: Option<DefId>,
}

#[derive(Clone, Debug)]
crate struct Macro {
    crate source: String,
}

#[derive(Clone, Debug)]
crate struct ProcMacro {
    crate kind: MacroKind,
    crate helpers: Vec<Symbol>,
}

/// An type binding on an associated type (e.g., `A = Bar` in `Foo<A = Bar>` or
/// `A: Send + Sync` in `Foo<A: Send + Sync>`).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate struct TypeBinding {
    crate name: Symbol,
    crate kind: TypeBindingKind,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
crate enum TypeBindingKind {
    Equality { ty: Type },
    Constraint { bounds: Vec<GenericBound> },
}

impl TypeBinding {
    crate fn ty(&self) -> &Type {
        match self.kind {
            TypeBindingKind::Equality { ref ty } => ty,
            _ => panic!("expected equality type binding for parenthesized generic args"),
        }
    }
}
