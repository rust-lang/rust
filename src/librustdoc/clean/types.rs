use std::cell::RefCell;
use std::default::Default;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;
use std::{slice, vec};

use rustc_ast::ast::{self, AttrStyle};
use rustc_ast::attr;
use rustc_ast::util::comments::strip_doc_comment_decoration;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_hir::lang_items;
use rustc_hir::Mutability;
use rustc_index::vec::IndexVec;
use rustc_middle::middle::stability;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::DUMMY_SP;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{self, FileName};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi::Abi;

use crate::clean::cfg::Cfg;
use crate::clean::external_path;
use crate::clean::inline;
use crate::clean::types::Type::{QPath, ResolvedPath};
use crate::core::DocContext;
use crate::doctree;
use crate::html::item_type::ItemType;
use crate::html::render::{cache, ExternalLocation};

use self::FnRetTy::*;
use self::ItemEnum::*;
use self::SelfTy::*;
use self::Type::*;

thread_local!(pub static MAX_DEF_ID: RefCell<FxHashMap<CrateNum, DefId>> = Default::default());

#[derive(Clone, Debug)]
pub struct Crate {
    pub name: String,
    pub version: Option<String>,
    pub src: FileName,
    pub module: Option<Item>,
    pub externs: Vec<(CrateNum, ExternalCrate)>,
    pub primitives: Vec<(DefId, PrimitiveType, Attributes)>,
    // These are later on moved into `CACHEKEY`, leaving the map empty.
    // Only here so that they can be filtered through the rustdoc passes.
    pub external_traits: Rc<RefCell<FxHashMap<DefId, Trait>>>,
    pub masked_crates: FxHashSet<CrateNum>,
    pub collapsed: bool,
}

#[derive(Clone, Debug)]
pub struct ExternalCrate {
    pub name: String,
    pub src: FileName,
    pub attrs: Attributes,
    pub primitives: Vec<(DefId, PrimitiveType, Attributes)>,
    pub keywords: Vec<(DefId, String, Attributes)>,
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[derive(Clone)]
pub struct Item {
    /// Stringified span
    pub source: Span,
    /// Not everything has a name. E.g., impls
    pub name: Option<String>,
    pub attrs: Attributes,
    pub inner: ItemEnum,
    pub visibility: Visibility,
    pub def_id: DefId,
    pub stability: Option<Stability>,
    pub deprecation: Option<Deprecation>,
}

impl fmt::Debug for Item {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fake = self.is_fake();
        let def_id: &dyn fmt::Debug = if fake { &"**FAKE**" } else { &self.def_id };

        fmt.debug_struct("Item")
            .field("source", &self.source)
            .field("name", &self.name)
            .field("attrs", &self.attrs)
            .field("inner", &self.inner)
            .field("visibility", &self.visibility)
            .field("def_id", def_id)
            .field("stability", &self.stability)
            .field("deprecation", &self.deprecation)
            .finish()
    }
}

impl Item {
    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    pub fn doc_value(&self) -> Option<&str> {
        self.attrs.doc_value()
    }

    /// Finds all `doc` attributes as NameValues and returns their corresponding values, joined
    /// with newlines.
    pub fn collapsed_doc_value(&self) -> Option<String> {
        self.attrs.collapsed_doc_value()
    }

    pub fn links(&self) -> Vec<(String, String)> {
        self.attrs.links(&self.def_id.krate)
    }

    pub fn is_crate(&self) -> bool {
        match self.inner {
            StrippedItem(box ModuleItem(Module { is_crate: true, .. }))
            | ModuleItem(Module { is_crate: true, .. }) => true,
            _ => false,
        }
    }
    pub fn is_mod(&self) -> bool {
        self.type_() == ItemType::Module
    }
    pub fn is_trait(&self) -> bool {
        self.type_() == ItemType::Trait
    }
    pub fn is_struct(&self) -> bool {
        self.type_() == ItemType::Struct
    }
    pub fn is_enum(&self) -> bool {
        self.type_() == ItemType::Enum
    }
    pub fn is_variant(&self) -> bool {
        self.type_() == ItemType::Variant
    }
    pub fn is_associated_type(&self) -> bool {
        self.type_() == ItemType::AssocType
    }
    pub fn is_associated_const(&self) -> bool {
        self.type_() == ItemType::AssocConst
    }
    pub fn is_method(&self) -> bool {
        self.type_() == ItemType::Method
    }
    pub fn is_ty_method(&self) -> bool {
        self.type_() == ItemType::TyMethod
    }
    pub fn is_typedef(&self) -> bool {
        self.type_() == ItemType::Typedef
    }
    pub fn is_primitive(&self) -> bool {
        self.type_() == ItemType::Primitive
    }
    pub fn is_union(&self) -> bool {
        self.type_() == ItemType::Union
    }
    pub fn is_import(&self) -> bool {
        self.type_() == ItemType::Import
    }
    pub fn is_extern_crate(&self) -> bool {
        self.type_() == ItemType::ExternCrate
    }
    pub fn is_keyword(&self) -> bool {
        self.type_() == ItemType::Keyword
    }
    pub fn is_stripped(&self) -> bool {
        match self.inner {
            StrippedItem(..) => true,
            _ => false,
        }
    }
    pub fn has_stripped_fields(&self) -> Option<bool> {
        match self.inner {
            StructItem(ref _struct) => Some(_struct.fields_stripped),
            UnionItem(ref union) => Some(union.fields_stripped),
            VariantItem(Variant { kind: VariantKind::Struct(ref vstruct) }) => {
                Some(vstruct.fields_stripped)
            }
            _ => None,
        }
    }

    pub fn stability_class(&self) -> Option<String> {
        self.stability.as_ref().and_then(|ref s| {
            let mut classes = Vec::with_capacity(2);

            if s.level == stability::Unstable {
                classes.push("unstable");
            }

            // FIXME: what about non-staged API items that are deprecated?
            if self.deprecation.is_some() {
                classes.push("deprecated");
            }

            if !classes.is_empty() { Some(classes.join(" ")) } else { None }
        })
    }

    pub fn stable_since(&self) -> Option<&str> {
        self.stability.as_ref().map(|s| &s.since[..])
    }

    pub fn is_non_exhaustive(&self) -> bool {
        self.attrs.other_attrs.iter().any(|a| a.check_name(sym::non_exhaustive))
    }

    /// Returns a documentation-level item type from the item.
    pub fn type_(&self) -> ItemType {
        ItemType::from(self)
    }

    pub fn is_default(&self) -> bool {
        match self.inner {
            ItemEnum::MethodItem(ref meth) => {
                if let Some(defaultness) = meth.defaultness {
                    defaultness.has_value() && !defaultness.is_final()
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// See comments on next_def_id
    pub fn is_fake(&self) -> bool {
        MAX_DEF_ID.with(|m| {
            m.borrow().get(&self.def_id.krate).map(|id| self.def_id >= *id).unwrap_or(false)
        })
    }
}

#[derive(Clone, Debug)]
pub enum ItemEnum {
    ExternCrateItem(String, Option<String>),
    ImportItem(Import),
    StructItem(Struct),
    UnionItem(Union),
    EnumItem(Enum),
    FunctionItem(Function),
    ModuleItem(Module),
    TypedefItem(Typedef, bool /* is associated type */),
    OpaqueTyItem(OpaqueTy, bool /* is associated type */),
    StaticItem(Static),
    ConstantItem(Constant),
    TraitItem(Trait),
    TraitAliasItem(TraitAlias),
    ImplItem(Impl),
    /// A method signature only. Used for required methods in traits (ie,
    /// non-default-methods).
    TyMethodItem(TyMethod),
    /// A method with a body.
    MethodItem(Method),
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
    AssocTypeItem(Vec<GenericBound>, Option<Type>),
    /// An item that has been stripped by a rustdoc pass
    StrippedItem(Box<ItemEnum>),
    KeywordItem(String),
}

impl ItemEnum {
    pub fn is_associated(&self) -> bool {
        match *self {
            ItemEnum::TypedefItem(_, _) | ItemEnum::AssocTypeItem(_, _) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Module {
    pub items: Vec<Item>,
    pub is_crate: bool,
}

pub struct ListAttributesIter<'a> {
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
                if attr.check_name(self.name) {
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

pub trait AttributesExt {
    /// Finds an attribute as List and returns the list of attributes nested inside.
    fn lists(&self, name: Symbol) -> ListAttributesIter<'_>;
}

impl AttributesExt for [ast::Attribute] {
    fn lists(&self, name: Symbol) -> ListAttributesIter<'_> {
        ListAttributesIter { attrs: self.iter(), current_list: Vec::new().into_iter(), name }
    }
}

pub trait NestedAttributesExt {
    /// Returns `true` if the attribute list contains a specific `Word`
    fn has_word(self, word: Symbol) -> bool;
}

impl<I: IntoIterator<Item = ast::NestedMetaItem>> NestedAttributesExt for I {
    fn has_word(self, word: Symbol) -> bool {
        self.into_iter().any(|attr| attr.is_word() && attr.check_name(word))
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
pub enum DocFragment {
    /// A doc fragment created from a `///` or `//!` doc comment.
    SugaredDoc(usize, rustc_span::Span, String),
    /// A doc fragment created from a "raw" `#[doc=""]` attribute.
    RawDoc(usize, rustc_span::Span, String),
    /// A doc fragment created from a `#[doc(include="filename")]` attribute. Contains both the
    /// given filename and the file contents.
    Include(usize, rustc_span::Span, String, String),
}

impl DocFragment {
    pub fn as_str(&self) -> &str {
        match *self {
            DocFragment::SugaredDoc(_, _, ref s) => &s[..],
            DocFragment::RawDoc(_, _, ref s) => &s[..],
            DocFragment::Include(_, _, _, ref s) => &s[..],
        }
    }

    pub fn span(&self) -> rustc_span::Span {
        match *self {
            DocFragment::SugaredDoc(_, span, _)
            | DocFragment::RawDoc(_, span, _)
            | DocFragment::Include(_, span, _, _) => span,
        }
    }
}

impl<'a> FromIterator<&'a DocFragment> for String {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a DocFragment>,
    {
        iter.into_iter().fold(String::new(), |mut acc, frag| {
            if !acc.is_empty() {
                acc.push('\n');
            }
            match *frag {
                DocFragment::SugaredDoc(_, _, ref docs)
                | DocFragment::RawDoc(_, _, ref docs)
                | DocFragment::Include(_, _, _, ref docs) => acc.push_str(docs),
            }

            acc
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Attributes {
    pub doc_strings: Vec<DocFragment>,
    pub other_attrs: Vec<ast::Attribute>,
    pub cfg: Option<Arc<Cfg>>,
    pub span: Option<rustc_span::Span>,
    /// map from Rust paths to resolved defs and potential URL fragments
    pub links: Vec<(String, Option<DefId>, Option<String>)>,
    pub inner_docs: bool,
}

impl Attributes {
    /// Extracts the content from an attribute `#[doc(cfg(content))]`.
    pub fn extract_cfg(mi: &ast::MetaItem) -> Option<&ast::MetaItem> {
        use rustc_ast::ast::NestedMetaItem::MetaItem;

        if let ast::MetaItemKind::List(ref nmis) = mi.kind {
            if nmis.len() == 1 {
                if let MetaItem(ref cfg_mi) = nmis[0] {
                    if cfg_mi.check_name(sym::cfg) {
                        if let ast::MetaItemKind::List(ref cfg_nmis) = cfg_mi.kind {
                            if cfg_nmis.len() == 1 {
                                if let MetaItem(ref content_mi) = cfg_nmis[0] {
                                    return Some(content_mi);
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Reads a `MetaItem` from within an attribute, looks for whether it is a
    /// `#[doc(include="file")]`, and returns the filename and contents of the file as loaded from
    /// its expansion.
    pub fn extract_include(mi: &ast::MetaItem) -> Option<(String, String)> {
        mi.meta_item_list().and_then(|list| {
            for meta in list {
                if meta.check_name(sym::include) {
                    // the actual compiled `#[doc(include="filename")]` gets expanded to
                    // `#[doc(include(file="filename", contents="file contents")]` so we need to
                    // look for that instead
                    return meta.meta_item_list().and_then(|list| {
                        let mut filename: Option<String> = None;
                        let mut contents: Option<String> = None;

                        for it in list {
                            if it.check_name(sym::file) {
                                if let Some(name) = it.value_str() {
                                    filename = Some(name.to_string());
                                }
                            } else if it.check_name(sym::contents) {
                                if let Some(docs) = it.value_str() {
                                    contents = Some(docs.to_string());
                                }
                            }
                        }

                        if let (Some(filename), Some(contents)) = (filename, contents) {
                            Some((filename, contents))
                        } else {
                            None
                        }
                    });
                }
            }

            None
        })
    }

    pub fn has_doc_flag(&self, flag: Symbol) -> bool {
        for attr in &self.other_attrs {
            if !attr.check_name(sym::doc) {
                continue;
            }

            if let Some(items) = attr.meta_item_list() {
                if items.iter().filter_map(|i| i.meta_item()).any(|it| it.check_name(flag)) {
                    return true;
                }
            }
        }

        false
    }

    pub fn from_ast(diagnostic: &::rustc_errors::Handler, attrs: &[ast::Attribute]) -> Attributes {
        let mut doc_strings = vec![];
        let mut sp = None;
        let mut cfg = Cfg::True;
        let mut doc_line = 0;

        let other_attrs = attrs
            .iter()
            .filter_map(|attr| {
                if let Some(value) = attr.doc_str() {
                    let (value, mk_fragment): (_, fn(_, _, _) -> _) = if attr.is_doc_comment() {
                        (strip_doc_comment_decoration(value), DocFragment::SugaredDoc)
                    } else {
                        (value.to_string(), DocFragment::RawDoc)
                    };

                    let line = doc_line;
                    doc_line += value.lines().count();
                    doc_strings.push(mk_fragment(line, attr.span, value));

                    if sp.is_none() {
                        sp = Some(attr.span);
                    }
                    None
                } else {
                    if attr.check_name(sym::doc) {
                        if let Some(mi) = attr.meta() {
                            if let Some(cfg_mi) = Attributes::extract_cfg(&mi) {
                                // Extracted #[doc(cfg(...))]
                                match Cfg::parse(cfg_mi) {
                                    Ok(new_cfg) => cfg &= new_cfg,
                                    Err(e) => diagnostic.span_err(e.span, e.msg),
                                }
                            } else if let Some((filename, contents)) =
                                Attributes::extract_include(&mi)
                            {
                                let line = doc_line;
                                doc_line += contents.lines().count();
                                doc_strings.push(DocFragment::Include(
                                    line, attr.span, filename, contents,
                                ));
                            }
                        }
                    }
                    Some(attr.clone())
                }
            })
            .collect();

        // treat #[target_feature(enable = "feat")] attributes as if they were
        // #[doc(cfg(target_feature = "feat"))] attributes as well
        for attr in attrs.lists(sym::target_feature) {
            if attr.check_name(sym::enable) {
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

        let inner_docs = attrs
            .iter()
            .find(|a| a.doc_str().is_some())
            .map_or(true, |a| a.style == AttrStyle::Inner);

        Attributes {
            doc_strings,
            other_attrs,
            cfg: if cfg == Cfg::True { None } else { Some(Arc::new(cfg)) },
            span: sp,
            links: vec![],
            inner_docs,
        }
    }

    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    pub fn doc_value(&self) -> Option<&str> {
        self.doc_strings.first().map(|s| s.as_str())
    }

    /// Finds all `doc` attributes as NameValues and returns their corresponding values, joined
    /// with newlines.
    pub fn collapsed_doc_value(&self) -> Option<String> {
        if !self.doc_strings.is_empty() { Some(self.doc_strings.iter().collect()) } else { None }
    }

    /// Gets links as a vector
    ///
    /// Cache must be populated before call
    pub fn links(&self, krate: &CrateNum) -> Vec<(String, String)> {
        use crate::html::format::href;
        use crate::html::render::CURRENT_DEPTH;

        self.links
            .iter()
            .filter_map(|&(ref s, did, ref fragment)| {
                match did {
                    Some(did) => {
                        if let Some((mut href, ..)) = href(did) {
                            if let Some(ref fragment) = *fragment {
                                href.push_str("#");
                                href.push_str(fragment);
                            }
                            Some((s.clone(), href))
                        } else {
                            None
                        }
                    }
                    None => {
                        if let Some(ref fragment) = *fragment {
                            let cache = cache();
                            let url = match cache.extern_locations.get(krate) {
                                Some(&(_, _, ExternalLocation::Local)) => {
                                    let depth = CURRENT_DEPTH.with(|l| l.get());
                                    "../".repeat(depth)
                                }
                                Some(&(_, _, ExternalLocation::Remote(ref s))) => s.to_string(),
                                Some(&(_, _, ExternalLocation::Unknown)) | None => {
                                    String::from("https://doc.rust-lang.org/nightly")
                                }
                            };
                            // This is a primitive so the url is done "by hand".
                            let tail = fragment.find('#').unwrap_or_else(|| fragment.len());
                            Some((
                                s.clone(),
                                format!(
                                    "{}{}std/primitive.{}.html{}",
                                    url,
                                    if !url.ends_with('/') { "/" } else { "" },
                                    &fragment[..tail],
                                    &fragment[tail..]
                                ),
                            ))
                        } else {
                            panic!("This isn't a primitive?!");
                        }
                    }
                }
            })
            .collect()
    }

    pub fn get_doc_aliases(&self) -> FxHashSet<String> {
        self.other_attrs
            .lists(sym::doc)
            .filter(|a| a.check_name(sym::alias))
            .filter_map(|a| a.value_str().map(|s| s.to_string().replace("\"", "")))
            .filter(|v| !v.is_empty())
            .collect::<FxHashSet<_>>()
    }
}

impl PartialEq for Attributes {
    fn eq(&self, rhs: &Self) -> bool {
        self.doc_strings == rhs.doc_strings
            && self.cfg == rhs.cfg
            && self.span == rhs.span
            && self.links == rhs.links
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
        self.cfg.hash(hasher);
        self.span.hash(hasher);
        self.links.hash(hasher);
        for attr in &self.other_attrs {
            attr.id.hash(hasher);
        }
    }
}

impl AttributesExt for Attributes {
    fn lists(&self, name: Symbol) -> ListAttributesIter<'_> {
        self.other_attrs.lists(name)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum GenericBound {
    TraitBound(PolyTrait, hir::TraitBoundModifier),
    Outlives(Lifetime),
}

impl GenericBound {
    pub fn maybe_sized(cx: &DocContext<'_>) -> GenericBound {
        let did = cx.tcx.require_lang_item(lang_items::SizedTraitLangItem, None);
        let empty = cx.tcx.intern_substs(&[]);
        let path = external_path(cx, cx.tcx.item_name(did), Some(did), false, vec![], empty);
        inline::record_extern_fqn(cx, did, TypeKind::Trait);
        GenericBound::TraitBound(
            PolyTrait {
                trait_: ResolvedPath { path, param_names: None, did, is_generic: false },
                generic_params: Vec::new(),
            },
            hir::TraitBoundModifier::Maybe,
        )
    }

    pub fn is_sized_bound(&self, cx: &DocContext<'_>) -> bool {
        use rustc_hir::TraitBoundModifier as TBM;
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, TBM::None) = *self {
            if trait_.def_id() == cx.tcx.lang_items().sized_trait() {
                return true;
            }
        }
        false
    }

    pub fn get_poly_trait(&self) -> Option<PolyTrait> {
        if let GenericBound::TraitBound(ref p, _) = *self {
            return Some(p.clone());
        }
        None
    }

    pub fn get_trait_type(&self) -> Option<Type> {
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, _) = *self {
            Some(trait_.clone())
        } else {
            None
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Lifetime(pub String);

impl Lifetime {
    pub fn get_ref<'a>(&'a self) -> &'a str {
        let Lifetime(ref s) = *self;
        let s: &'a str = s;
        s
    }

    pub fn statik() -> Lifetime {
        Lifetime("'static".to_string())
    }
}

#[derive(Clone, Debug)]
pub enum WherePredicate {
    BoundPredicate { ty: Type, bounds: Vec<GenericBound> },
    RegionPredicate { lifetime: Lifetime, bounds: Vec<GenericBound> },
    EqPredicate { lhs: Type, rhs: Type },
}

impl WherePredicate {
    pub fn get_bounds(&self) -> Option<&[GenericBound]> {
        match *self {
            WherePredicate::BoundPredicate { ref bounds, .. } => Some(bounds),
            WherePredicate::RegionPredicate { ref bounds, .. } => Some(bounds),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum GenericParamDefKind {
    Lifetime,
    Type {
        did: DefId,
        bounds: Vec<GenericBound>,
        default: Option<Type>,
        synthetic: Option<hir::SyntheticTyParamKind>,
    },
    Const {
        did: DefId,
        ty: Type,
    },
}

impl GenericParamDefKind {
    pub fn is_type(&self) -> bool {
        match *self {
            GenericParamDefKind::Type { .. } => true,
            _ => false,
        }
    }

    // FIXME(eddyb) this either returns the default of a type parameter, or the
    // type of a `const` parameter. It seems that the intention is to *visit*
    // any embedded types, but `get_type` seems to be the wrong name for that.
    pub fn get_type(&self) -> Option<Type> {
        match self {
            GenericParamDefKind::Type { default, .. } => default.clone(),
            GenericParamDefKind::Const { ty, .. } => Some(ty.clone()),
            GenericParamDefKind::Lifetime => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct GenericParamDef {
    pub name: String,
    pub kind: GenericParamDefKind,
}

impl GenericParamDef {
    pub fn is_synthetic_type_param(&self) -> bool {
        match self.kind {
            GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => false,
            GenericParamDefKind::Type { ref synthetic, .. } => synthetic.is_some(),
        }
    }

    pub fn is_type(&self) -> bool {
        self.kind.is_type()
    }

    pub fn get_type(&self) -> Option<Type> {
        self.kind.get_type()
    }

    pub fn get_bounds(&self) -> Option<&[GenericBound]> {
        match self.kind {
            GenericParamDefKind::Type { ref bounds, .. } => Some(bounds),
            _ => None,
        }
    }
}

// maybe use a Generic enum and use Vec<Generic>?
#[derive(Clone, Debug, Default)]
pub struct Generics {
    pub params: Vec<GenericParamDef>,
    pub where_predicates: Vec<WherePredicate>,
}

#[derive(Clone, Debug)]
pub struct Method {
    pub generics: Generics,
    pub decl: FnDecl,
    pub header: hir::FnHeader,
    pub defaultness: Option<hir::Defaultness>,
    pub all_types: Vec<(Type, TypeKind)>,
    pub ret_types: Vec<(Type, TypeKind)>,
}

#[derive(Clone, Debug)]
pub struct TyMethod {
    pub header: hir::FnHeader,
    pub decl: FnDecl,
    pub generics: Generics,
    pub all_types: Vec<(Type, TypeKind)>,
    pub ret_types: Vec<(Type, TypeKind)>,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub decl: FnDecl,
    pub generics: Generics,
    pub header: hir::FnHeader,
    pub all_types: Vec<(Type, TypeKind)>,
    pub ret_types: Vec<(Type, TypeKind)>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct FnDecl {
    pub inputs: Arguments,
    pub output: FnRetTy,
    pub c_variadic: bool,
    pub attrs: Attributes,
}

impl FnDecl {
    pub fn self_type(&self) -> Option<SelfTy> {
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
    pub fn sugared_async_return_type(&self) -> FnRetTy {
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
pub struct Arguments {
    pub values: Vec<Argument>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Argument {
    pub type_: Type,
    pub name: String,
}

#[derive(Clone, PartialEq, Debug)]
pub enum SelfTy {
    SelfValue,
    SelfBorrowed(Option<Lifetime>, Mutability),
    SelfExplicit(Type),
}

impl Argument {
    pub fn to_self(&self) -> Option<SelfTy> {
        if self.name != "self" {
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
pub enum FnRetTy {
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
}

#[derive(Clone, Debug)]
pub struct Trait {
    pub auto: bool,
    pub unsafety: hir::Unsafety,
    pub items: Vec<Item>,
    pub generics: Generics,
    pub bounds: Vec<GenericBound>,
    pub is_spotlight: bool,
    pub is_auto: bool,
}

#[derive(Clone, Debug)]
pub struct TraitAlias {
    pub generics: Generics,
    pub bounds: Vec<GenericBound>,
}

/// A trait reference, which may have higher ranked lifetimes.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct PolyTrait {
    pub trait_: Type,
    pub generic_params: Vec<GenericParamDef>,
}

/// A representation of a type suitable for hyperlinking purposes. Ideally, one can get the original
/// type out of the AST/`TyCtxt` given one of these, if more information is needed. Most
/// importantly, it does not preserve mutability or boxes.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Type {
    /// Structs/enums/traits (most that would be an `hir::TyKind::Path`).
    ResolvedPath {
        path: Path,
        param_names: Option<Vec<GenericBound>>,
        did: DefId,
        /// `true` if is a `T::Name` path for associated types.
        is_generic: bool,
    },
    /// For parameterized types, so the consumer of the JSON don't go
    /// looking for types which don't exist anywhere.
    Generic(String),
    /// Primitives are the fixed-size numeric types (plus int/usize/float), char,
    /// arrays, slices, and tuples.
    Primitive(PrimitiveType),
    /// `extern "ABI" fn`
    BareFunction(Box<BareFunctionDecl>),
    Tuple(Vec<Type>),
    Slice(Box<Type>),
    Array(Box<Type>, String),
    Never,
    RawPointer(Mutability, Box<Type>),
    BorrowedRef {
        lifetime: Option<Lifetime>,
        mutability: Mutability,
        type_: Box<Type>,
    },

    // `<Type as Trait>::Name`
    QPath {
        name: String,
        self_type: Box<Type>,
        trait_: Box<Type>,
    },

    // `_`
    Infer,

    // `impl TraitA + TraitB + ...`
    ImplTrait(Vec<GenericBound>),
}

#[derive(Clone, PartialEq, Eq, Hash, Copy, Debug)]
pub enum PrimitiveType {
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

#[derive(Clone, PartialEq, Eq, Hash, Copy, Debug)]
pub enum TypeKind {
    Enum,
    Function,
    Module,
    Const,
    Static,
    Struct,
    Union,
    Trait,
    Typedef,
    Foreign,
    Macro,
    Attr,
    Derive,
    TraitAlias,
}

pub trait GetDefId {
    fn def_id(&self) -> Option<DefId>;
}

impl<T: GetDefId> GetDefId for Option<T> {
    fn def_id(&self) -> Option<DefId> {
        self.as_ref().and_then(|d| d.def_id())
    }
}

impl Type {
    pub fn primitive_type(&self) -> Option<PrimitiveType> {
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
            BorrowedRef { type_: box Generic(..), .. } => Some(PrimitiveType::Reference),
            BareFunction(..) => Some(PrimitiveType::Fn),
            Never => Some(PrimitiveType::Never),
            _ => None,
        }
    }

    pub fn is_generic(&self) -> bool {
        match *self {
            ResolvedPath { is_generic, .. } => is_generic,
            _ => false,
        }
    }

    pub fn is_self_type(&self) -> bool {
        match *self {
            Generic(ref name) => name == "Self",
            _ => false,
        }
    }

    pub fn generics(&self) -> Option<Vec<Type>> {
        match *self {
            ResolvedPath { ref path, .. } => path.segments.last().and_then(|seg| {
                if let GenericArgs::AngleBracketed { ref args, .. } = seg.args {
                    Some(
                        args.iter()
                            .filter_map(|arg| match arg {
                                GenericArg::Type(ty) => Some(ty.clone()),
                                _ => None,
                            })
                            .collect(),
                    )
                } else {
                    None
                }
            }),
            _ => None,
        }
    }

    pub fn bindings(&self) -> Option<&[TypeBinding]> {
        match *self {
            ResolvedPath { ref path, .. } => path.segments.last().and_then(|seg| {
                if let GenericArgs::AngleBracketed { ref bindings, .. } = seg.args {
                    Some(&**bindings)
                } else {
                    None
                }
            }),
            _ => None,
        }
    }

    pub fn is_full_generic(&self) -> bool {
        match *self {
            Type::Generic(_) => true,
            _ => false,
        }
    }

    pub fn projection(&self) -> Option<(&Type, DefId, &str)> {
        let (self_, trait_, name) = match self {
            QPath { ref self_type, ref trait_, ref name } => (self_type, trait_, name),
            _ => return None,
        };
        let trait_did = match **trait_ {
            ResolvedPath { did, .. } => did,
            _ => return None,
        };
        Some((&self_, trait_did, name))
    }
}

impl GetDefId for Type {
    fn def_id(&self) -> Option<DefId> {
        match *self {
            ResolvedPath { did, .. } => Some(did),
            Primitive(p) => crate::html::render::cache().primitive_locations.get(&p).cloned(),
            BorrowedRef { type_: box Generic(..), .. } => {
                Primitive(PrimitiveType::Reference).def_id()
            }
            BorrowedRef { ref type_, .. } => type_.def_id(),
            Tuple(ref tys) => {
                if tys.is_empty() {
                    Primitive(PrimitiveType::Unit).def_id()
                } else {
                    Primitive(PrimitiveType::Tuple).def_id()
                }
            }
            BareFunction(..) => Primitive(PrimitiveType::Fn).def_id(),
            Never => Primitive(PrimitiveType::Never).def_id(),
            Slice(..) => Primitive(PrimitiveType::Slice).def_id(),
            Array(..) => Primitive(PrimitiveType::Array).def_id(),
            RawPointer(..) => Primitive(PrimitiveType::RawPointer).def_id(),
            QPath { ref self_type, .. } => self_type.def_id(),
            _ => None,
        }
    }
}

impl PrimitiveType {
    pub fn from_symbol(s: Symbol) -> Option<PrimitiveType> {
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

    pub fn as_str(&self) -> &'static str {
        use self::PrimitiveType::*;
        match *self {
            Isize => "isize",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            I128 => "i128",
            Usize => "usize",
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            U128 => "u128",
            F32 => "f32",
            F64 => "f64",
            Str => "str",
            Bool => "bool",
            Char => "char",
            Array => "array",
            Slice => "slice",
            Tuple => "tuple",
            Unit => "unit",
            RawPointer => "pointer",
            Reference => "reference",
            Fn => "fn",
            Never => "never",
        }
    }

    pub fn to_url_str(&self) -> &'static str {
        self.as_str()
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

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Visibility {
    Public,
    Inherited,
    Crate,
    Restricted(DefId, Path),
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub struct_type: doctree::StructType,
    pub generics: Generics,
    pub fields: Vec<Item>,
    pub fields_stripped: bool,
}

#[derive(Clone, Debug)]
pub struct Union {
    pub struct_type: doctree::StructType,
    pub generics: Generics,
    pub fields: Vec<Item>,
    pub fields_stripped: bool,
}

/// This is a more limited form of the standard Struct, different in that
/// it lacks the things most items have (name, id, parameterization). Found
/// only as a variant in an enum.
#[derive(Clone, Debug)]
pub struct VariantStruct {
    pub struct_type: doctree::StructType,
    pub fields: Vec<Item>,
    pub fields_stripped: bool,
}

#[derive(Clone, Debug)]
pub struct Enum {
    pub variants: IndexVec<VariantIdx, Item>,
    pub generics: Generics,
    pub variants_stripped: bool,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub kind: VariantKind,
}

#[derive(Clone, Debug)]
pub enum VariantKind {
    CLike,
    Tuple(Vec<Type>),
    Struct(VariantStruct),
}

#[derive(Clone, Debug)]
pub struct Span {
    pub filename: FileName,
    pub cnum: CrateNum,
    pub loline: usize,
    pub locol: usize,
    pub hiline: usize,
    pub hicol: usize,
    pub original: rustc_span::Span,
}

impl Span {
    pub fn empty() -> Span {
        Span {
            filename: FileName::Anon(0),
            cnum: LOCAL_CRATE,
            loline: 0,
            locol: 0,
            hiline: 0,
            hicol: 0,
            original: rustc_span::DUMMY_SP,
        }
    }

    pub fn span(&self) -> rustc_span::Span {
        self.original
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Path {
    pub global: bool,
    pub res: Res,
    pub segments: Vec<PathSegment>,
}

impl Path {
    pub fn last_name(&self) -> &str {
        self.segments.last().expect("segments were empty").name.as_str()
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum GenericArg {
    Lifetime(Lifetime),
    Type(Type),
    Const(Constant),
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum GenericArgs {
    AngleBracketed { args: Vec<GenericArg>, bindings: Vec<TypeBinding> },
    Parenthesized { inputs: Vec<Type>, output: Option<Type> },
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct PathSegment {
    pub name: String,
    pub args: GenericArgs,
}

#[derive(Clone, Debug)]
pub struct Typedef {
    pub type_: Type,
    pub generics: Generics,
    // Type of target item.
    pub item_type: Option<Type>,
}

impl GetDefId for Typedef {
    fn def_id(&self) -> Option<DefId> {
        self.type_.def_id()
    }
}

#[derive(Clone, Debug)]
pub struct OpaqueTy {
    pub bounds: Vec<GenericBound>,
    pub generics: Generics,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct BareFunctionDecl {
    pub unsafety: hir::Unsafety,
    pub generic_params: Vec<GenericParamDef>,
    pub decl: FnDecl,
    pub abi: Abi,
}

#[derive(Clone, Debug)]
pub struct Static {
    pub type_: Type,
    pub mutability: Mutability,
    /// It's useful to have the value of a static documented, but I have no
    /// desire to represent expressions (that'd basically be all of the AST,
    /// which is huge!). So, have a string.
    pub expr: String,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Constant {
    pub type_: Type,
    pub expr: String,
    pub value: Option<String>,
    pub is_literal: bool,
}

#[derive(Clone, PartialEq, Debug)]
pub enum ImplPolarity {
    Positive,
    Negative,
}

#[derive(Clone, Debug)]
pub struct Impl {
    pub unsafety: hir::Unsafety,
    pub generics: Generics,
    pub provided_trait_methods: FxHashSet<String>,
    pub trait_: Option<Type>,
    pub for_: Type,
    pub items: Vec<Item>,
    pub polarity: Option<ImplPolarity>,
    pub synthetic: bool,
    pub blanket_impl: Option<Type>,
}

#[derive(Clone, Debug)]
pub enum Import {
    // use source as str;
    Simple(String, ImportSource),
    // use source::*;
    Glob(ImportSource),
}

#[derive(Clone, Debug)]
pub struct ImportSource {
    pub path: Path,
    pub did: Option<DefId>,
}

#[derive(Clone, Debug)]
pub struct Macro {
    pub source: String,
    pub imported_from: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ProcMacro {
    pub kind: MacroKind,
    pub helpers: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Stability {
    pub level: stability::StabilityLevel,
    pub feature: Option<String>,
    pub since: String,
    pub unstable_reason: Option<String>,
    pub issue: Option<NonZeroU32>,
}

#[derive(Clone, Debug)]
pub struct Deprecation {
    pub since: Option<String>,
    pub note: Option<String>,
    pub is_since_rustc_version: bool,
}

/// An type binding on an associated type (e.g., `A = Bar` in `Foo<A = Bar>` or
/// `A: Send + Sync` in `Foo<A: Send + Sync>`).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TypeBinding {
    pub name: String,
    pub kind: TypeBindingKind,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum TypeBindingKind {
    Equality { ty: Type },
    Constraint { bounds: Vec<GenericBound> },
}

impl TypeBinding {
    pub fn ty(&self) -> &Type {
        match self.kind {
            TypeBindingKind::Equality { ref ty } => ty,
            _ => panic!("expected equality type binding for parenthesized generic args"),
        }
    }
}
