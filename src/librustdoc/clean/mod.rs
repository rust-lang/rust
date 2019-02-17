//! This module contains the "cleaned" pieces of the AST, and the functions
//! that clean them.

pub mod inline;
pub mod cfg;
mod simplify;
mod auto_trait;
mod blanket_impl;
pub mod def_ctor;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc_data_structures::sync::Lrc;
use rustc_target::spec::abi::Abi;
use rustc_typeck::hir_ty_to_ty;
use rustc::infer::region_constraints::{RegionConstraintData, Constraint};
use rustc::middle::resolve_lifetime as rl;
use rustc::middle::lang_items;
use rustc::middle::stability;
use rustc::mir::interpret::GlobalId;
use rustc::hir::{self, GenericArg, HirVec};
use rustc::hir::def::{self, Def, CtorKind};
use rustc::hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc::ty::subst::Substs;
use rustc::ty::{self, TyCtxt, Region, RegionVid, Ty, AdtKind};
use rustc::ty::fold::TypeFolder;
use rustc::ty::layout::VariantIdx;
use rustc::util::nodemap::{FxHashMap, FxHashSet};
use syntax::ast::{self, AttrStyle, Ident};
use syntax::attr;
use syntax::ext::base::MacroKind;
use syntax::source_map::{dummy_spanned, Spanned};
use syntax::ptr::P;
use syntax::symbol::keywords::{self, Keyword};
use syntax::symbol::InternedString;
use syntax_pos::{self, DUMMY_SP, Pos, FileName};

use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::default::Default;
use std::{mem, slice, vec};
use std::iter::{FromIterator, once};
use std::rc::Rc;
use std::str::FromStr;
use std::cell::RefCell;
use std::sync::Arc;
use std::u32;

use parking_lot::ReentrantMutex;

use core::{self, DocContext};
use doctree;
use visit_ast;
use html::render::{cache, ExternalLocation};
use html::item_type::ItemType;

use self::cfg::Cfg;
use self::auto_trait::AutoTraitFinder;
use self::blanket_impl::BlanketImplFinder;

pub use self::Type::*;
pub use self::Mutability::*;
pub use self::ItemEnum::*;
pub use self::SelfTy::*;
pub use self::FunctionRetTy::*;
pub use self::Visibility::{Public, Inherited};

thread_local!(pub static MAX_DEF_ID: RefCell<FxHashMap<CrateNum, DefId>> = Default::default());

const FN_OUTPUT_NAME: &'static str = "Output";

// extract the stability index for a node from tcx, if possible
fn get_stability(cx: &DocContext, def_id: DefId) -> Option<Stability> {
    cx.tcx.lookup_stability(def_id).clean(cx)
}

fn get_deprecation(cx: &DocContext, def_id: DefId) -> Option<Deprecation> {
    cx.tcx.lookup_deprecation(def_id).clean(cx)
}

pub trait Clean<T> {
    fn clean(&self, cx: &DocContext) -> T;
}

impl<T: Clean<U>, U> Clean<Vec<U>> for [T] {
    fn clean(&self, cx: &DocContext) -> Vec<U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

impl<T: Clean<U>, U, V: Idx> Clean<IndexVec<V, U>> for IndexVec<V, T> {
    fn clean(&self, cx: &DocContext) -> IndexVec<V, U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

impl<T: Clean<U>, U> Clean<U> for P<T> {
    fn clean(&self, cx: &DocContext) -> U {
        (**self).clean(cx)
    }
}

impl<T: Clean<U>, U> Clean<U> for Rc<T> {
    fn clean(&self, cx: &DocContext) -> U {
        (**self).clean(cx)
    }
}

impl<T: Clean<U>, U> Clean<Option<U>> for Option<T> {
    fn clean(&self, cx: &DocContext) -> Option<U> {
        self.as_ref().map(|v| v.clean(cx))
    }
}

impl<T, U> Clean<U> for ty::Binder<T> where T: Clean<U> {
    fn clean(&self, cx: &DocContext) -> U {
        self.skip_binder().clean(cx)
    }
}

impl<T: Clean<U>, U> Clean<Vec<U>> for P<[T]> {
    fn clean(&self, cx: &DocContext) -> Vec<U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

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
    pub external_traits: Arc<ReentrantMutex<RefCell<FxHashMap<DefId, Trait>>>>,
    pub masked_crates: FxHashSet<CrateNum>,
}

impl<'a, 'tcx, 'rcx> Clean<Crate> for visit_ast::RustdocVisitor<'a, 'tcx, 'rcx> {
    fn clean(&self, cx: &DocContext) -> Crate {
        use ::visit_lib::LibEmbargoVisitor;

        {
            let mut r = cx.renderinfo.borrow_mut();
            r.deref_trait_did = cx.tcx.lang_items().deref_trait();
            r.deref_mut_trait_did = cx.tcx.lang_items().deref_mut_trait();
            r.owned_box_did = cx.tcx.lang_items().owned_box();
        }

        let mut externs = Vec::new();
        for &cnum in cx.tcx.crates().iter() {
            externs.push((cnum, cnum.clean(cx)));
            // Analyze doc-reachability for extern items
            LibEmbargoVisitor::new(cx).visit_lib(cnum);
        }
        externs.sort_by(|&(a, _), &(b, _)| a.cmp(&b));

        // Clean the crate, translating the entire libsyntax AST to one that is
        // understood by rustdoc.
        let mut module = self.module.clean(cx);
        let mut masked_crates = FxHashSet::default();

        match module.inner {
            ModuleItem(ref module) => {
                for it in &module.items {
                    // `compiler_builtins` should be masked too, but we can't apply
                    // `#[doc(masked)]` to the injected `extern crate` because it's unstable.
                    if it.is_extern_crate()
                        && (it.attrs.has_doc_flag("masked")
                            || self.cx.tcx.is_compiler_builtins(it.def_id.krate))
                    {
                        masked_crates.insert(it.def_id.krate);
                    }
                }
            }
            _ => unreachable!(),
        }

        let ExternalCrate { name, src, primitives, keywords, .. } = LOCAL_CRATE.clean(cx);
        {
            let m = match module.inner {
                ModuleItem(ref mut m) => m,
                _ => unreachable!(),
            };
            m.items.extend(primitives.iter().map(|&(def_id, prim, ref attrs)| {
                Item {
                    source: Span::empty(),
                    name: Some(prim.to_url_str().to_string()),
                    attrs: attrs.clone(),
                    visibility: Some(Public),
                    stability: get_stability(cx, def_id),
                    deprecation: get_deprecation(cx, def_id),
                    def_id,
                    inner: PrimitiveItem(prim),
                }
            }));
            m.items.extend(keywords.into_iter().map(|(def_id, kw, attrs)| {
                Item {
                    source: Span::empty(),
                    name: Some(kw.clone()),
                    attrs: attrs,
                    visibility: Some(Public),
                    stability: get_stability(cx, def_id),
                    deprecation: get_deprecation(cx, def_id),
                    def_id,
                    inner: KeywordItem(kw),
                }
            }));
        }

        Crate {
            name,
            version: None,
            src,
            module: Some(module),
            externs,
            primitives,
            external_traits: cx.external_traits.clone(),
            masked_crates,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ExternalCrate {
    pub name: String,
    pub src: FileName,
    pub attrs: Attributes,
    pub primitives: Vec<(DefId, PrimitiveType, Attributes)>,
    pub keywords: Vec<(DefId, String, Attributes)>,
}

impl Clean<ExternalCrate> for CrateNum {
    fn clean(&self, cx: &DocContext) -> ExternalCrate {
        let root = DefId { krate: *self, index: CRATE_DEF_INDEX };
        let krate_span = cx.tcx.def_span(root);
        let krate_src = cx.sess().source_map().span_to_filename(krate_span);

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
        let as_primitive = |def: Def| {
            if let Def::Mod(def_id) = def {
                let attrs = cx.tcx.get_attrs(def_id).clean(cx);
                let mut prim = None;
                for attr in attrs.lists("doc") {
                    if let Some(v) = attr.value_str() {
                        if attr.check_name("primitive") {
                            prim = PrimitiveType::from_str(&v.as_str());
                            if prim.is_some() {
                                break;
                            }
                            // FIXME: should warn on unknown primitives?
                        }
                    }
                }
                return prim.map(|p| (def_id, p, attrs));
            }
            None
        };
        let primitives = if root.is_local() {
            cx.tcx.hir().krate().module.item_ids.iter().filter_map(|&id| {
                let item = cx.tcx.hir().expect_item(id.id);
                match item.node {
                    hir::ItemKind::Mod(_) => {
                        as_primitive(Def::Mod(cx.tcx.hir().local_def_id(id.id)))
                    }
                    hir::ItemKind::Use(ref path, hir::UseKind::Single)
                    if item.vis.node.is_pub() => {
                        as_primitive(path.def).map(|(_, prim, attrs)| {
                            // Pretend the primitive is local.
                            (cx.tcx.hir().local_def_id(id.id), prim, attrs)
                        })
                    }
                    _ => None
                }
            }).collect()
        } else {
            cx.tcx.item_children(root).iter().map(|item| item.def)
              .filter_map(as_primitive).collect()
        };

        let as_keyword = |def: Def| {
            if let Def::Mod(def_id) = def {
                let attrs = cx.tcx.get_attrs(def_id).clean(cx);
                let mut keyword = None;
                for attr in attrs.lists("doc") {
                    if let Some(v) = attr.value_str() {
                        if attr.check_name("keyword") {
                            keyword = Keyword::from_str(&v.as_str()).ok()
                                                                    .map(|x| x.name().to_string());
                            if keyword.is_some() {
                                break
                            }
                            // FIXME: should warn on unknown keywords?
                        }
                    }
                }
                return keyword.map(|p| (def_id, p, attrs));
            }
            None
        };
        let keywords = if root.is_local() {
            cx.tcx.hir().krate().module.item_ids.iter().filter_map(|&id| {
                let item = cx.tcx.hir().expect_item(id.id);
                match item.node {
                    hir::ItemKind::Mod(_) => {
                        as_keyword(Def::Mod(cx.tcx.hir().local_def_id(id.id)))
                    }
                    hir::ItemKind::Use(ref path, hir::UseKind::Single)
                    if item.vis.node.is_pub() => {
                        as_keyword(path.def).map(|(_, prim, attrs)| {
                            (cx.tcx.hir().local_def_id(id.id), prim, attrs)
                        })
                    }
                    _ => None
                }
            }).collect()
        } else {
            cx.tcx.item_children(root).iter().map(|item| item.def)
              .filter_map(as_keyword).collect()
        };

        ExternalCrate {
            name: cx.tcx.crate_name(*self).to_string(),
            src: krate_src,
            attrs: cx.tcx.get_attrs(root).clean(cx),
            primitives,
            keywords,
        }
    }
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct Item {
    /// Stringified span
    pub source: Span,
    /// Not everything has a name. E.g., impls
    pub name: Option<String>,
    pub attrs: Attributes,
    pub inner: ItemEnum,
    pub visibility: Option<Visibility>,
    pub def_id: DefId,
    pub stability: Option<Stability>,
    pub deprecation: Option<Deprecation>,
}

impl fmt::Debug for Item {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {

        let fake = MAX_DEF_ID.with(|m| m.borrow().get(&self.def_id.krate)
                                   .map(|id| self.def_id >= *id).unwrap_or(false));
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
    pub fn doc_value<'a>(&'a self) -> Option<&'a str> {
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
            StrippedItem(box ModuleItem(Module { is_crate: true, ..})) |
            ModuleItem(Module { is_crate: true, ..}) => true,
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
    pub fn is_associated_type(&self) -> bool {
        self.type_() == ItemType::AssociatedType
    }
    pub fn is_associated_const(&self) -> bool {
        self.type_() == ItemType::AssociatedConst
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
        match self.inner { StrippedItem(..) => true, _ => false }
    }
    pub fn has_stripped_fields(&self) -> Option<bool> {
        match self.inner {
            StructItem(ref _struct) => Some(_struct.fields_stripped),
            UnionItem(ref union) => Some(union.fields_stripped),
            VariantItem(Variant { kind: VariantKind::Struct(ref vstruct)} ) => {
                Some(vstruct.fields_stripped)
            },
            _ => None,
        }
    }

    pub fn stability_class(&self) -> Option<String> {
        self.stability.as_ref().and_then(|ref s| {
            let mut classes = Vec::with_capacity(2);

            if s.level == stability::Unstable {
                classes.push("unstable");
            }

            if s.deprecation.is_some() {
                classes.push("deprecated");
            }

            if classes.len() != 0 {
                Some(classes.join(" "))
            } else {
                None
            }
        })
    }

    pub fn stable_since(&self) -> Option<&str> {
        self.stability.as_ref().map(|s| &s.since[..])
    }

    pub fn is_non_exhaustive(&self) -> bool {
        self.attrs.other_attrs.iter()
            .any(|a| a.name().as_str() == "non_exhaustive")
    }

    /// Returns a documentation-level item type from the item.
    pub fn type_(&self) -> ItemType {
        ItemType::from(self)
    }

    /// Returns the info in the item's `#[deprecated]` or `#[rustc_deprecated]` attributes.
    ///
    /// If the item is not deprecated, returns `None`.
    pub fn deprecation(&self) -> Option<&Deprecation> {
        self.deprecation
            .as_ref()
            .or_else(|| self.stability.as_ref().and_then(|s| s.deprecation.as_ref()))
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum ItemEnum {
    ExternCrateItem(String, Option<String>),
    ImportItem(Import),
    StructItem(Struct),
    UnionItem(Union),
    EnumItem(Enum),
    FunctionItem(Function),
    ModuleItem(Module),
    TypedefItem(Typedef, bool /* is associated type */),
    ExistentialItem(Existential, bool /* is associated type */),
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
    AssociatedConstItem(Type, Option<String>),
    AssociatedTypeItem(Vec<GenericBound>, Option<Type>),
    /// An item that has been stripped by a rustdoc pass
    StrippedItem(Box<ItemEnum>),
    KeywordItem(String),
}

impl ItemEnum {
    pub fn generics(&self) -> Option<&Generics> {
        Some(match *self {
            ItemEnum::StructItem(ref s) => &s.generics,
            ItemEnum::EnumItem(ref e) => &e.generics,
            ItemEnum::FunctionItem(ref f) => &f.generics,
            ItemEnum::TypedefItem(ref t, _) => &t.generics,
            ItemEnum::ExistentialItem(ref t, _) => &t.generics,
            ItemEnum::TraitItem(ref t) => &t.generics,
            ItemEnum::ImplItem(ref i) => &i.generics,
            ItemEnum::TyMethodItem(ref i) => &i.generics,
            ItemEnum::MethodItem(ref i) => &i.generics,
            ItemEnum::ForeignFunctionItem(ref f) => &f.generics,
            ItemEnum::TraitAliasItem(ref ta) => &ta.generics,
            _ => return None,
        })
    }

    pub fn is_associated(&self) -> bool {
        match *self {
            ItemEnum::TypedefItem(_, _) |
            ItemEnum::AssociatedTypeItem(_, _) => true,
            _ => false,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Module {
    pub items: Vec<Item>,
    pub is_crate: bool,
}

impl Clean<Item> for doctree::Module {
    fn clean(&self, cx: &DocContext) -> Item {
        let name = if self.name.is_some() {
            self.name.expect("No name provided").clean(cx)
        } else {
            String::new()
        };

        // maintain a stack of mod ids, for doc comment path resolution
        // but we also need to resolve the module's own docs based on whether its docs were written
        // inside or outside the module, so check for that
        let attrs = self.attrs.clean(cx);

        let mut items: Vec<Item> = vec![];
        items.extend(self.extern_crates.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.imports.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.structs.iter().map(|x| x.clean(cx)));
        items.extend(self.unions.iter().map(|x| x.clean(cx)));
        items.extend(self.enums.iter().map(|x| x.clean(cx)));
        items.extend(self.fns.iter().map(|x| x.clean(cx)));
        items.extend(self.foreigns.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.mods.iter().map(|x| x.clean(cx)));
        items.extend(self.typedefs.iter().map(|x| x.clean(cx)));
        items.extend(self.existentials.iter().map(|x| x.clean(cx)));
        items.extend(self.statics.iter().map(|x| x.clean(cx)));
        items.extend(self.constants.iter().map(|x| x.clean(cx)));
        items.extend(self.traits.iter().map(|x| x.clean(cx)));
        items.extend(self.impls.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.macros.iter().map(|x| x.clean(cx)));
        items.extend(self.proc_macros.iter().map(|x| x.clean(cx)));
        items.extend(self.trait_aliases.iter().map(|x| x.clean(cx)));

        // determine if we should display the inner contents or
        // the outer `mod` item for the source code.
        let whence = {
            let cm = cx.sess().source_map();
            let outer = cm.lookup_char_pos(self.where_outer.lo());
            let inner = cm.lookup_char_pos(self.where_inner.lo());
            if outer.file.start_pos == inner.file.start_pos {
                // mod foo { ... }
                self.where_outer
            } else {
                // mod foo; (and a separate SourceFile for the contents)
                self.where_inner
            }
        };

        Item {
            name: Some(name),
            attrs,
            source: whence.clean(cx),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            inner: ModuleItem(Module {
               is_crate: self.is_crate,
               items,
            })
        }
    }
}

pub struct ListAttributesIter<'a> {
    attrs: slice::Iter<'a, ast::Attribute>,
    current_list: vec::IntoIter<ast::NestedMetaItem>,
    name: &'a str
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
    fn lists<'a>(&'a self, name: &'a str) -> ListAttributesIter<'a>;
}

impl AttributesExt for [ast::Attribute] {
    fn lists<'a>(&'a self, name: &'a str) -> ListAttributesIter<'a> {
        ListAttributesIter {
            attrs: self.iter(),
            current_list: Vec::new().into_iter(),
            name,
        }
    }
}

pub trait NestedAttributesExt {
    /// Returns `true` if the attribute list contains a specific `Word`
    fn has_word(self, word: &str) -> bool;
}

impl<I: IntoIterator<Item=ast::NestedMetaItem>> NestedAttributesExt for I {
    fn has_word(self, word: &str) -> bool {
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
#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub enum DocFragment {
    /// A doc fragment created from a `///` or `//!` doc comment.
    SugaredDoc(usize, syntax_pos::Span, String),
    /// A doc fragment created from a "raw" `#[doc=""]` attribute.
    RawDoc(usize, syntax_pos::Span, String),
    /// A doc fragment created from a `#[doc(include="filename")]` attribute. Contains both the
    /// given filename and the file contents.
    Include(usize, syntax_pos::Span, String, String),
}

impl DocFragment {
    pub fn as_str(&self) -> &str {
        match *self {
            DocFragment::SugaredDoc(_, _, ref s) => &s[..],
            DocFragment::RawDoc(_, _, ref s) => &s[..],
            DocFragment::Include(_, _, _, ref s) => &s[..],
        }
    }

    pub fn span(&self) -> syntax_pos::Span {
        match *self {
            DocFragment::SugaredDoc(_, span, _) |
                DocFragment::RawDoc(_, span, _) |
                DocFragment::Include(_, span, _, _) => span,
        }
    }
}

impl<'a> FromIterator<&'a DocFragment> for String {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a DocFragment>
    {
        iter.into_iter().fold(String::new(), |mut acc, frag| {
            if !acc.is_empty() {
                acc.push('\n');
            }
            match *frag {
                DocFragment::SugaredDoc(_, _, ref docs)
                    | DocFragment::RawDoc(_, _, ref docs)
                    | DocFragment::Include(_, _, _, ref docs) =>
                    acc.push_str(docs),
            }

            acc
        })
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Default)]
pub struct Attributes {
    pub doc_strings: Vec<DocFragment>,
    pub other_attrs: Vec<ast::Attribute>,
    pub cfg: Option<Arc<Cfg>>,
    pub span: Option<syntax_pos::Span>,
    /// map from Rust paths to resolved defs and potential URL fragments
    pub links: Vec<(String, Option<DefId>, Option<String>)>,
    pub inner_docs: bool,
}

impl Attributes {
    /// Extracts the content from an attribute `#[doc(cfg(content))]`.
    fn extract_cfg(mi: &ast::MetaItem) -> Option<&ast::MetaItem> {
        use syntax::ast::NestedMetaItemKind::MetaItem;

        if let ast::MetaItemKind::List(ref nmis) = mi.node {
            if nmis.len() == 1 {
                if let MetaItem(ref cfg_mi) = nmis[0].node {
                    if cfg_mi.check_name("cfg") {
                        if let ast::MetaItemKind::List(ref cfg_nmis) = cfg_mi.node {
                            if cfg_nmis.len() == 1 {
                                if let MetaItem(ref content_mi) = cfg_nmis[0].node {
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
    fn extract_include(mi: &ast::MetaItem)
        -> Option<(String, String)>
    {
        mi.meta_item_list().and_then(|list| {
            for meta in list {
                if meta.check_name("include") {
                    // the actual compiled `#[doc(include="filename")]` gets expanded to
                    // `#[doc(include(file="filename", contents="file contents")]` so we need to
                    // look for that instead
                    return meta.meta_item_list().and_then(|list| {
                        let mut filename: Option<String> = None;
                        let mut contents: Option<String> = None;

                        for it in list {
                            if it.check_name("file") {
                                if let Some(name) = it.value_str() {
                                    filename = Some(name.to_string());
                                }
                            } else if it.check_name("contents") {
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

    pub fn has_doc_flag(&self, flag: &str) -> bool {
        for attr in &self.other_attrs {
            if !attr.check_name("doc") { continue; }

            if let Some(items) = attr.meta_item_list() {
                if items.iter().filter_map(|i| i.meta_item()).any(|it| it.check_name(flag)) {
                    return true;
                }
            }
        }

        false
    }

    pub fn from_ast(diagnostic: &::errors::Handler,
                    attrs: &[ast::Attribute]) -> Attributes {
        let mut doc_strings = vec![];
        let mut sp = None;
        let mut cfg = Cfg::True;
        let mut doc_line = 0;

        let other_attrs = attrs.iter().filter_map(|attr| {
            attr.with_desugared_doc(|attr| {
                if attr.check_name("doc") {
                    if let Some(mi) = attr.meta() {
                        if let Some(value) = mi.value_str() {
                            // Extracted #[doc = "..."]
                            let value = value.to_string();
                            let line = doc_line;
                            doc_line += value.lines().count();

                            if attr.is_sugared_doc {
                                doc_strings.push(DocFragment::SugaredDoc(line, attr.span, value));
                            } else {
                                doc_strings.push(DocFragment::RawDoc(line, attr.span, value));
                            }

                            if sp.is_none() {
                                sp = Some(attr.span);
                            }
                            return None;
                        } else if let Some(cfg_mi) = Attributes::extract_cfg(&mi) {
                            // Extracted #[doc(cfg(...))]
                            match Cfg::parse(cfg_mi) {
                                Ok(new_cfg) => cfg &= new_cfg,
                                Err(e) => diagnostic.span_err(e.span, e.msg),
                            }
                            return None;
                        } else if let Some((filename, contents)) = Attributes::extract_include(&mi)
                        {
                            let line = doc_line;
                            doc_line += contents.lines().count();
                            doc_strings.push(DocFragment::Include(line,
                                                                  attr.span,
                                                                  filename,
                                                                  contents));
                        }
                    }
                }
                Some(attr.clone())
            })
        }).collect();

        // treat #[target_feature(enable = "feat")] attributes as if they were
        // #[doc(cfg(target_feature = "feat"))] attributes as well
        for attr in attrs.lists("target_feature") {
            if attr.check_name("enable") {
                if let Some(feat) = attr.value_str() {
                    let meta = attr::mk_name_value_item_str(Ident::from_str("target_feature"),
                                                            dummy_spanned(feat));
                    if let Ok(feat_cfg) = Cfg::parse(&meta) {
                        cfg &= feat_cfg;
                    }
                }
            }
        }

        let inner_docs = attrs.iter()
                              .filter(|a| a.check_name("doc"))
                              .next()
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
    pub fn doc_value<'a>(&'a self) -> Option<&'a str> {
        self.doc_strings.first().map(|s| s.as_str())
    }

    /// Finds all `doc` attributes as NameValues and returns their corresponding values, joined
    /// with newlines.
    pub fn collapsed_doc_value(&self) -> Option<String> {
        if !self.doc_strings.is_empty() {
            Some(self.doc_strings.iter().collect())
        } else {
            None
        }
    }

    /// Gets links as a vector
    ///
    /// Cache must be populated before call
    pub fn links(&self, krate: &CrateNum) -> Vec<(String, String)> {
        use html::format::href;
        self.links.iter().filter_map(|&(ref s, did, ref fragment)| {
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
                            Some(&(_, ref src, ExternalLocation::Local)) =>
                                src.to_str().expect("invalid file path"),
                            Some(&(_, _, ExternalLocation::Remote(ref s))) => s,
                            Some(&(_, _, ExternalLocation::Unknown)) | None =>
                                "https://doc.rust-lang.org/nightly",
                        };
                        // This is a primitive so the url is done "by hand".
                        Some((s.clone(),
                              format!("{}{}std/primitive.{}.html",
                                      url,
                                      if !url.ends_with('/') { "/" } else { "" },
                                      fragment)))
                    } else {
                        panic!("This isn't a primitive?!");
                    }
                }
            }
        }).collect()
    }
}

impl PartialEq for Attributes {
    fn eq(&self, rhs: &Self) -> bool {
        self.doc_strings == rhs.doc_strings &&
        self.cfg == rhs.cfg &&
        self.span == rhs.span &&
        self.links == rhs.links &&
        self.other_attrs.iter().map(|attr| attr.id).eq(rhs.other_attrs.iter().map(|attr| attr.id))
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
    fn lists<'a>(&'a self, name: &'a str) -> ListAttributesIter<'a> {
        self.other_attrs.lists(name)
    }
}

impl Clean<Attributes> for [ast::Attribute] {
    fn clean(&self, cx: &DocContext) -> Attributes {
        Attributes::from_ast(cx.sess().diagnostic(), self)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub enum GenericBound {
    TraitBound(PolyTrait, hir::TraitBoundModifier),
    Outlives(Lifetime),
}

impl GenericBound {
    fn maybe_sized(cx: &DocContext) -> GenericBound {
        let did = cx.tcx.require_lang_item(lang_items::SizedTraitLangItem);
        let empty = cx.tcx.intern_substs(&[]);
        let path = external_path(cx, &cx.tcx.item_name(did).as_str(),
            Some(did), false, vec![], empty);
        inline::record_extern_fqn(cx, did, TypeKind::Trait);
        GenericBound::TraitBound(PolyTrait {
            trait_: ResolvedPath {
                path,
                typarams: None,
                did,
                is_generic: false,
            },
            generic_params: Vec::new(),
        }, hir::TraitBoundModifier::Maybe)
    }

    fn is_sized_bound(&self, cx: &DocContext) -> bool {
        use rustc::hir::TraitBoundModifier as TBM;
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, TBM::None) = *self {
            if trait_.def_id() == cx.tcx.lang_items().sized_trait() {
                return true;
            }
        }
        false
    }

    fn get_poly_trait(&self) -> Option<PolyTrait> {
        if let GenericBound::TraitBound(ref p, _) = *self {
            return Some(p.clone())
        }
        None
    }

    fn get_trait_type(&self) -> Option<Type> {
        if let GenericBound::TraitBound(PolyTrait { ref trait_, .. }, _) = *self {
            return Some(trait_.clone());
        }
        None
    }
}

impl Clean<GenericBound> for hir::GenericBound {
    fn clean(&self, cx: &DocContext) -> GenericBound {
        match *self {
            hir::GenericBound::Outlives(lt) => GenericBound::Outlives(lt.clean(cx)),
            hir::GenericBound::Trait(ref t, modifier) => {
                GenericBound::TraitBound(t.clean(cx), modifier)
            }
        }
    }
}

fn external_generic_args(cx: &DocContext, trait_did: Option<DefId>, has_self: bool,
                        bindings: Vec<TypeBinding>, substs: &Substs) -> GenericArgs {
    let lifetimes = substs.regions().filter_map(|v| v.clean(cx)).collect();
    let types = substs.types().skip(has_self as usize).collect::<Vec<_>>();

    match trait_did {
        // Attempt to sugar an external path like Fn<(A, B,), C> to Fn(A, B) -> C
        Some(did) if cx.tcx.lang_items().fn_trait_kind(did).is_some() => {
            assert_eq!(types.len(), 1);
            let inputs = match types[0].sty {
                ty::Tuple(ref tys) => tys.iter().map(|t| t.clean(cx)).collect(),
                _ => {
                    return GenericArgs::AngleBracketed {
                        lifetimes,
                        types: types.clean(cx),
                        bindings,
                    }
                }
            };
            let output = None;
            // FIXME(#20299) return type comes from a projection now
            // match types[1].sty {
            //     ty::Tuple(ref v) if v.is_empty() => None, // -> ()
            //     _ => Some(types[1].clean(cx))
            // };
            GenericArgs::Parenthesized {
                inputs,
                output,
            }
        },
        _ => {
            GenericArgs::AngleBracketed {
                lifetimes,
                types: types.clean(cx),
                bindings,
            }
        }
    }
}

// trait_did should be set to a trait's DefId if called on a TraitRef, in order to sugar
// from Fn<(A, B,), C> to Fn(A, B) -> C
fn external_path(cx: &DocContext, name: &str, trait_did: Option<DefId>, has_self: bool,
                 bindings: Vec<TypeBinding>, substs: &Substs) -> Path {
    Path {
        global: false,
        def: Def::Err,
        segments: vec![PathSegment {
            name: name.to_string(),
            args: external_generic_args(cx, trait_did, has_self, bindings, substs)
        }],
    }
}

impl<'a, 'tcx> Clean<GenericBound> for (&'a ty::TraitRef<'tcx>, Vec<TypeBinding>) {
    fn clean(&self, cx: &DocContext) -> GenericBound {
        let (trait_ref, ref bounds) = *self;
        inline::record_extern_fqn(cx, trait_ref.def_id, TypeKind::Trait);
        let path = external_path(cx, &cx.tcx.item_name(trait_ref.def_id).as_str(),
                                 Some(trait_ref.def_id), true, bounds.clone(), trait_ref.substs);

        debug!("ty::TraitRef\n  subst: {:?}\n", trait_ref.substs);

        // collect any late bound regions
        let mut late_bounds = vec![];
        for ty_s in trait_ref.input_types().skip(1) {
            if let ty::Tuple(ts) = ty_s.sty {
                for &ty_s in ts {
                    if let ty::Ref(ref reg, _, _) = ty_s.sty {
                        if let &ty::RegionKind::ReLateBound(..) = *reg {
                            debug!("  hit an ReLateBound {:?}", reg);
                            if let Some(Lifetime(name)) = reg.clean(cx) {
                                late_bounds.push(GenericParamDef {
                                    name,
                                    kind: GenericParamDefKind::Lifetime,
                                });
                            }
                        }
                    }
                }
            }
        }

        GenericBound::TraitBound(
            PolyTrait {
                trait_: ResolvedPath {
                    path,
                    typarams: None,
                    did: trait_ref.def_id,
                    is_generic: false,
                },
                generic_params: late_bounds,
            },
            hir::TraitBoundModifier::None
        )
    }
}

impl<'tcx> Clean<GenericBound> for ty::TraitRef<'tcx> {
    fn clean(&self, cx: &DocContext) -> GenericBound {
        (self, vec![]).clean(cx)
    }
}

impl<'tcx> Clean<Option<Vec<GenericBound>>> for Substs<'tcx> {
    fn clean(&self, cx: &DocContext) -> Option<Vec<GenericBound>> {
        let mut v = Vec::new();
        v.extend(self.regions().filter_map(|r| r.clean(cx)).map(GenericBound::Outlives));
        v.extend(self.types().map(|t| GenericBound::TraitBound(PolyTrait {
            trait_: t.clean(cx),
            generic_params: Vec::new(),
        }, hir::TraitBoundModifier::None)));
        if !v.is_empty() {Some(v)} else {None}
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct Lifetime(String);

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

impl Clean<Lifetime> for hir::Lifetime {
    fn clean(&self, cx: &DocContext) -> Lifetime {
        if self.id != ast::DUMMY_NODE_ID {
            let def = cx.tcx.named_region(self.hir_id);
            match def {
                Some(rl::Region::EarlyBound(_, node_id, _)) |
                Some(rl::Region::LateBound(_, node_id, _)) |
                Some(rl::Region::Free(_, node_id)) => {
                    if let Some(lt) = cx.lt_substs.borrow().get(&node_id).cloned() {
                        return lt;
                    }
                }
                _ => {}
            }
        }
        Lifetime(self.name.ident().to_string())
    }
}

impl Clean<Lifetime> for hir::GenericParam {
    fn clean(&self, _: &DocContext) -> Lifetime {
        match self.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                if self.bounds.len() > 0 {
                    let mut bounds = self.bounds.iter().map(|bound| match bound {
                        hir::GenericBound::Outlives(lt) => lt,
                        _ => panic!(),
                    });
                    let name = bounds.next().expect("no more bounds").name.ident();
                    let mut s = format!("{}: {}", self.name.ident(), name);
                    for bound in bounds {
                        s.push_str(&format!(" + {}", bound.name.ident()));
                    }
                    Lifetime(s)
                } else {
                    Lifetime(self.name.ident().to_string())
                }
            }
            _ => panic!(),
        }
    }
}

impl<'tcx> Clean<Lifetime> for ty::GenericParamDef {
    fn clean(&self, _cx: &DocContext) -> Lifetime {
        Lifetime(self.name.to_string())
    }
}

impl Clean<Option<Lifetime>> for ty::RegionKind {
    fn clean(&self, cx: &DocContext) -> Option<Lifetime> {
        match *self {
            ty::ReStatic => Some(Lifetime::statik()),
            ty::ReLateBound(_, ty::BrNamed(_, name)) => Some(Lifetime(name.to_string())),
            ty::ReEarlyBound(ref data) => Some(Lifetime(data.name.clean(cx))),

            ty::ReLateBound(..) |
            ty::ReFree(..) |
            ty::ReScope(..) |
            ty::ReVar(..) |
            ty::RePlaceholder(..) |
            ty::ReEmpty |
            ty::ReClosureBound(_) |
            ty::ReErased => {
                debug!("Cannot clean region {:?}", self);
                None
            }
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub enum WherePredicate {
    BoundPredicate { ty: Type, bounds: Vec<GenericBound> },
    RegionPredicate { lifetime: Lifetime, bounds: Vec<GenericBound> },
    EqPredicate { lhs: Type, rhs: Type },
}

impl Clean<WherePredicate> for hir::WherePredicate {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        match *self {
            hir::WherePredicate::BoundPredicate(ref wbp) => {
                WherePredicate::BoundPredicate {
                    ty: wbp.bounded_ty.clean(cx),
                    bounds: wbp.bounds.clean(cx)
                }
            }

            hir::WherePredicate::RegionPredicate(ref wrp) => {
                WherePredicate::RegionPredicate {
                    lifetime: wrp.lifetime.clean(cx),
                    bounds: wrp.bounds.clean(cx)
                }
            }

            hir::WherePredicate::EqPredicate(ref wrp) => {
                WherePredicate::EqPredicate {
                    lhs: wrp.lhs_ty.clean(cx),
                    rhs: wrp.rhs_ty.clean(cx)
                }
            }
        }
    }
}

impl<'a> Clean<Option<WherePredicate>> for ty::Predicate<'a> {
    fn clean(&self, cx: &DocContext) -> Option<WherePredicate> {
        use rustc::ty::Predicate;

        match *self {
            Predicate::Trait(ref pred) => Some(pred.clean(cx)),
            Predicate::Subtype(ref pred) => Some(pred.clean(cx)),
            Predicate::RegionOutlives(ref pred) => pred.clean(cx),
            Predicate::TypeOutlives(ref pred) => pred.clean(cx),
            Predicate::Projection(ref pred) => Some(pred.clean(cx)),

            Predicate::WellFormed(..) |
            Predicate::ObjectSafe(..) |
            Predicate::ClosureKind(..) |
            Predicate::ConstEvaluatable(..) => panic!("not user writable"),
        }
    }
}

impl<'a> Clean<WherePredicate> for ty::TraitPredicate<'a> {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        WherePredicate::BoundPredicate {
            ty: self.trait_ref.self_ty().clean(cx),
            bounds: vec![self.trait_ref.clean(cx)]
        }
    }
}

impl<'tcx> Clean<WherePredicate> for ty::SubtypePredicate<'tcx> {
    fn clean(&self, _cx: &DocContext) -> WherePredicate {
        panic!("subtype predicates are an internal rustc artifact \
                and should not be seen by rustdoc")
    }
}

impl<'tcx> Clean<Option<WherePredicate>> for
    ty::OutlivesPredicate<ty::Region<'tcx>,ty::Region<'tcx>> {

    fn clean(&self, cx: &DocContext) -> Option<WherePredicate> {
        let ty::OutlivesPredicate(ref a, ref b) = *self;

        match (a, b) {
            (ty::ReEmpty, ty::ReEmpty) => {
                return None;
            },
            _ => {}
        }

        Some(WherePredicate::RegionPredicate {
            lifetime: a.clean(cx).expect("failed to clean lifetime"),
            bounds: vec![GenericBound::Outlives(b.clean(cx).expect("failed to clean bounds"))]
        })
    }
}

impl<'tcx> Clean<Option<WherePredicate>> for ty::OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>> {
    fn clean(&self, cx: &DocContext) -> Option<WherePredicate> {
        let ty::OutlivesPredicate(ref ty, ref lt) = *self;

        match lt {
            ty::ReEmpty => return None,
            _ => {}
        }

        Some(WherePredicate::BoundPredicate {
            ty: ty.clean(cx),
            bounds: vec![GenericBound::Outlives(lt.clean(cx).expect("failed to clean lifetimes"))]
        })
    }
}

impl<'tcx> Clean<WherePredicate> for ty::ProjectionPredicate<'tcx> {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        WherePredicate::EqPredicate {
            lhs: self.projection_ty.clean(cx),
            rhs: self.ty.clean(cx)
        }
    }
}

impl<'tcx> Clean<Type> for ty::ProjectionTy<'tcx> {
    fn clean(&self, cx: &DocContext) -> Type {
        let trait_ = match self.trait_ref(cx.tcx).clean(cx) {
            GenericBound::TraitBound(t, _) => t.trait_,
            GenericBound::Outlives(_) => panic!("cleaning a trait got a lifetime"),
        };
        Type::QPath {
            name: cx.tcx.associated_item(self.item_def_id).ident.name.clean(cx),
            self_type: box self.self_ty().clean(cx),
            trait_: box trait_
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub enum GenericParamDefKind {
    Lifetime,
    Type {
        did: DefId,
        bounds: Vec<GenericBound>,
        default: Option<Type>,
        synthetic: Option<hir::SyntheticTyParamKind>,
    },
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct GenericParamDef {
    pub name: String,

    pub kind: GenericParamDefKind,
}

impl GenericParamDef {
    pub fn is_synthetic_type_param(&self) -> bool {
        match self.kind {
            GenericParamDefKind::Lifetime => false,
            GenericParamDefKind::Type { ref synthetic, .. } => synthetic.is_some(),
        }
    }
}

impl<'tcx> Clean<GenericParamDef> for ty::GenericParamDef {
    fn clean(&self, cx: &DocContext) -> GenericParamDef {
        let (name, kind) = match self.kind {
            ty::GenericParamDefKind::Lifetime => {
                (self.name.to_string(), GenericParamDefKind::Lifetime)
            }
            ty::GenericParamDefKind::Type { has_default, .. } => {
                cx.renderinfo.borrow_mut().external_typarams
                             .insert(self.def_id, self.name.clean(cx));
                let default = if has_default {
                    Some(cx.tcx.type_of(self.def_id).clean(cx))
                } else {
                    None
                };
                (self.name.clean(cx), GenericParamDefKind::Type {
                    did: self.def_id,
                    bounds: vec![], // These are filled in from the where-clauses.
                    default,
                    synthetic: None,
                })
            }
        };

        GenericParamDef {
            name,
            kind,
        }
    }
}

impl Clean<GenericParamDef> for hir::GenericParam {
    fn clean(&self, cx: &DocContext) -> GenericParamDef {
        let (name, kind) = match self.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                let name = if self.bounds.len() > 0 {
                    let mut bounds = self.bounds.iter().map(|bound| match bound {
                        hir::GenericBound::Outlives(lt) => lt,
                        _ => panic!(),
                    });
                    let name = bounds.next().expect("no more bounds").name.ident();
                    let mut s = format!("{}: {}", self.name.ident(), name);
                    for bound in bounds {
                        s.push_str(&format!(" + {}", bound.name.ident()));
                    }
                    s
                } else {
                    self.name.ident().to_string()
                };
                (name, GenericParamDefKind::Lifetime)
            }
            hir::GenericParamKind::Type { ref default, synthetic, .. } => {
                (self.name.ident().name.clean(cx), GenericParamDefKind::Type {
                    did: cx.tcx.hir().local_def_id(self.id),
                    bounds: self.bounds.clean(cx),
                    default: default.clean(cx),
                    synthetic: synthetic,
                })
            }
        };

        GenericParamDef {
            name,
            kind,
        }
    }
}

// maybe use a Generic enum and use Vec<Generic>?
#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Default, Hash)]
pub struct Generics {
    pub params: Vec<GenericParamDef>,
    pub where_predicates: Vec<WherePredicate>,
}

impl Clean<Generics> for hir::Generics {
    fn clean(&self, cx: &DocContext) -> Generics {
        // Synthetic type-parameters are inserted after normal ones.
        // In order for normal parameters to be able to refer to synthetic ones,
        // scans them first.
        fn is_impl_trait(param: &hir::GenericParam) -> bool {
            match param.kind {
                hir::GenericParamKind::Type { synthetic, .. } => {
                    synthetic == Some(hir::SyntheticTyParamKind::ImplTrait)
                }
                _ => false,
            }
        }
        let impl_trait_params = self.params
            .iter()
            .filter(|param| is_impl_trait(param))
            .map(|param| {
                let param: GenericParamDef = param.clean(cx);
                match param.kind {
                    GenericParamDefKind::Lifetime => unreachable!(),
                    GenericParamDefKind::Type { did, ref bounds, .. } => {
                        cx.impl_trait_bounds.borrow_mut().insert(did, bounds.clone());
                    }
                }
                param
            })
            .collect::<Vec<_>>();

        let mut params = Vec::with_capacity(self.params.len());
        for p in self.params.iter().filter(|p| !is_impl_trait(p)) {
            let p = p.clean(cx);
            params.push(p);
        }
        params.extend(impl_trait_params);

        let mut generics = Generics {
            params,
            where_predicates: self.where_clause.predicates.clean(cx),
        };

        // Some duplicates are generated for ?Sized bounds between type params and where
        // predicates. The point in here is to move the bounds definitions from type params
        // to where predicates when such cases occur.
        for where_pred in &mut generics.where_predicates {
            match *where_pred {
                WherePredicate::BoundPredicate { ty: Generic(ref name), ref mut bounds } => {
                    if bounds.is_empty() {
                        for param in &mut generics.params {
                            match param.kind {
                                GenericParamDefKind::Lifetime => {}
                                GenericParamDefKind::Type { bounds: ref mut ty_bounds, .. } => {
                                    if &param.name == name {
                                        mem::swap(bounds, ty_bounds);
                                        break
                                    }
                                }
                            }
                        }
                    }
                }
                _ => continue,
            }
        }
        generics
    }
}

impl<'a, 'tcx> Clean<Generics> for (&'a ty::Generics,
                                    &'a Lrc<ty::GenericPredicates<'tcx>>) {
    fn clean(&self, cx: &DocContext) -> Generics {
        use self::WherePredicate as WP;

        let (gens, preds) = *self;

        // Bounds in the type_params and lifetimes fields are repeated in the
        // predicates field (see rustc_typeck::collect::ty_generics), so remove
        // them.
        let stripped_typarams = gens.params.iter().filter_map(|param| match param.kind {
            ty::GenericParamDefKind::Lifetime => None,
            ty::GenericParamDefKind::Type { .. } => {
                if param.name == keywords::SelfUpper.name().as_str() {
                    assert_eq!(param.index, 0);
                    return None;
                }
                Some(param.clean(cx))
            }
        }).collect::<Vec<GenericParamDef>>();

        let mut where_predicates = preds.predicates.iter()
            .flat_map(|(p, _)| p.clean(cx))
            .collect::<Vec<_>>();

        // Type parameters and have a Sized bound by default unless removed with
        // ?Sized. Scan through the predicates and mark any type parameter with
        // a Sized bound, removing the bounds as we find them.
        //
        // Note that associated types also have a sized bound by default, but we
        // don't actually know the set of associated types right here so that's
        // handled in cleaning associated types
        let mut sized_params = FxHashSet::default();
        where_predicates.retain(|pred| {
            match *pred {
                WP::BoundPredicate { ty: Generic(ref g), ref bounds } => {
                    if bounds.iter().any(|b| b.is_sized_bound(cx)) {
                        sized_params.insert(g.clone());
                        false
                    } else {
                        true
                    }
                }
                _ => true,
            }
        });

        // Run through the type parameters again and insert a ?Sized
        // unbound for any we didn't find to be Sized.
        for tp in &stripped_typarams {
            if !sized_params.contains(&tp.name) {
                where_predicates.push(WP::BoundPredicate {
                    ty: Type::Generic(tp.name.clone()),
                    bounds: vec![GenericBound::maybe_sized(cx)],
                })
            }
        }

        // It would be nice to collect all of the bounds on a type and recombine
        // them if possible, to avoid e.g., `where T: Foo, T: Bar, T: Sized, T: 'a`
        // and instead see `where T: Foo + Bar + Sized + 'a`

        Generics {
            params: gens.params
                        .iter()
                        .flat_map(|param| match param.kind {
                            ty::GenericParamDefKind::Lifetime => Some(param.clean(cx)),
                            ty::GenericParamDefKind::Type { .. } => None,
                        }).chain(simplify::ty_params(stripped_typarams).into_iter())
                        .collect(),
            where_predicates: simplify::where_clauses(cx, where_predicates),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Method {
    pub generics: Generics,
    pub decl: FnDecl,
    pub header: hir::FnHeader,
}

impl<'a> Clean<Method> for (&'a hir::MethodSig, &'a hir::Generics, hir::BodyId) {
    fn clean(&self, cx: &DocContext) -> Method {
        let (generics, decl) = enter_impl_trait(cx, || {
            (self.1.clean(cx), (&*self.0.decl, self.2).clean(cx))
        });
        Method {
            decl,
            generics,
            header: self.0.header,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct TyMethod {
    pub header: hir::FnHeader,
    pub decl: FnDecl,
    pub generics: Generics,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Function {
    pub decl: FnDecl,
    pub generics: Generics,
    pub header: hir::FnHeader,
}

impl Clean<Item> for doctree::Function {
    fn clean(&self, cx: &DocContext) -> Item {
        let (generics, decl) = enter_impl_trait(cx, || {
            (self.generics.clean(cx), (&self.decl, self.body).clean(cx))
        });

        let did = cx.tcx.hir().local_def_id(self.id);
        let constness = if cx.tcx.is_min_const_fn(did) {
            hir::Constness::Const
        } else {
            hir::Constness::NotConst
        };
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            def_id: did,
            inner: FunctionItem(Function {
                decl,
                generics,
                header: hir::FnHeader { constness, ..self.header },
            }),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct FnDecl {
    pub inputs: Arguments,
    pub output: FunctionRetTy,
    pub variadic: bool,
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
    pub fn sugared_async_return_type(&self) -> FunctionRetTy {
        match &self.output {
            FunctionRetTy::Return(Type::ImplTrait(bounds)) => {
                match &bounds[0] {
                    GenericBound::TraitBound(PolyTrait { trait_, .. }, ..) => {
                        let bindings = trait_.bindings().unwrap();
                        FunctionRetTy::Return(bindings[0].ty.clone())
                    }
                    _ => panic!("unexpected desugaring of async function"),
                }
            }
            _ => panic!("unexpected desugaring of async function"),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct Arguments {
    pub values: Vec<Argument>,
}

impl<'a> Clean<Arguments> for (&'a [hir::Ty], &'a [ast::Ident]) {
    fn clean(&self, cx: &DocContext) -> Arguments {
        Arguments {
            values: self.0.iter().enumerate().map(|(i, ty)| {
                let mut name = self.1.get(i).map(|ident| ident.to_string())
                                            .unwrap_or(String::new());
                if name.is_empty() {
                    name = "_".to_string();
                }
                Argument {
                    name,
                    type_: ty.clean(cx),
                }
            }).collect()
        }
    }
}

impl<'a> Clean<Arguments> for (&'a [hir::Ty], hir::BodyId) {
    fn clean(&self, cx: &DocContext) -> Arguments {
        let body = cx.tcx.hir().body(self.1);

        Arguments {
            values: self.0.iter().enumerate().map(|(i, ty)| {
                Argument {
                    name: name_from_pat(&body.arguments[i].pat),
                    type_: ty.clean(cx),
                }
            }).collect()
        }
    }
}

impl<'a, A: Copy> Clean<FnDecl> for (&'a hir::FnDecl, A)
    where (&'a [hir::Ty], A): Clean<Arguments>
{
    fn clean(&self, cx: &DocContext) -> FnDecl {
        FnDecl {
            inputs: (&self.0.inputs[..], self.1).clean(cx),
            output: self.0.output.clean(cx),
            variadic: self.0.variadic,
            attrs: Attributes::default()
        }
    }
}

impl<'a, 'tcx> Clean<FnDecl> for (DefId, ty::PolyFnSig<'tcx>) {
    fn clean(&self, cx: &DocContext) -> FnDecl {
        let (did, sig) = *self;
        let mut names = if cx.tcx.hir().as_local_node_id(did).is_some() {
            vec![].into_iter()
        } else {
            cx.tcx.fn_arg_names(did).into_iter()
        };

        FnDecl {
            output: Return(sig.skip_binder().output().clean(cx)),
            attrs: Attributes::default(),
            variadic: sig.skip_binder().variadic,
            inputs: Arguments {
                values: sig.skip_binder().inputs().iter().map(|t| {
                    Argument {
                        type_: t.clean(cx),
                        name: names.next().map_or(String::new(), |name| name.to_string()),
                    }
                }).collect(),
            },
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct Argument {
    pub type_: Type,
    pub name: String,
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
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
            BorrowedRef{ref lifetime, mutability, ref type_} if type_.is_self_type() => {
                Some(SelfBorrowed(lifetime.clone(), mutability))
            }
            _ => Some(SelfExplicit(self.type_.clone()))
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub enum FunctionRetTy {
    Return(Type),
    DefaultReturn,
}

impl Clean<FunctionRetTy> for hir::FunctionRetTy {
    fn clean(&self, cx: &DocContext) -> FunctionRetTy {
        match *self {
            hir::Return(ref typ) => Return(typ.clean(cx)),
            hir::DefaultReturn(..) => DefaultReturn,
        }
    }
}

impl GetDefId for FunctionRetTy {
    fn def_id(&self) -> Option<DefId> {
        match *self {
            Return(ref ty) => ty.def_id(),
            DefaultReturn => None,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Trait {
    pub auto: bool,
    pub unsafety: hir::Unsafety,
    pub items: Vec<Item>,
    pub generics: Generics,
    pub bounds: Vec<GenericBound>,
    pub is_spotlight: bool,
    pub is_auto: bool,
}

impl Clean<Item> for doctree::Trait {
    fn clean(&self, cx: &DocContext) -> Item {
        let attrs = self.attrs.clean(cx);
        let is_spotlight = attrs.has_doc_flag("spotlight");
        Item {
            name: Some(self.name.clean(cx)),
            attrs: attrs,
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: TraitItem(Trait {
                auto: self.is_auto.clean(cx),
                unsafety: self.unsafety,
                items: self.items.clean(cx),
                generics: self.generics.clean(cx),
                bounds: self.bounds.clean(cx),
                is_spotlight,
                is_auto: self.is_auto.clean(cx),
            }),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct TraitAlias {
    pub generics: Generics,
    pub bounds: Vec<GenericBound>,
}

impl Clean<Item> for doctree::TraitAlias {
    fn clean(&self, cx: &DocContext) -> Item {
        let attrs = self.attrs.clean(cx);
        Item {
            name: Some(self.name.clean(cx)),
            attrs,
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: TraitAliasItem(TraitAlias {
                generics: self.generics.clean(cx),
                bounds: self.bounds.clean(cx),
            }),
        }
    }
}

impl Clean<bool> for hir::IsAuto {
    fn clean(&self, _: &DocContext) -> bool {
        match *self {
            hir::IsAuto::Yes => true,
            hir::IsAuto::No => false,
        }
    }
}

impl Clean<Type> for hir::TraitRef {
    fn clean(&self, cx: &DocContext) -> Type {
        resolve_type(cx, self.path.clean(cx), self.ref_id)
    }
}

impl Clean<PolyTrait> for hir::PolyTraitRef {
    fn clean(&self, cx: &DocContext) -> PolyTrait {
        PolyTrait {
            trait_: self.trait_ref.clean(cx),
            generic_params: self.bound_generic_params.clean(cx)
        }
    }
}

impl Clean<Item> for hir::TraitItem {
    fn clean(&self, cx: &DocContext) -> Item {
        let inner = match self.node {
            hir::TraitItemKind::Const(ref ty, default) => {
                AssociatedConstItem(ty.clean(cx),
                                    default.map(|e| print_const_expr(cx, e)))
            }
            hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Provided(body)) => {
                MethodItem((sig, &self.generics, body).clean(cx))
            }
            hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Required(ref names)) => {
                let (generics, decl) = enter_impl_trait(cx, || {
                    (self.generics.clean(cx), (&*sig.decl, &names[..]).clean(cx))
                });
                TyMethodItem(TyMethod {
                    header: sig.header,
                    decl,
                    generics,
                })
            }
            hir::TraitItemKind::Type(ref bounds, ref default) => {
                AssociatedTypeItem(bounds.clean(cx), default.clean(cx))
            }
        };
        Item {
            name: Some(self.ident.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: None,
            stability: get_stability(cx, cx.tcx.hir().local_def_id(self.id)),
            deprecation: get_deprecation(cx, cx.tcx.hir().local_def_id(self.id)),
            inner,
        }
    }
}

impl Clean<Item> for hir::ImplItem {
    fn clean(&self, cx: &DocContext) -> Item {
        let inner = match self.node {
            hir::ImplItemKind::Const(ref ty, expr) => {
                AssociatedConstItem(ty.clean(cx),
                                    Some(print_const_expr(cx, expr)))
            }
            hir::ImplItemKind::Method(ref sig, body) => {
                MethodItem((sig, &self.generics, body).clean(cx))
            }
            hir::ImplItemKind::Type(ref ty) => TypedefItem(Typedef {
                type_: ty.clean(cx),
                generics: Generics::default(),
            }, true),
            hir::ImplItemKind::Existential(ref bounds) => ExistentialItem(Existential {
                bounds: bounds.clean(cx),
                generics: Generics::default(),
            }, true),
        };
        Item {
            name: Some(self.ident.name.clean(cx)),
            source: self.span.clean(cx),
            attrs: self.attrs.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, cx.tcx.hir().local_def_id(self.id)),
            deprecation: get_deprecation(cx, cx.tcx.hir().local_def_id(self.id)),
            inner,
        }
    }
}

impl<'tcx> Clean<Item> for ty::AssociatedItem {
    fn clean(&self, cx: &DocContext) -> Item {
        let inner = match self.kind {
            ty::AssociatedKind::Const => {
                let ty = cx.tcx.type_of(self.def_id);
                let default = if self.defaultness.has_value() {
                    Some(inline::print_inlined_const(cx, self.def_id))
                } else {
                    None
                };
                AssociatedConstItem(ty.clean(cx), default)
            }
            ty::AssociatedKind::Method => {
                let generics = (cx.tcx.generics_of(self.def_id),
                                &cx.tcx.predicates_of(self.def_id)).clean(cx);
                let sig = cx.tcx.fn_sig(self.def_id);
                let mut decl = (self.def_id, sig).clean(cx);

                if self.method_has_self_argument {
                    let self_ty = match self.container {
                        ty::ImplContainer(def_id) => {
                            cx.tcx.type_of(def_id)
                        }
                        ty::TraitContainer(_) => cx.tcx.mk_self_type()
                    };
                    let self_arg_ty = *sig.input(0).skip_binder();
                    if self_arg_ty == self_ty {
                        decl.inputs.values[0].type_ = Generic(String::from("Self"));
                    } else if let ty::Ref(_, ty, _) = self_arg_ty.sty {
                        if ty == self_ty {
                            match decl.inputs.values[0].type_ {
                                BorrowedRef{ref mut type_, ..} => {
                                    **type_ = Generic(String::from("Self"))
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }

                let provided = match self.container {
                    ty::ImplContainer(_) => true,
                    ty::TraitContainer(_) => self.defaultness.has_value()
                };
                if provided {
                    let constness = if cx.tcx.is_min_const_fn(self.def_id) {
                        hir::Constness::Const
                    } else {
                        hir::Constness::NotConst
                    };
                    MethodItem(Method {
                        generics,
                        decl,
                        header: hir::FnHeader {
                            unsafety: sig.unsafety(),
                            abi: sig.abi(),
                            constness,
                            asyncness: hir::IsAsync::NotAsync,
                        }
                    })
                } else {
                    TyMethodItem(TyMethod {
                        generics,
                        decl,
                        header: hir::FnHeader {
                            unsafety: sig.unsafety(),
                            abi: sig.abi(),
                            constness: hir::Constness::NotConst,
                            asyncness: hir::IsAsync::NotAsync,
                        }
                    })
                }
            }
            ty::AssociatedKind::Type => {
                let my_name = self.ident.name.clean(cx);

                if let ty::TraitContainer(did) = self.container {
                    // When loading a cross-crate associated type, the bounds for this type
                    // are actually located on the trait/impl itself, so we need to load
                    // all of the generics from there and then look for bounds that are
                    // applied to this associated type in question.
                    let predicates = cx.tcx.predicates_of(did);
                    let generics = (cx.tcx.generics_of(did), &predicates).clean(cx);
                    let mut bounds = generics.where_predicates.iter().filter_map(|pred| {
                        let (name, self_type, trait_, bounds) = match *pred {
                            WherePredicate::BoundPredicate {
                                ty: QPath { ref name, ref self_type, ref trait_ },
                                ref bounds
                            } => (name, self_type, trait_, bounds),
                            _ => return None,
                        };
                        if *name != my_name { return None }
                        match **trait_ {
                            ResolvedPath { did, .. } if did == self.container.id() => {}
                            _ => return None,
                        }
                        match **self_type {
                            Generic(ref s) if *s == "Self" => {}
                            _ => return None,
                        }
                        Some(bounds)
                    }).flat_map(|i| i.iter().cloned()).collect::<Vec<_>>();
                    // Our Sized/?Sized bound didn't get handled when creating the generics
                    // because we didn't actually get our whole set of bounds until just now
                    // (some of them may have come from the trait). If we do have a sized
                    // bound, we remove it, and if we don't then we add the `?Sized` bound
                    // at the end.
                    match bounds.iter().position(|b| b.is_sized_bound(cx)) {
                        Some(i) => { bounds.remove(i); }
                        None => bounds.push(GenericBound::maybe_sized(cx)),
                    }

                    let ty = if self.defaultness.has_value() {
                        Some(cx.tcx.type_of(self.def_id))
                    } else {
                        None
                    };

                    AssociatedTypeItem(bounds, ty.clean(cx))
                } else {
                    TypedefItem(Typedef {
                        type_: cx.tcx.type_of(self.def_id).clean(cx),
                        generics: Generics {
                            params: Vec::new(),
                            where_predicates: Vec::new(),
                        },
                    }, true)
                }
            }
            ty::AssociatedKind::Existential => unimplemented!(),
        };

        let visibility = match self.container {
            ty::ImplContainer(_) => self.vis.clean(cx),
            ty::TraitContainer(_) => None,
        };

        Item {
            name: Some(self.ident.name.clean(cx)),
            visibility,
            stability: get_stability(cx, self.def_id),
            deprecation: get_deprecation(cx, self.def_id),
            def_id: self.def_id,
            attrs: inline::load_attrs(cx, self.def_id),
            source: cx.tcx.def_span(self.def_id).clean(cx),
            inner,
        }
    }
}

/// A trait reference, which may have higher ranked lifetimes.
#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct PolyTrait {
    pub trait_: Type,
    pub generic_params: Vec<GenericParamDef>,
}

/// A representation of a Type suitable for hyperlinking purposes. Ideally one can get the original
/// type out of the AST/TyCtxt given one of these, if more information is needed. Most importantly
/// it does not preserve mutability or boxes.
#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub enum Type {
    /// Structs/enums/traits (most that'd be an `hir::TyKind::Path`).
    ResolvedPath {
        path: Path,
        typarams: Option<Vec<GenericBound>>,
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
    /// extern "ABI" fn
    BareFunction(Box<BareFunctionDecl>),
    Tuple(Vec<Type>),
    Slice(Box<Type>),
    Array(Box<Type>, String),
    Never,
    Unique(Box<Type>),
    RawPointer(Mutability, Box<Type>),
    BorrowedRef {
        lifetime: Option<Lifetime>,
        mutability: Mutability,
        type_: Box<Type>,
    },

    // <Type as Trait>::Name
    QPath {
        name: String,
        self_type: Box<Type>,
        trait_: Box<Type>
    },

    // _
    Infer,

    // impl TraitA+TraitB
    ImplTrait(Vec<GenericBound>),
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Copy, Debug)]
pub enum PrimitiveType {
    Isize, I8, I16, I32, I64, I128,
    Usize, U8, U16, U32, U64, U128,
    F32, F64,
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

#[derive(Clone, RustcEncodable, RustcDecodable, Copy, Debug)]
pub enum TypeKind {
    Enum,
    Function,
    Module,
    Const,
    Static,
    Struct,
    Union,
    Trait,
    Variant,
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
            Primitive(p) | BorrowedRef { type_: box Primitive(p), ..} => Some(p),
            Slice(..) | BorrowedRef { type_: box Slice(..), .. } => Some(PrimitiveType::Slice),
            Array(..) | BorrowedRef { type_: box Array(..), .. } => Some(PrimitiveType::Array),
            Tuple(ref tys) => if tys.is_empty() {
                Some(PrimitiveType::Unit)
            } else {
                Some(PrimitiveType::Tuple)
            },
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
            _ => false
        }
    }

    pub fn generics(&self) -> Option<&[Type]> {
        match *self {
            ResolvedPath { ref path, .. } => {
                path.segments.last().and_then(|seg| {
                    if let GenericArgs::AngleBracketed { ref types, .. } = seg.args {
                        Some(&**types)
                    } else {
                        None
                    }
                })
            }
            _ => None,
        }
    }

    pub fn bindings(&self) -> Option<&[TypeBinding]> {
        match *self {
            ResolvedPath { ref path, .. } => {
                path.segments.last().and_then(|seg| {
                    if let GenericArgs::AngleBracketed { ref bindings, .. } = seg.args {
                        Some(&**bindings)
                    } else {
                        None
                    }
                })
            }
            _ => None
        }
    }
}

impl GetDefId for Type {
    fn def_id(&self) -> Option<DefId> {
        match *self {
            ResolvedPath { did, .. } => Some(did),
            Primitive(p) => ::html::render::cache().primitive_locations.get(&p).cloned(),
            BorrowedRef { type_: box Generic(..), .. } =>
                Primitive(PrimitiveType::Reference).def_id(),
            BorrowedRef { ref type_, .. } => type_.def_id(),
            Tuple(ref tys) => if tys.is_empty() {
                Primitive(PrimitiveType::Unit).def_id()
            } else {
                Primitive(PrimitiveType::Tuple).def_id()
            },
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
    fn from_str(s: &str) -> Option<PrimitiveType> {
        match s {
            "isize" => Some(PrimitiveType::Isize),
            "i8" => Some(PrimitiveType::I8),
            "i16" => Some(PrimitiveType::I16),
            "i32" => Some(PrimitiveType::I32),
            "i64" => Some(PrimitiveType::I64),
            "i128" => Some(PrimitiveType::I128),
            "usize" => Some(PrimitiveType::Usize),
            "u8" => Some(PrimitiveType::U8),
            "u16" => Some(PrimitiveType::U16),
            "u32" => Some(PrimitiveType::U32),
            "u64" => Some(PrimitiveType::U64),
            "u128" => Some(PrimitiveType::U128),
            "bool" => Some(PrimitiveType::Bool),
            "char" => Some(PrimitiveType::Char),
            "str" => Some(PrimitiveType::Str),
            "f32" => Some(PrimitiveType::F32),
            "f64" => Some(PrimitiveType::F64),
            "array" => Some(PrimitiveType::Array),
            "slice" => Some(PrimitiveType::Slice),
            "tuple" => Some(PrimitiveType::Tuple),
            "unit" => Some(PrimitiveType::Unit),
            "pointer" => Some(PrimitiveType::RawPointer),
            "reference" => Some(PrimitiveType::Reference),
            "fn" => Some(PrimitiveType::Fn),
            "never" => Some(PrimitiveType::Never),
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

impl Clean<Type> for hir::Ty {
    fn clean(&self, cx: &DocContext) -> Type {
        use rustc::hir::*;

        match self.node {
            TyKind::Never => Never,
            TyKind::Ptr(ref m) => RawPointer(m.mutbl.clean(cx), box m.ty.clean(cx)),
            TyKind::Rptr(ref l, ref m) => {
                let lifetime = if l.is_elided() {
                    None
                } else {
                    Some(l.clean(cx))
                };
                BorrowedRef {lifetime: lifetime, mutability: m.mutbl.clean(cx),
                             type_: box m.ty.clean(cx)}
            }
            TyKind::Slice(ref ty) => Slice(box ty.clean(cx)),
            TyKind::Array(ref ty, ref length) => {
                let def_id = cx.tcx.hir().local_def_id(length.id);
                let param_env = cx.tcx.param_env(def_id);
                let substs = Substs::identity_for_item(cx.tcx, def_id);
                let cid = GlobalId {
                    instance: ty::Instance::new(def_id, substs),
                    promoted: None
                };
                let length = match cx.tcx.const_eval(param_env.and(cid)) {
                    Ok(length) => print_const(cx, ty::LazyConst::Evaluated(length)),
                    Err(_) => "_".to_string(),
                };
                Array(box ty.clean(cx), length)
            },
            TyKind::Tup(ref tys) => Tuple(tys.clean(cx)),
            TyKind::Def(item_id, _) => {
                let item = cx.tcx.hir().expect_item(item_id.id);
                if let hir::ItemKind::Existential(ref ty) = item.node {
                    ImplTrait(ty.bounds.clean(cx))
                } else {
                    unreachable!()
                }
            }
            TyKind::Path(hir::QPath::Resolved(None, ref path)) => {
                if let Some(new_ty) = cx.ty_substs.borrow().get(&path.def).cloned() {
                    return new_ty;
                }

                if let Def::TyParam(did) = path.def {
                    if let Some(bounds) = cx.impl_trait_bounds.borrow_mut().remove(&did) {
                        return ImplTrait(bounds);
                    }
                }

                let mut alias = None;
                if let Def::TyAlias(def_id) = path.def {
                    // Substitute private type aliases
                    if let Some(node_id) = cx.tcx.hir().as_local_node_id(def_id) {
                        if !cx.renderinfo.borrow().access_levels.is_exported(def_id) {
                            alias = Some(&cx.tcx.hir().expect_item(node_id).node);
                        }
                    }
                };

                if let Some(&hir::ItemKind::Ty(ref ty, ref generics)) = alias {
                    let provided_params = &path.segments.last().expect("segments were empty");
                    let mut ty_substs = FxHashMap::default();
                    let mut lt_substs = FxHashMap::default();
                    provided_params.with_generic_args(|generic_args| {
                        let mut indices: GenericParamCount = Default::default();
                        for param in generics.params.iter() {
                            match param.kind {
                                hir::GenericParamKind::Lifetime { .. } => {
                                    let mut j = 0;
                                    let lifetime = generic_args.args.iter().find_map(|arg| {
                                        match arg {
                                            GenericArg::Lifetime(lt) => {
                                                if indices.lifetimes == j {
                                                    return Some(lt);
                                                }
                                                j += 1;
                                                None
                                            }
                                            _ => None,
                                        }
                                    });
                                    if let Some(lt) = lifetime.cloned() {
                                        if !lt.is_elided() {
                                            let lt_def_id =
                                                cx.tcx.hir().local_def_id(param.id);
                                            lt_substs.insert(lt_def_id, lt.clean(cx));
                                        }
                                    }
                                    indices.lifetimes += 1;
                                }
                                hir::GenericParamKind::Type { ref default, .. } => {
                                    let ty_param_def =
                                        Def::TyParam(cx.tcx.hir().local_def_id(param.id));
                                    let mut j = 0;
                                    let type_ = generic_args.args.iter().find_map(|arg| {
                                        match arg {
                                            GenericArg::Type(ty) => {
                                                if indices.types == j {
                                                    return Some(ty);
                                                }
                                                j += 1;
                                                None
                                            }
                                            _ => None,
                                        }
                                    });
                                    if let Some(ty) = type_.cloned() {
                                        ty_substs.insert(ty_param_def, ty.clean(cx));
                                    } else if let Some(default) = default.clone() {
                                        ty_substs.insert(ty_param_def,
                                                         default.into_inner().clean(cx));
                                    }
                                    indices.types += 1;
                                }
                            }
                        }
                    });
                    return cx.enter_alias(ty_substs, lt_substs, || ty.clean(cx));
                }
                resolve_type(cx, path.clean(cx), self.id)
            }
            TyKind::Path(hir::QPath::Resolved(Some(ref qself), ref p)) => {
                let mut segments: Vec<_> = p.segments.clone().into();
                segments.pop();
                let trait_path = hir::Path {
                    span: p.span,
                    def: Def::Trait(cx.tcx.associated_item(p.def.def_id()).container.id()),
                    segments: segments.into(),
                };
                Type::QPath {
                    name: p.segments.last().expect("segments were empty").ident.name.clean(cx),
                    self_type: box qself.clean(cx),
                    trait_: box resolve_type(cx, trait_path.clean(cx), self.id)
                }
            }
            TyKind::Path(hir::QPath::TypeRelative(ref qself, ref segment)) => {
                let mut def = Def::Err;
                let ty = hir_ty_to_ty(cx.tcx, self);
                if let ty::Projection(proj) = ty.sty {
                    def = Def::Trait(proj.trait_ref(cx.tcx).def_id);
                }
                let trait_path = hir::Path {
                    span: self.span,
                    def,
                    segments: vec![].into(),
                };
                Type::QPath {
                    name: segment.ident.name.clean(cx),
                    self_type: box qself.clean(cx),
                    trait_: box resolve_type(cx, trait_path.clean(cx), self.id)
                }
            }
            TyKind::TraitObject(ref bounds, ref lifetime) => {
                match bounds[0].clean(cx).trait_ {
                    ResolvedPath { path, typarams: None, did, is_generic } => {
                        let mut bounds: Vec<self::GenericBound> = bounds[1..].iter().map(|bound| {
                            self::GenericBound::TraitBound(bound.clean(cx),
                                                           hir::TraitBoundModifier::None)
                        }).collect();
                        if !lifetime.is_elided() {
                            bounds.push(self::GenericBound::Outlives(lifetime.clean(cx)));
                        }
                        ResolvedPath { path, typarams: Some(bounds), did, is_generic, }
                    }
                    _ => Infer // shouldn't happen
                }
            }
            TyKind::BareFn(ref barefn) => BareFunction(box barefn.clean(cx)),
            TyKind::Infer | TyKind::Err => Infer,
            TyKind::Typeof(..) => panic!("Unimplemented type {:?}", self.node),
        }
    }
}

impl<'tcx> Clean<Type> for Ty<'tcx> {
    fn clean(&self, cx: &DocContext) -> Type {
        match self.sty {
            ty::Never => Never,
            ty::Bool => Primitive(PrimitiveType::Bool),
            ty::Char => Primitive(PrimitiveType::Char),
            ty::Int(int_ty) => Primitive(int_ty.into()),
            ty::Uint(uint_ty) => Primitive(uint_ty.into()),
            ty::Float(float_ty) => Primitive(float_ty.into()),
            ty::Str => Primitive(PrimitiveType::Str),
            ty::Slice(ty) => Slice(box ty.clean(cx)),
            ty::Array(ty, n) => {
                let mut n = *cx.tcx.lift(&n).expect("array lift failed");
                if let ty::LazyConst::Unevaluated(def_id, substs) = n {
                    let param_env = cx.tcx.param_env(def_id);
                    let cid = GlobalId {
                        instance: ty::Instance::new(def_id, substs),
                        promoted: None
                    };
                    if let Ok(new_n) = cx.tcx.const_eval(param_env.and(cid)) {
                        n = ty::LazyConst::Evaluated(new_n);
                    }
                };
                let n = print_const(cx, n);
                Array(box ty.clean(cx), n)
            }
            ty::RawPtr(mt) => RawPointer(mt.mutbl.clean(cx), box mt.ty.clean(cx)),
            ty::Ref(r, ty, mutbl) => BorrowedRef {
                lifetime: r.clean(cx),
                mutability: mutbl.clean(cx),
                type_: box ty.clean(cx),
            },
            ty::FnDef(..) |
            ty::FnPtr(_) => {
                let ty = cx.tcx.lift(self).expect("FnPtr lift failed");
                let sig = ty.fn_sig(cx.tcx);
                BareFunction(box BareFunctionDecl {
                    unsafety: sig.unsafety(),
                    generic_params: Vec::new(),
                    decl: (cx.tcx.hir().local_def_id(ast::CRATE_NODE_ID), sig).clean(cx),
                    abi: sig.abi(),
                })
            }
            ty::Adt(def, substs) => {
                let did = def.did;
                let kind = match def.adt_kind() {
                    AdtKind::Struct => TypeKind::Struct,
                    AdtKind::Union => TypeKind::Union,
                    AdtKind::Enum => TypeKind::Enum,
                };
                inline::record_extern_fqn(cx, did, kind);
                let path = external_path(cx, &cx.tcx.item_name(did).as_str(),
                                         None, false, vec![], substs);
                ResolvedPath {
                    path,
                    typarams: None,
                    did,
                    is_generic: false,
                }
            }
            ty::Foreign(did) => {
                inline::record_extern_fqn(cx, did, TypeKind::Foreign);
                let path = external_path(cx, &cx.tcx.item_name(did).as_str(),
                                         None, false, vec![], Substs::empty());
                ResolvedPath {
                    path: path,
                    typarams: None,
                    did: did,
                    is_generic: false,
                }
            }
            ty::Dynamic(ref obj, ref reg) => {
                // HACK: pick the first `did` as the `did` of the trait object. Someone
                // might want to implement "native" support for marker-trait-only
                // trait objects.
                let mut dids = obj.principal_def_id().into_iter().chain(obj.auto_traits());
                let did = dids.next().unwrap_or_else(|| {
                    panic!("found trait object `{:?}` with no traits?", self)
                });
                let substs = match obj.principal() {
                    Some(principal) => principal.skip_binder().substs,
                    // marker traits have no substs.
                    _ => cx.tcx.intern_substs(&[])
                };

                inline::record_extern_fqn(cx, did, TypeKind::Trait);

                let mut typarams = vec![];
                reg.clean(cx).map(|b| typarams.push(GenericBound::Outlives(b)));
                for did in dids {
                    let empty = cx.tcx.intern_substs(&[]);
                    let path = external_path(cx, &cx.tcx.item_name(did).as_str(),
                        Some(did), false, vec![], empty);
                    inline::record_extern_fqn(cx, did, TypeKind::Trait);
                    let bound = GenericBound::TraitBound(PolyTrait {
                        trait_: ResolvedPath {
                            path,
                            typarams: None,
                            did,
                            is_generic: false,
                        },
                        generic_params: Vec::new(),
                    }, hir::TraitBoundModifier::None);
                    typarams.push(bound);
                }

                let mut bindings = vec![];
                for pb in obj.projection_bounds() {
                    bindings.push(TypeBinding {
                        name: cx.tcx.associated_item(pb.item_def_id()).ident.name.clean(cx),
                        ty: pb.skip_binder().ty.clean(cx)
                    });
                }

                let path = external_path(cx, &cx.tcx.item_name(did).as_str(), Some(did),
                    false, bindings, substs);
                ResolvedPath {
                    path,
                    typarams: Some(typarams),
                    did,
                    is_generic: false,
                }
            }
            ty::Tuple(ref t) => Tuple(t.clean(cx)),

            ty::Projection(ref data) => data.clean(cx),

            ty::Param(ref p) => Generic(p.name.to_string()),

            ty::Opaque(def_id, substs) => {
                // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
                // by looking up the projections associated with the def_id.
                let predicates_of = cx.tcx.predicates_of(def_id);
                let substs = cx.tcx.lift(&substs).expect("Opaque lift failed");
                let bounds = predicates_of.instantiate(cx.tcx, substs);
                let mut regions = vec![];
                let mut has_sized = false;
                let mut bounds = bounds.predicates.iter().filter_map(|predicate| {
                    let trait_ref = if let Some(tr) = predicate.to_opt_poly_trait_ref() {
                        tr
                    } else if let ty::Predicate::TypeOutlives(pred) = *predicate {
                        // these should turn up at the end
                        pred.skip_binder().1.clean(cx).map(|r| {
                            regions.push(GenericBound::Outlives(r))
                        });
                        return None;
                    } else {
                        return None;
                    };

                    if let Some(sized) = cx.tcx.lang_items().sized_trait() {
                        if trait_ref.def_id() == sized {
                            has_sized = true;
                            return None;
                        }
                    }

                    let bounds = bounds.predicates.iter().filter_map(|pred|
                        if let ty::Predicate::Projection(proj) = *pred {
                            let proj = proj.skip_binder();
                            if proj.projection_ty.trait_ref(cx.tcx) == *trait_ref.skip_binder() {
                                Some(TypeBinding {
                                    name: cx.tcx.associated_item(proj.projection_ty.item_def_id)
                                                .ident.name.clean(cx),
                                    ty: proj.ty.clean(cx),
                                })
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    ).collect();

                    Some((trait_ref.skip_binder(), bounds).clean(cx))
                }).collect::<Vec<_>>();
                bounds.extend(regions);
                if !has_sized && !bounds.is_empty() {
                    bounds.insert(0, GenericBound::maybe_sized(cx));
                }
                ImplTrait(bounds)
            }

            ty::Closure(..) | ty::Generator(..) => Tuple(vec![]), // FIXME(pcwalton)

            ty::Bound(..) => panic!("Bound"),
            ty::Placeholder(..) => panic!("Placeholder"),
            ty::UnnormalizedProjection(..) => panic!("UnnormalizedProjection"),
            ty::GeneratorWitness(..) => panic!("GeneratorWitness"),
            ty::Infer(..) => panic!("Infer"),
            ty::Error => panic!("Error"),
        }
    }
}

impl Clean<Item> for hir::StructField {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.ident.name).clean(cx),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, cx.tcx.hir().local_def_id(self.id)),
            deprecation: get_deprecation(cx, cx.tcx.hir().local_def_id(self.id)),
            def_id: cx.tcx.hir().local_def_id(self.id),
            inner: StructFieldItem(self.ty.clean(cx)),
        }
    }
}

impl<'tcx> Clean<Item> for ty::FieldDef {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.ident.name).clean(cx),
            attrs: cx.tcx.get_attrs(self.did).clean(cx),
            source: cx.tcx.def_span(self.did).clean(cx),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, self.did),
            deprecation: get_deprecation(cx, self.did),
            def_id: self.did,
            inner: StructFieldItem(cx.tcx.type_of(self.did).clean(cx)),
        }
    }
}

#[derive(Clone, PartialEq, Eq, RustcDecodable, RustcEncodable, Debug)]
pub enum Visibility {
    Public,
    Inherited,
    Crate,
    Restricted(DefId, Path),
}

impl Clean<Option<Visibility>> for hir::Visibility {
    fn clean(&self, cx: &DocContext) -> Option<Visibility> {
        Some(match self.node {
            hir::VisibilityKind::Public => Visibility::Public,
            hir::VisibilityKind::Inherited => Visibility::Inherited,
            hir::VisibilityKind::Crate(_) => Visibility::Crate,
            hir::VisibilityKind::Restricted { ref path, .. } => {
                let path = path.clean(cx);
                let did = register_def(cx, path.def);
                Visibility::Restricted(did, path)
            }
        })
    }
}

impl Clean<Option<Visibility>> for ty::Visibility {
    fn clean(&self, _: &DocContext) -> Option<Visibility> {
        Some(if *self == ty::Visibility::Public { Public } else { Inherited })
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Struct {
    pub struct_type: doctree::StructType,
    pub generics: Generics,
    pub fields: Vec<Item>,
    pub fields_stripped: bool,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Union {
    pub struct_type: doctree::StructType,
    pub generics: Generics,
    pub fields: Vec<Item>,
    pub fields_stripped: bool,
}

impl Clean<Item> for doctree::Struct {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: StructItem(Struct {
                struct_type: self.struct_type,
                generics: self.generics.clean(cx),
                fields: self.fields.clean(cx),
                fields_stripped: false,
            }),
        }
    }
}

impl Clean<Item> for doctree::Union {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: UnionItem(Union {
                struct_type: self.struct_type,
                generics: self.generics.clean(cx),
                fields: self.fields.clean(cx),
                fields_stripped: false,
            }),
        }
    }
}

/// This is a more limited form of the standard Struct, different in that
/// it lacks the things most items have (name, id, parameterization). Found
/// only as a variant in an enum.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct VariantStruct {
    pub struct_type: doctree::StructType,
    pub fields: Vec<Item>,
    pub fields_stripped: bool,
}

impl Clean<VariantStruct> for ::rustc::hir::VariantData {
    fn clean(&self, cx: &DocContext) -> VariantStruct {
        VariantStruct {
            struct_type: doctree::struct_type_from_def(self),
            fields: self.fields().iter().map(|x| x.clean(cx)).collect(),
            fields_stripped: false,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Enum {
    pub variants: IndexVec<VariantIdx, Item>,
    pub generics: Generics,
    pub variants_stripped: bool,
}

impl Clean<Item> for doctree::Enum {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: EnumItem(Enum {
                variants: self.variants.iter().map(|v| v.clean(cx)).collect(),
                generics: self.generics.clean(cx),
                variants_stripped: false,
            }),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Variant {
    pub kind: VariantKind,
}

impl Clean<Item> for doctree::Variant {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            visibility: None,
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.def.id()),
            inner: VariantItem(Variant {
                kind: self.def.clean(cx),
            }),
        }
    }
}

impl<'tcx> Clean<Item> for ty::VariantDef {
    fn clean(&self, cx: &DocContext) -> Item {
        let kind = match self.ctor_kind {
            CtorKind::Const => VariantKind::CLike,
            CtorKind::Fn => {
                VariantKind::Tuple(
                    self.fields.iter().map(|f| cx.tcx.type_of(f.did).clean(cx)).collect()
                )
            }
            CtorKind::Fictive => {
                VariantKind::Struct(VariantStruct {
                    struct_type: doctree::Plain,
                    fields_stripped: false,
                    fields: self.fields.iter().map(|field| {
                        Item {
                            source: cx.tcx.def_span(field.did).clean(cx),
                            name: Some(field.ident.name.clean(cx)),
                            attrs: cx.tcx.get_attrs(field.did).clean(cx),
                            visibility: field.vis.clean(cx),
                            def_id: field.did,
                            stability: get_stability(cx, field.did),
                            deprecation: get_deprecation(cx, field.did),
                            inner: StructFieldItem(cx.tcx.type_of(field.did).clean(cx))
                        }
                    }).collect()
                })
            }
        };
        Item {
            name: Some(self.ident.clean(cx)),
            attrs: inline::load_attrs(cx, self.did),
            source: cx.tcx.def_span(self.did).clean(cx),
            visibility: Some(Inherited),
            def_id: self.did,
            inner: VariantItem(Variant { kind }),
            stability: get_stability(cx, self.did),
            deprecation: get_deprecation(cx, self.did),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum VariantKind {
    CLike,
    Tuple(Vec<Type>),
    Struct(VariantStruct),
}

impl Clean<VariantKind> for hir::VariantData {
    fn clean(&self, cx: &DocContext) -> VariantKind {
        if self.is_struct() {
            VariantKind::Struct(self.clean(cx))
        } else if self.is_unit() {
            VariantKind::CLike
        } else {
            VariantKind::Tuple(self.fields().iter().map(|x| x.ty.clean(cx)).collect())
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Span {
    pub filename: FileName,
    pub loline: usize,
    pub locol: usize,
    pub hiline: usize,
    pub hicol: usize,
}

impl Span {
    pub fn empty() -> Span {
        Span {
            filename: FileName::Anon(0),
            loline: 0, locol: 0,
            hiline: 0, hicol: 0,
        }
    }
}

impl Clean<Span> for syntax_pos::Span {
    fn clean(&self, cx: &DocContext) -> Span {
        if self.is_dummy() {
            return Span::empty();
        }

        let cm = cx.sess().source_map();
        let filename = cm.span_to_filename(*self);
        let lo = cm.lookup_char_pos(self.lo());
        let hi = cm.lookup_char_pos(self.hi());
        Span {
            filename,
            loline: lo.line,
            locol: lo.col.to_usize(),
            hiline: hi.line,
            hicol: hi.col.to_usize(),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct Path {
    pub global: bool,
    pub def: Def,
    pub segments: Vec<PathSegment>,
}

impl Path {
    pub fn last_name(&self) -> &str {
        self.segments.last().expect("segments were empty").name.as_str()
    }
}

impl Clean<Path> for hir::Path {
    fn clean(&self, cx: &DocContext) -> Path {
        Path {
            global: self.is_global(),
            def: self.def,
            segments: if self.is_global() { &self.segments[1..] } else { &self.segments }.clean(cx),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub enum GenericArgs {
    AngleBracketed {
        lifetimes: Vec<Lifetime>,
        types: Vec<Type>,
        bindings: Vec<TypeBinding>,
    },
    Parenthesized {
        inputs: Vec<Type>,
        output: Option<Type>,
    }
}

impl Clean<GenericArgs> for hir::GenericArgs {
    fn clean(&self, cx: &DocContext) -> GenericArgs {
        if self.parenthesized {
            let output = self.bindings[0].ty.clean(cx);
            GenericArgs::Parenthesized {
                inputs: self.inputs().clean(cx),
                output: if output != Type::Tuple(Vec::new()) { Some(output) } else { None }
            }
        } else {
            let (mut lifetimes, mut types) = (vec![], vec![]);
            let mut elided_lifetimes = true;
            for arg in &self.args {
                match arg {
                    GenericArg::Lifetime(lt) => {
                        if !lt.is_elided() {
                            elided_lifetimes = false;
                        }
                        lifetimes.push(lt.clean(cx));
                    }
                    GenericArg::Type(ty) => {
                        types.push(ty.clean(cx));
                    }
                }
            }
            GenericArgs::AngleBracketed {
                lifetimes: if elided_lifetimes { vec![] } else { lifetimes },
                types,
                bindings: self.bindings.clean(cx),
            }
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct PathSegment {
    pub name: String,
    pub args: GenericArgs,
}

impl Clean<PathSegment> for hir::PathSegment {
    fn clean(&self, cx: &DocContext) -> PathSegment {
        PathSegment {
            name: self.ident.name.clean(cx),
            args: self.with_generic_args(|generic_args| generic_args.clean(cx))
        }
    }
}

fn strip_type(ty: Type) -> Type {
    match ty {
        Type::ResolvedPath { path, typarams, did, is_generic } => {
            Type::ResolvedPath { path: strip_path(&path), typarams, did, is_generic }
        }
        Type::Tuple(inner_tys) => {
            Type::Tuple(inner_tys.iter().map(|t| strip_type(t.clone())).collect())
        }
        Type::Slice(inner_ty) => Type::Slice(Box::new(strip_type(*inner_ty))),
        Type::Array(inner_ty, s) => Type::Array(Box::new(strip_type(*inner_ty)), s),
        Type::Unique(inner_ty) => Type::Unique(Box::new(strip_type(*inner_ty))),
        Type::RawPointer(m, inner_ty) => Type::RawPointer(m, Box::new(strip_type(*inner_ty))),
        Type::BorrowedRef { lifetime, mutability, type_ } => {
            Type::BorrowedRef { lifetime, mutability, type_: Box::new(strip_type(*type_)) }
        }
        Type::QPath { name, self_type, trait_ } => {
            Type::QPath {
                name,
                self_type: Box::new(strip_type(*self_type)), trait_: Box::new(strip_type(*trait_))
            }
        }
        _ => ty
    }
}

fn strip_path(path: &Path) -> Path {
    let segments = path.segments.iter().map(|s| {
        PathSegment {
            name: s.name.clone(),
            args: GenericArgs::AngleBracketed {
                lifetimes: Vec::new(),
                types: Vec::new(),
                bindings: Vec::new(),
            }
        }
    }).collect();

    Path {
        global: path.global,
        def: path.def.clone(),
        segments,
    }
}

fn qpath_to_string(p: &hir::QPath) -> String {
    let segments = match *p {
        hir::QPath::Resolved(_, ref path) => &path.segments,
        hir::QPath::TypeRelative(_, ref segment) => return segment.ident.to_string(),
    };

    let mut s = String::new();
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            s.push_str("::");
        }
        if seg.ident.name != keywords::PathRoot.name() {
            s.push_str(&*seg.ident.as_str());
        }
    }
    s
}

impl Clean<String> for Ident {
    #[inline]
    fn clean(&self, cx: &DocContext) -> String {
        self.name.clean(cx)
    }
}

impl Clean<String> for ast::Name {
    #[inline]
    fn clean(&self, _: &DocContext) -> String {
        self.to_string()
    }
}

impl Clean<String> for InternedString {
    #[inline]
    fn clean(&self, _: &DocContext) -> String {
        self.to_string()
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Typedef {
    pub type_: Type,
    pub generics: Generics,
}

impl Clean<Item> for doctree::Typedef {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id.clone()),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: TypedefItem(Typedef {
                type_: self.ty.clean(cx),
                generics: self.gen.clean(cx),
            }, false),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Existential {
    pub bounds: Vec<GenericBound>,
    pub generics: Generics,
}

impl Clean<Item> for doctree::Existential {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id.clone()),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: ExistentialItem(Existential {
                bounds: self.exist_ty.bounds.clean(cx),
                generics: self.exist_ty.generics.clean(cx),
            }, false),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Debug, Hash)]
pub struct BareFunctionDecl {
    pub unsafety: hir::Unsafety,
    pub generic_params: Vec<GenericParamDef>,
    pub decl: FnDecl,
    pub abi: Abi,
}

impl Clean<BareFunctionDecl> for hir::BareFnTy {
    fn clean(&self, cx: &DocContext) -> BareFunctionDecl {
        let (generic_params, decl) = enter_impl_trait(cx, || {
            (self.generic_params.clean(cx), (&*self.decl, &self.arg_names[..]).clean(cx))
        });
        BareFunctionDecl {
            unsafety: self.unsafety,
            abi: self.abi,
            decl,
            generic_params,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Static {
    pub type_: Type,
    pub mutability: Mutability,
    /// It's useful to have the value of a static documented, but I have no
    /// desire to represent expressions (that'd basically be all of the AST,
    /// which is huge!). So, have a string.
    pub expr: String,
}

impl Clean<Item> for doctree::Static {
    fn clean(&self, cx: &DocContext) -> Item {
        debug!("cleaning static {}: {:?}", self.name.clean(cx), self);
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: StaticItem(Static {
                type_: self.type_.clean(cx),
                mutability: self.mutability.clean(cx),
                expr: print_const_expr(cx, self.expr),
            }),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Constant {
    pub type_: Type,
    pub expr: String,
}

impl Clean<Item> for doctree::Constant {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: ConstantItem(Constant {
                type_: self.type_.clean(cx),
                expr: print_const_expr(cx, self.expr),
            }),
        }
    }
}

#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Copy, Hash)]
pub enum Mutability {
    Mutable,
    Immutable,
}

impl Clean<Mutability> for hir::Mutability {
    fn clean(&self, _: &DocContext) -> Mutability {
        match self {
            &hir::MutMutable => Mutable,
            &hir::MutImmutable => Immutable,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Copy, Debug, Hash)]
pub enum ImplPolarity {
    Positive,
    Negative,
}

impl Clean<ImplPolarity> for hir::ImplPolarity {
    fn clean(&self, _: &DocContext) -> ImplPolarity {
        match self {
            &hir::ImplPolarity::Positive => ImplPolarity::Positive,
            &hir::ImplPolarity::Negative => ImplPolarity::Negative,
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
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

pub fn get_auto_traits_with_node_id(cx: &DocContext, id: ast::NodeId, name: String) -> Vec<Item> {
    let finder = AutoTraitFinder::new(cx);
    finder.get_with_node_id(id, name)
}

pub fn get_auto_traits_with_def_id(cx: &DocContext, id: DefId) -> Vec<Item> {
    let finder = AutoTraitFinder::new(cx);

    finder.get_with_def_id(id)
}

pub fn get_blanket_impls_with_node_id(cx: &DocContext, id: ast::NodeId, name: String) -> Vec<Item> {
    let finder = BlanketImplFinder::new(cx);
    finder.get_with_node_id(id, name)
}

pub fn get_blanket_impls_with_def_id(cx: &DocContext, id: DefId) -> Vec<Item> {
    let finder = BlanketImplFinder::new(cx);

    finder.get_with_def_id(id)
}

impl Clean<Vec<Item>> for doctree::Impl {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {
        let mut ret = Vec::new();
        let trait_ = self.trait_.clean(cx);
        let items = self.items.clean(cx);

        // If this impl block is an implementation of the Deref trait, then we
        // need to try inlining the target's inherent impl blocks as well.
        if trait_.def_id() == cx.tcx.lang_items().deref_trait() {
            build_deref_target_impls(cx, &items, &mut ret);
        }

        let provided = trait_.def_id().map(|did| {
            cx.tcx.provided_trait_methods(did)
                  .into_iter()
                  .map(|meth| meth.ident.to_string())
                  .collect()
        }).unwrap_or_default();

        ret.push(Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: ImplItem(Impl {
                unsafety: self.unsafety,
                generics: self.generics.clean(cx),
                provided_trait_methods: provided,
                trait_,
                for_: self.for_.clean(cx),
                items,
                polarity: Some(self.polarity.clean(cx)),
                synthetic: false,
                blanket_impl: None,
            })
        });
        ret
    }
}

fn build_deref_target_impls(cx: &DocContext,
                            items: &[Item],
                            ret: &mut Vec<Item>) {
    use self::PrimitiveType::*;
    let tcx = cx.tcx;

    for item in items {
        let target = match item.inner {
            TypedefItem(ref t, true) => &t.type_,
            _ => continue,
        };
        let primitive = match *target {
            ResolvedPath { did, .. } if did.is_local() => continue,
            ResolvedPath { did, .. } => {
                ret.extend(inline::build_impls(cx, did));
                continue
            }
            _ => match target.primitive_type() {
                Some(prim) => prim,
                None => continue,
            }
        };
        let did = match primitive {
            Isize => tcx.lang_items().isize_impl(),
            I8 => tcx.lang_items().i8_impl(),
            I16 => tcx.lang_items().i16_impl(),
            I32 => tcx.lang_items().i32_impl(),
            I64 => tcx.lang_items().i64_impl(),
            I128 => tcx.lang_items().i128_impl(),
            Usize => tcx.lang_items().usize_impl(),
            U8 => tcx.lang_items().u8_impl(),
            U16 => tcx.lang_items().u16_impl(),
            U32 => tcx.lang_items().u32_impl(),
            U64 => tcx.lang_items().u64_impl(),
            U128 => tcx.lang_items().u128_impl(),
            F32 => tcx.lang_items().f32_impl(),
            F64 => tcx.lang_items().f64_impl(),
            Char => tcx.lang_items().char_impl(),
            Bool => None,
            Str => tcx.lang_items().str_impl(),
            Slice => tcx.lang_items().slice_impl(),
            Array => tcx.lang_items().slice_impl(),
            Tuple => None,
            Unit => None,
            RawPointer => tcx.lang_items().const_ptr_impl(),
            Reference => None,
            Fn => None,
            Never => None,
        };
        if let Some(did) = did {
            if !did.is_local() {
                inline::build_impl(cx, did, ret);
            }
        }
    }
}

impl Clean<Vec<Item>> for doctree::ExternCrate {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {

        let please_inline = self.vis.node.is_pub() && self.attrs.iter().any(|a| {
            a.name() == "doc" && match a.meta_item_list() {
                Some(l) => attr::list_contains_name(&l, "inline"),
                None => false,
            }
        });

        if please_inline {
            let mut visited = FxHashSet::default();

            let def = Def::Mod(DefId {
                krate: self.cnum,
                index: CRATE_DEF_INDEX,
            });

            if let Some(items) = inline::try_inline(cx, def, self.name, &mut visited) {
                return items;
            }
        }

        vec![Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: DefId { krate: self.cnum, index: CRATE_DEF_INDEX },
            visibility: self.vis.clean(cx),
            stability: None,
            deprecation: None,
            inner: ExternCrateItem(self.name.clean(cx), self.path.clone())
        }]
    }
}

impl Clean<Vec<Item>> for doctree::Import {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {
        // We consider inlining the documentation of `pub use` statements, but we
        // forcefully don't inline if this is not public or if the
        // #[doc(no_inline)] attribute is present.
        // Don't inline doc(hidden) imports so they can be stripped at a later stage.
        let mut denied = !self.vis.node.is_pub() || self.attrs.iter().any(|a| {
            a.name() == "doc" && match a.meta_item_list() {
                Some(l) => attr::list_contains_name(&l, "no_inline") ||
                           attr::list_contains_name(&l, "hidden"),
                None => false,
            }
        });
        // Also check whether imports were asked to be inlined, in case we're trying to re-export a
        // crate in Rust 2018+
        let please_inline = self.attrs.lists("doc").has_word("inline");
        let path = self.path.clean(cx);
        let inner = if self.glob {
            if !denied {
                let mut visited = FxHashSet::default();
                if let Some(items) = inline::try_inline_glob(cx, path.def, &mut visited) {
                    return items;
                }
            }

            Import::Glob(resolve_use_source(cx, path))
        } else {
            let name = self.name;
            if !please_inline {
                match path.def {
                    Def::Mod(did) => if !did.is_local() && did.index == CRATE_DEF_INDEX {
                        // if we're `pub use`ing an extern crate root, don't inline it unless we
                        // were specifically asked for it
                        denied = true;
                    }
                    _ => {}
                }
            }
            if !denied {
                let mut visited = FxHashSet::default();
                if let Some(items) = inline::try_inline(cx, path.def, name, &mut visited) {
                    return items;
                }
            }
            Import::Simple(name.clean(cx), resolve_use_source(cx, path))
        };

        vec![Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.hir().local_def_id(ast::CRATE_NODE_ID),
            visibility: self.vis.clean(cx),
            stability: None,
            deprecation: None,
            inner: ImportItem(inner)
        }]
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum Import {
    // use source as str;
    Simple(String, ImportSource),
    // use source::*;
    Glob(ImportSource)
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ImportSource {
    pub path: Path,
    pub did: Option<DefId>,
}

impl Clean<Vec<Item>> for hir::ForeignMod {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {
        let mut items = self.items.clean(cx);
        for item in &mut items {
            if let ForeignFunctionItem(ref mut f) = item.inner {
                f.header.abi = self.abi;
            }
        }
        items
    }
}

impl Clean<Item> for hir::ForeignItem {
    fn clean(&self, cx: &DocContext) -> Item {
        let inner = match self.node {
            hir::ForeignItemKind::Fn(ref decl, ref names, ref generics) => {
                let (generics, decl) = enter_impl_trait(cx, || {
                    (generics.clean(cx), (&**decl, &names[..]).clean(cx))
                });
                ForeignFunctionItem(Function {
                    decl,
                    generics,
                    header: hir::FnHeader {
                        unsafety: hir::Unsafety::Unsafe,
                        abi: Abi::Rust,
                        constness: hir::Constness::NotConst,
                        asyncness: hir::IsAsync::NotAsync,
                    },
                })
            }
            hir::ForeignItemKind::Static(ref ty, mutbl) => {
                ForeignStaticItem(Static {
                    type_: ty.clean(cx),
                    mutability: if mutbl {Mutable} else {Immutable},
                    expr: String::new(),
                })
            }
            hir::ForeignItemKind::Type => {
                ForeignTypeItem
            }
        };

        Item {
            name: Some(self.ident.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, cx.tcx.hir().local_def_id(self.id)),
            deprecation: get_deprecation(cx, cx.tcx.hir().local_def_id(self.id)),
            inner,
        }
    }
}

// Utilities

pub trait ToSource {
    fn to_src(&self, cx: &DocContext) -> String;
}

impl ToSource for syntax_pos::Span {
    fn to_src(&self, cx: &DocContext) -> String {
        debug!("converting span {:?} to snippet", self.clean(cx));
        let sn = match cx.sess().source_map().span_to_snippet(*self) {
            Ok(x) => x,
            Err(_) => String::new()
        };
        debug!("got snippet {}", sn);
        sn
    }
}

fn name_from_pat(p: &hir::Pat) -> String {
    use rustc::hir::*;
    debug!("Trying to get a name from pattern: {:?}", p);

    match p.node {
        PatKind::Wild => "_".to_string(),
        PatKind::Binding(_, _, _, ident, _) => ident.to_string(),
        PatKind::TupleStruct(ref p, ..) | PatKind::Path(ref p) => qpath_to_string(p),
        PatKind::Struct(ref name, ref fields, etc) => {
            format!("{} {{ {}{} }}", qpath_to_string(name),
                fields.iter().map(|&Spanned { node: ref fp, .. }|
                                  format!("{}: {}", fp.ident, name_from_pat(&*fp.pat)))
                             .collect::<Vec<String>>().join(", "),
                if etc { ", .." } else { "" }
            )
        }
        PatKind::Tuple(ref elts, _) => format!("({})", elts.iter().map(|p| name_from_pat(&**p))
                                            .collect::<Vec<String>>().join(", ")),
        PatKind::Box(ref p) => name_from_pat(&**p),
        PatKind::Ref(ref p, _) => name_from_pat(&**p),
        PatKind::Lit(..) => {
            warn!("tried to get argument name from PatKind::Lit, \
                  which is silly in function arguments");
            "()".to_string()
        },
        PatKind::Range(..) => panic!("tried to get argument name from PatKind::Range, \
                              which is not allowed in function arguments"),
        PatKind::Slice(ref begin, ref mid, ref end) => {
            let begin = begin.iter().map(|p| name_from_pat(&**p));
            let mid = mid.as_ref().map(|p| format!("..{}", name_from_pat(&**p))).into_iter();
            let end = end.iter().map(|p| name_from_pat(&**p));
            format!("[{}]", begin.chain(mid).chain(end).collect::<Vec<_>>().join(", "))
        },
    }
}

fn print_const(cx: &DocContext, n: ty::LazyConst) -> String {
    match n {
        ty::LazyConst::Unevaluated(def_id, _) => {
            if let Some(node_id) = cx.tcx.hir().as_local_node_id(def_id) {
                print_const_expr(cx, cx.tcx.hir().body_owned_by(node_id))
            } else {
                inline::print_inlined_const(cx, def_id)
            }
        },
        ty::LazyConst::Evaluated(n) => {
            let mut s = String::new();
            ::rustc::mir::fmt_const_val(&mut s, n).expect("fmt_const_val failed");
            // array lengths are obviously usize
            if s.ends_with("usize") {
                let n = s.len() - "usize".len();
                s.truncate(n);
            }
            s
        },
    }
}

fn print_const_expr(cx: &DocContext, body: hir::BodyId) -> String {
    cx.tcx.hir().hir_to_pretty_string(body.hir_id)
}

/// Given a type Path, resolve it to a Type using the TyCtxt
fn resolve_type(cx: &DocContext,
                path: Path,
                id: ast::NodeId) -> Type {
    if id == ast::DUMMY_NODE_ID {
        debug!("resolve_type({:?})", path);
    } else {
        debug!("resolve_type({:?},{:?})", path, id);
    }

    let is_generic = match path.def {
        Def::PrimTy(p) => match p {
            hir::Str => return Primitive(PrimitiveType::Str),
            hir::Bool => return Primitive(PrimitiveType::Bool),
            hir::Char => return Primitive(PrimitiveType::Char),
            hir::Int(int_ty) => return Primitive(int_ty.into()),
            hir::Uint(uint_ty) => return Primitive(uint_ty.into()),
            hir::Float(float_ty) => return Primitive(float_ty.into()),
        },
        Def::SelfTy(..) if path.segments.len() == 1 => {
            return Generic(keywords::SelfUpper.name().to_string());
        }
        Def::TyParam(..) if path.segments.len() == 1 => {
            return Generic(format!("{:#}", path));
        }
        Def::SelfTy(..) | Def::TyParam(..) | Def::AssociatedTy(..) => true,
        _ => false,
    };
    let did = register_def(&*cx, path.def);
    ResolvedPath { path: path, typarams: None, did: did, is_generic: is_generic }
}

pub fn register_def(cx: &DocContext, def: Def) -> DefId {
    debug!("register_def({:?})", def);

    let (did, kind) = match def {
        Def::Fn(i) => (i, TypeKind::Function),
        Def::TyAlias(i) => (i, TypeKind::Typedef),
        Def::Enum(i) => (i, TypeKind::Enum),
        Def::Trait(i) => (i, TypeKind::Trait),
        Def::Struct(i) => (i, TypeKind::Struct),
        Def::Union(i) => (i, TypeKind::Union),
        Def::Mod(i) => (i, TypeKind::Module),
        Def::ForeignTy(i) => (i, TypeKind::Foreign),
        Def::Const(i) => (i, TypeKind::Const),
        Def::Static(i, _) => (i, TypeKind::Static),
        Def::Variant(i) => (cx.tcx.parent_def_id(i).expect("cannot get parent def id"),
                            TypeKind::Enum),
        Def::Macro(i, mac_kind) => match mac_kind {
            MacroKind::Bang => (i, TypeKind::Macro),
            MacroKind::Attr => (i, TypeKind::Attr),
            MacroKind::Derive => (i, TypeKind::Derive),
            MacroKind::ProcMacroStub => unreachable!(),
        },
        Def::TraitAlias(i) => (i, TypeKind::TraitAlias),
        Def::SelfTy(Some(def_id), _) => (def_id, TypeKind::Trait),
        Def::SelfTy(_, Some(impl_def_id)) => return impl_def_id,
        _ => return def.def_id()
    };
    if did.is_local() { return did }
    inline::record_extern_fqn(cx, did, kind);
    if let TypeKind::Trait = kind {
        inline::record_extern_trait(cx, did);
    }
    did
}

fn resolve_use_source(cx: &DocContext, path: Path) -> ImportSource {
    ImportSource {
        did: if path.def.opt_def_id().is_none() {
            None
        } else {
            Some(register_def(cx, path.def))
        },
        path,
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Macro {
    pub source: String,
    pub imported_from: Option<String>,
}

impl Clean<Item> for doctree::Macro {
    fn clean(&self, cx: &DocContext) -> Item {
        let name = self.name.clean(cx);
        Item {
            name: Some(name.clone()),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            visibility: Some(Public),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            def_id: self.def_id,
            inner: MacroItem(Macro {
                source: format!("macro_rules! {} {{\n{}}}",
                                name,
                                self.matchers.iter().map(|span| {
                                    format!("    {} => {{ ... }};\n", span.to_src(cx))
                                }).collect::<String>()),
                imported_from: self.imported_from.clean(cx),
            }),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ProcMacro {
    pub kind: MacroKind,
    pub helpers: Vec<String>,
}

impl Clean<Item> for doctree::ProcMacro {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            visibility: Some(Public),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            def_id: cx.tcx.hir().local_def_id(self.id),
            inner: ProcMacroItem(ProcMacro {
                kind: self.kind,
                helpers: self.helpers.clean(cx),
            }),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Stability {
    pub level: stability::StabilityLevel,
    pub feature: Option<String>,
    pub since: String,
    pub deprecation: Option<Deprecation>,
    pub unstable_reason: Option<String>,
    pub issue: Option<u32>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Deprecation {
    pub since: Option<String>,
    pub note: Option<String>,
}

impl Clean<Stability> for attr::Stability {
    fn clean(&self, _: &DocContext) -> Stability {
        Stability {
            level: stability::StabilityLevel::from_attr_level(&self.level),
            feature: Some(self.feature.to_string()).filter(|f| !f.is_empty()),
            since: match self.level {
                attr::Stable {ref since} => since.to_string(),
                _ => String::new(),
            },
            deprecation: self.rustc_depr.as_ref().map(|d| {
                Deprecation {
                    note: Some(d.reason.to_string()).filter(|r| !r.is_empty()),
                    since: Some(d.since.to_string()).filter(|d| !d.is_empty()),
                }
            }),
            unstable_reason: match self.level {
                attr::Unstable { reason: Some(ref reason), .. } => Some(reason.to_string()),
                _ => None,
            },
            issue: match self.level {
                attr::Unstable {issue, ..} => Some(issue),
                _ => None,
            }
        }
    }
}

impl<'a> Clean<Stability> for &'a attr::Stability {
    fn clean(&self, dc: &DocContext) -> Stability {
        (**self).clean(dc)
    }
}

impl Clean<Deprecation> for attr::Deprecation {
    fn clean(&self, _: &DocContext) -> Deprecation {
        Deprecation {
            since: self.since.map(|s| s.to_string()).filter(|s| !s.is_empty()),
            note: self.note.map(|n| n.to_string()).filter(|n| !n.is_empty()),
        }
    }
}

/// An equality constraint on an associated type, e.g., `A = Bar` in `Foo<A = Bar>`
#[derive(Clone, PartialEq, Eq, RustcDecodable, RustcEncodable, Debug, Hash)]
pub struct TypeBinding {
    pub name: String,
    pub ty: Type
}

impl Clean<TypeBinding> for hir::TypeBinding {
    fn clean(&self, cx: &DocContext) -> TypeBinding {
        TypeBinding {
            name: self.ident.name.clean(cx),
            ty: self.ty.clean(cx)
        }
    }
}

pub fn def_id_to_path(cx: &DocContext, did: DefId, name: Option<String>) -> Vec<String> {
    let crate_name = name.unwrap_or_else(|| cx.tcx.crate_name(did.krate).to_string());
    let relative = cx.tcx.def_path(did).data.into_iter().filter_map(|elem| {
        // extern blocks have an empty name
        let s = elem.data.to_string();
        if !s.is_empty() {
            Some(s)
        } else {
            None
        }
    });
    once(crate_name).chain(relative).collect()
}

pub fn enter_impl_trait<F, R>(cx: &DocContext, f: F) -> R
where
    F: FnOnce() -> R,
{
    let old_bounds = mem::replace(&mut *cx.impl_trait_bounds.borrow_mut(), Default::default());
    let r = f();
    assert!(cx.impl_trait_bounds.borrow().is_empty());
    *cx.impl_trait_bounds.borrow_mut() = old_bounds;
    r
}

// Start of code copied from rust-clippy

pub fn path_to_def_local(tcx: &TyCtxt, path: &[&str]) -> Option<DefId> {
    let krate = tcx.hir().krate();
    let mut items = krate.module.item_ids.clone();
    let mut path_it = path.iter().peekable();

    loop {
        let segment = path_it.next()?;

        for item_id in mem::replace(&mut items, HirVec::new()).iter() {
            let item = tcx.hir().expect_item(item_id.id);
            if item.ident.name == *segment {
                if path_it.peek().is_none() {
                    return Some(tcx.hir().local_def_id(item_id.id))
                }

                items = match &item.node {
                    &hir::ItemKind::Mod(ref m) => m.item_ids.clone(),
                    _ => panic!("Unexpected item {:?} in path {:?} path")
                };
                break;
            }
        }
    }
}

pub fn path_to_def(tcx: &TyCtxt, path: &[&str]) -> Option<DefId> {
    let crates = tcx.crates();

    let krate = crates
        .iter()
        .find(|&&krate| tcx.crate_name(krate) == path[0]);

    if let Some(krate) = krate {
        let krate = DefId {
            krate: *krate,
            index: CRATE_DEF_INDEX,
        };
        let mut items = tcx.item_children(krate);
        let mut path_it = path.iter().skip(1).peekable();

        loop {
            let segment = path_it.next()?;

            for item in mem::replace(&mut items, Lrc::new(vec![])).iter() {
                if item.ident.name == *segment {
                    if path_it.peek().is_none() {
                        return match item.def {
                            def::Def::Trait(did) => Some(did),
                            _ => None,
                        }
                    }

                    items = tcx.item_children(item.def.def_id());
                    break;
                }
            }
        }
    } else {
        None
    }
}

pub fn get_path_for_type<F>(tcx: TyCtxt, def_id: DefId, def_ctor: F) -> hir::Path
where F: Fn(DefId) -> Def {
    #[derive(Debug)]
    struct AbsolutePathBuffer {
        names: Vec<String>,
    }

    impl ty::item_path::ItemPathBuffer for AbsolutePathBuffer {
        fn root_mode(&self) -> &ty::item_path::RootMode {
            const ABSOLUTE: &'static ty::item_path::RootMode = &ty::item_path::RootMode::Absolute;
            ABSOLUTE
        }

        fn push(&mut self, text: &str) {
            self.names.push(text.to_owned());
        }
    }

    let mut apb = AbsolutePathBuffer { names: vec![] };

    tcx.push_item_path(&mut apb, def_id, false);

    hir::Path {
        span: DUMMY_SP,
        def: def_ctor(def_id),
        segments: hir::HirVec::from_vec(apb.names.iter().map(|s| hir::PathSegment {
            ident: ast::Ident::from_str(&s),
            id: None,
            hir_id: None,
            def: None,
            args: None,
            infer_types: false,
        }).collect())
    }
}

// End of code copied from rust-clippy


#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
enum RegionTarget<'tcx> {
    Region(Region<'tcx>),
    RegionVid(RegionVid)
}

#[derive(Default, Debug, Clone)]
struct RegionDeps<'tcx> {
    larger: FxHashSet<RegionTarget<'tcx>>,
    smaller: FxHashSet<RegionTarget<'tcx>>
}

#[derive(Eq, PartialEq, Hash, Debug)]
enum SimpleBound {
    TraitBound(Vec<PathSegment>, Vec<SimpleBound>, Vec<GenericParamDef>, hir::TraitBoundModifier),
    Outlives(Lifetime),
}

enum AutoTraitResult {
    ExplicitImpl,
    PositiveImpl(Generics),
    NegativeImpl,
}

impl AutoTraitResult {
    fn is_auto(&self) -> bool {
        match *self {
            AutoTraitResult::PositiveImpl(_) | AutoTraitResult::NegativeImpl => true,
            _ => false,
        }
    }
}

impl From<GenericBound> for SimpleBound {
    fn from(bound: GenericBound) -> Self {
        match bound.clone() {
            GenericBound::Outlives(l) => SimpleBound::Outlives(l),
            GenericBound::TraitBound(t, mod_) => match t.trait_ {
                Type::ResolvedPath { path, typarams, .. } => {
                    SimpleBound::TraitBound(path.segments,
                                            typarams
                                                .map_or_else(|| Vec::new(), |v| v.iter()
                                                        .map(|p| SimpleBound::from(p.clone()))
                                                        .collect()),
                                            t.generic_params,
                                            mod_)
                }
                _ => panic!("Unexpected bound {:?}", bound),
            }
        }
    }
}
