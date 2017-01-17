// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains the "cleaned" pieces of the AST, and the functions
//! that clean them.

pub use self::Type::*;
pub use self::Mutability::*;
pub use self::ItemEnum::*;
pub use self::TyParamBound::*;
pub use self::SelfTy::*;
pub use self::FunctionRetTy::*;
pub use self::Visibility::*;

use syntax::abi::Abi;
use syntax::ast;
use syntax::attr;
use syntax::codemap::Spanned;
use syntax::ptr::P;
use syntax::symbol::keywords;
use syntax_pos::{self, DUMMY_SP, Pos};

use rustc::middle::privacy::AccessLevels;
use rustc::middle::resolve_lifetime::DefRegion::*;
use rustc::middle::lang_items;
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc::ty::subst::Substs;
use rustc::ty::{self, AdtKind};
use rustc::middle::stability;
use rustc::util::nodemap::{FxHashMap, FxHashSet};

use rustc::hir;

use std::path::PathBuf;
use std::rc::Rc;
use std::slice;
use std::sync::Arc;
use std::u32;
use std::mem;

use core::DocContext;
use doctree;
use visit_ast;
use html::item_type::ItemType;

pub mod inline;
mod simplify;

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
        self.0.clean(cx)
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
    pub src: PathBuf,
    pub module: Option<Item>,
    pub externs: Vec<(CrateNum, ExternalCrate)>,
    pub primitives: Vec<(DefId, PrimitiveType, Attributes)>,
    pub access_levels: Arc<AccessLevels<DefId>>,
    // These are later on moved into `CACHEKEY`, leaving the map empty.
    // Only here so that they can be filtered through the rustdoc passes.
    pub external_traits: FxHashMap<DefId, Trait>,
}

impl<'a, 'tcx> Clean<Crate> for visit_ast::RustdocVisitor<'a, 'tcx> {
    fn clean(&self, cx: &DocContext) -> Crate {
        use ::visit_lib::LibEmbargoVisitor;

        {
            let mut r = cx.renderinfo.borrow_mut();
            r.deref_trait_did = cx.tcx.lang_items.deref_trait();
            r.deref_mut_trait_did = cx.tcx.lang_items.deref_mut_trait();
        }

        let mut externs = Vec::new();
        for cnum in cx.sess().cstore.crates() {
            externs.push((cnum, cnum.clean(cx)));
            // Analyze doc-reachability for extern items
            LibEmbargoVisitor::new(cx).visit_lib(cnum);
        }
        externs.sort_by(|&(a, _), &(b, _)| a.cmp(&b));

        // Clean the crate, translating the entire libsyntax AST to one that is
        // understood by rustdoc.
        let mut module = self.module.clean(cx);

        let ExternalCrate { name, src, primitives, .. } = LOCAL_CRATE.clean(cx);
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
                    def_id: def_id,
                    inner: PrimitiveItem(prim),
                }
            }));
        }

        let mut access_levels = cx.access_levels.borrow_mut();
        let mut external_traits = cx.external_traits.borrow_mut();

        Crate {
            name: name,
            src: src,
            module: Some(module),
            externs: externs,
            primitives: primitives,
            access_levels: Arc::new(mem::replace(&mut access_levels, Default::default())),
            external_traits: mem::replace(&mut external_traits, Default::default()),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ExternalCrate {
    pub name: String,
    pub src: PathBuf,
    pub attrs: Attributes,
    pub primitives: Vec<(DefId, PrimitiveType, Attributes)>,
}

impl Clean<ExternalCrate> for CrateNum {
    fn clean(&self, cx: &DocContext) -> ExternalCrate {
        let root = DefId { krate: *self, index: CRATE_DEF_INDEX };
        let krate_span = cx.tcx.def_span(root);
        let krate_src = cx.sess().codemap().span_to_filename(krate_span);

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
                        }
                    }
                }
                return prim.map(|p| (def_id, p, attrs));
            }
            None
        };
        let primitives = if root.is_local() {
            cx.tcx.map.krate().module.item_ids.iter().filter_map(|&id| {
                let item = cx.tcx.map.expect_item(id.id);
                match item.node {
                    hir::ItemMod(_) => {
                        as_primitive(Def::Mod(cx.tcx.map.local_def_id(id.id)))
                    }
                    hir::ItemUse(ref path, hir::UseKind::Single)
                    if item.vis == hir::Visibility::Public => {
                        as_primitive(path.def).map(|(_, prim, attrs)| {
                            // Pretend the primitive is local.
                            (cx.tcx.map.local_def_id(id.id), prim, attrs)
                        })
                    }
                    _ => None
                }
            }).collect()
        } else {
            cx.tcx.sess.cstore.item_children(root).iter().map(|item| item.def)
              .filter_map(as_primitive).collect()
        };

        ExternalCrate {
            name: cx.tcx.crate_name(*self).to_string(),
            src: PathBuf::from(krate_src),
            attrs: cx.tcx.get_attrs(root).clean(cx),
            primitives: primitives,
        }
    }
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
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

impl Item {
    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    pub fn doc_value<'a>(&'a self) -> Option<&'a str> {
        self.attrs.doc_value()
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
        self.type_() == ItemType::Module
    }
    pub fn is_fn(&self) -> bool {
        self.type_() == ItemType::Function
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
    pub fn is_primitive(&self) -> bool {
        self.type_() == ItemType::Primitive
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

    pub fn stability_class(&self) -> String {
        self.stability.as_ref().map(|ref s| {
            let mut base = match s.level {
                stability::Unstable => "unstable".to_string(),
                stability::Stable => String::new(),
            };
            if !s.deprecated_since.is_empty() {
                base.push_str(" deprecated");
            }
            base
        }).unwrap_or(String::new())
    }

    pub fn stable_since(&self) -> Option<&str> {
        self.stability.as_ref().map(|s| &s.since[..])
    }

    /// Returns a documentation-level item type from the item.
    pub fn type_(&self) -> ItemType {
        ItemType::from(self)
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
    StaticItem(Static),
    ConstantItem(Constant),
    TraitItem(Trait),
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
    MacroItem(Macro),
    PrimitiveItem(PrimitiveType),
    AssociatedConstItem(Type, Option<String>),
    AssociatedTypeItem(Vec<TyParamBound>, Option<Type>),
    DefaultImplItem(DefaultImpl),
    /// An item that has been stripped by a rustdoc pass
    StrippedItem(Box<ItemEnum>),
}

impl ItemEnum {
    pub fn generics(&self) -> Option<&Generics> {
        Some(match *self {
            ItemEnum::StructItem(ref s) => &s.generics,
            ItemEnum::EnumItem(ref e) => &e.generics,
            ItemEnum::FunctionItem(ref f) => &f.generics,
            ItemEnum::TypedefItem(ref t, _) => &t.generics,
            ItemEnum::TraitItem(ref t) => &t.generics,
            ItemEnum::ImplItem(ref i) => &i.generics,
            ItemEnum::TyMethodItem(ref i) => &i.generics,
            ItemEnum::MethodItem(ref i) => &i.generics,
            ItemEnum::ForeignFunctionItem(ref f) => &f.generics,
            _ => return None,
        })
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
            self.name.unwrap().clean(cx)
        } else {
            "".to_string()
        };

        let mut items: Vec<Item> = vec![];
        items.extend(self.extern_crates.iter().map(|x| x.clean(cx)));
        items.extend(self.imports.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.structs.iter().map(|x| x.clean(cx)));
        items.extend(self.unions.iter().map(|x| x.clean(cx)));
        items.extend(self.enums.iter().map(|x| x.clean(cx)));
        items.extend(self.fns.iter().map(|x| x.clean(cx)));
        items.extend(self.foreigns.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.mods.iter().map(|x| x.clean(cx)));
        items.extend(self.typedefs.iter().map(|x| x.clean(cx)));
        items.extend(self.statics.iter().map(|x| x.clean(cx)));
        items.extend(self.constants.iter().map(|x| x.clean(cx)));
        items.extend(self.traits.iter().map(|x| x.clean(cx)));
        items.extend(self.impls.iter().flat_map(|x| x.clean(cx)));
        items.extend(self.macros.iter().map(|x| x.clean(cx)));
        items.extend(self.def_traits.iter().map(|x| x.clean(cx)));

        // determine if we should display the inner contents or
        // the outer `mod` item for the source code.
        let whence = {
            let cm = cx.sess().codemap();
            let outer = cm.lookup_char_pos(self.where_outer.lo);
            let inner = cm.lookup_char_pos(self.where_inner.lo);
            if outer.file.start_pos == inner.file.start_pos {
                // mod foo { ... }
                self.where_outer
            } else {
                // mod foo; (and a separate FileMap for the contents)
                self.where_inner
            }
        };

        Item {
            name: Some(name),
            attrs: self.attrs.clean(cx),
            source: whence.clean(cx),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            inner: ModuleItem(Module {
               is_crate: self.is_crate,
               items: items
            })
        }
    }
}

pub struct ListAttributesIter<'a> {
    attrs: slice::Iter<'a, ast::Attribute>,
    current_list: slice::Iter<'a, ast::NestedMetaItem>,
    name: &'a str
}

impl<'a> Iterator for ListAttributesIter<'a> {
    type Item = &'a ast::NestedMetaItem;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(nested) = self.current_list.next() {
            return Some(nested);
        }

        for attr in &mut self.attrs {
            if let Some(ref list) = attr.meta_item_list() {
                if attr.check_name(self.name) {
                    self.current_list = list.iter();
                    if let Some(nested) = self.current_list.next() {
                        return Some(nested);
                    }
                }
            }
        }

        None
    }
}

pub trait AttributesExt {
    /// Finds an attribute as List and returns the list of attributes nested inside.
    fn lists<'a>(&'a self, &'a str) -> ListAttributesIter<'a>;
}

impl AttributesExt for [ast::Attribute] {
    fn lists<'a>(&'a self, name: &'a str) -> ListAttributesIter<'a> {
        ListAttributesIter {
            attrs: self.iter(),
            current_list: [].iter(),
            name: name
        }
    }
}

pub trait NestedAttributesExt {
    /// Returns whether the attribute list contains a specific `Word`
    fn has_word(self, &str) -> bool;
}

impl<'a, I: IntoIterator<Item=&'a ast::NestedMetaItem>> NestedAttributesExt for I {
    fn has_word(self, word: &str) -> bool {
        self.into_iter().any(|attr| attr.is_word() && attr.check_name(word))
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug, Default)]
pub struct Attributes {
    pub doc_strings: Vec<String>,
    pub other_attrs: Vec<ast::Attribute>
}

impl Attributes {
    pub fn from_ast(attrs: &[ast::Attribute]) -> Attributes {
        let mut doc_strings = vec![];
        let other_attrs = attrs.iter().filter_map(|attr| {
            attr.with_desugared_doc(|attr| {
                if let Some(value) = attr.value_str() {
                    if attr.check_name("doc") {
                        doc_strings.push(value.to_string());
                        return None;
                    }
                }

                Some(attr.clone())
            })
        }).collect();
        Attributes {
            doc_strings: doc_strings,
            other_attrs: other_attrs
        }
    }

    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    pub fn doc_value<'a>(&'a self) -> Option<&'a str> {
        self.doc_strings.first().map(|s| &s[..])
    }
}

impl AttributesExt for Attributes {
    fn lists<'a>(&'a self, name: &'a str) -> ListAttributesIter<'a> {
        self.other_attrs.lists(name)
    }
}

impl Clean<Attributes> for [ast::Attribute] {
    fn clean(&self, _cx: &DocContext) -> Attributes {
        Attributes::from_ast(self)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct TyParam {
    pub name: String,
    pub did: DefId,
    pub bounds: Vec<TyParamBound>,
    pub default: Option<Type>,
}

impl Clean<TyParam> for hir::TyParam {
    fn clean(&self, cx: &DocContext) -> TyParam {
        TyParam {
            name: self.name.clean(cx),
            did: cx.tcx.map.local_def_id(self.id),
            bounds: self.bounds.clean(cx),
            default: self.default.clean(cx),
        }
    }
}

impl<'tcx> Clean<TyParam> for ty::TypeParameterDef<'tcx> {
    fn clean(&self, cx: &DocContext) -> TyParam {
        cx.renderinfo.borrow_mut().external_typarams.insert(self.def_id, self.name.clean(cx));
        TyParam {
            name: self.name.clean(cx),
            did: self.def_id,
            bounds: vec![], // these are filled in from the where-clauses
            default: self.default.clean(cx),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub enum TyParamBound {
    RegionBound(Lifetime),
    TraitBound(PolyTrait, hir::TraitBoundModifier)
}

impl TyParamBound {
    fn maybe_sized(cx: &DocContext) -> TyParamBound {
        let did = cx.tcx.require_lang_item(lang_items::SizedTraitLangItem);
        let empty = cx.tcx.intern_substs(&[]);
        let path = external_path(cx, &cx.tcx.item_name(did).as_str(),
            Some(did), false, vec![], empty);
        inline::record_extern_fqn(cx, did, TypeKind::Trait);
        TraitBound(PolyTrait {
            trait_: ResolvedPath {
                path: path,
                typarams: None,
                did: did,
                is_generic: false,
            },
            lifetimes: vec![]
        }, hir::TraitBoundModifier::Maybe)
    }

    fn is_sized_bound(&self, cx: &DocContext) -> bool {
        use rustc::hir::TraitBoundModifier as TBM;
        if let TyParamBound::TraitBound(PolyTrait { ref trait_, .. }, TBM::None) = *self {
            if trait_.def_id() == cx.tcx.lang_items.sized_trait() {
                return true;
            }
        }
        false
    }
}

impl Clean<TyParamBound> for hir::TyParamBound {
    fn clean(&self, cx: &DocContext) -> TyParamBound {
        match *self {
            hir::RegionTyParamBound(lt) => RegionBound(lt.clean(cx)),
            hir::TraitTyParamBound(ref t, modifier) => TraitBound(t.clean(cx), modifier),
        }
    }
}

fn external_path_params(cx: &DocContext, trait_did: Option<DefId>, has_self: bool,
                        bindings: Vec<TypeBinding>, substs: &Substs) -> PathParameters {
    let lifetimes = substs.regions().filter_map(|v| v.clean(cx)).collect();
    let types = substs.types().skip(has_self as usize).collect::<Vec<_>>();

    match trait_did {
        // Attempt to sugar an external path like Fn<(A, B,), C> to Fn(A, B) -> C
        Some(did) if cx.tcx.lang_items.fn_trait_kind(did).is_some() => {
            assert_eq!(types.len(), 1);
            let inputs = match types[0].sty {
                ty::TyTuple(ref tys) => tys.iter().map(|t| t.clean(cx)).collect(),
                _ => {
                    return PathParameters::AngleBracketed {
                        lifetimes: lifetimes,
                        types: types.clean(cx),
                        bindings: bindings
                    }
                }
            };
            let output = None;
            // FIXME(#20299) return type comes from a projection now
            // match types[1].sty {
            //     ty::TyTuple(ref v) if v.is_empty() => None, // -> ()
            //     _ => Some(types[1].clean(cx))
            // };
            PathParameters::Parenthesized {
                inputs: inputs,
                output: output
            }
        },
        _ => {
            PathParameters::AngleBracketed {
                lifetimes: lifetimes,
                types: types.clean(cx),
                bindings: bindings
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
            params: external_path_params(cx, trait_did, has_self, bindings, substs)
        }],
    }
}

impl<'tcx> Clean<TyParamBound> for ty::TraitRef<'tcx> {
    fn clean(&self, cx: &DocContext) -> TyParamBound {
        inline::record_extern_fqn(cx, self.def_id, TypeKind::Trait);
        let path = external_path(cx, &cx.tcx.item_name(self.def_id).as_str(),
                                 Some(self.def_id), true, vec![], self.substs);

        debug!("ty::TraitRef\n  subst: {:?}\n", self.substs);

        // collect any late bound regions
        let mut late_bounds = vec![];
        for ty_s in self.input_types().skip(1) {
            if let ty::TyTuple(ts) = ty_s.sty {
                for &ty_s in ts {
                    if let ty::TyRef(ref reg, _) = ty_s.sty {
                        if let &ty::Region::ReLateBound(..) = *reg {
                            debug!("  hit an ReLateBound {:?}", reg);
                            if let Some(lt) = reg.clean(cx) {
                                late_bounds.push(lt);
                            }
                        }
                    }
                }
            }
        }

        TraitBound(
            PolyTrait {
                trait_: ResolvedPath {
                    path: path,
                    typarams: None,
                    did: self.def_id,
                    is_generic: false,
                },
                lifetimes: late_bounds,
            },
            hir::TraitBoundModifier::None
        )
    }
}

impl<'tcx> Clean<Option<Vec<TyParamBound>>> for Substs<'tcx> {
    fn clean(&self, cx: &DocContext) -> Option<Vec<TyParamBound>> {
        let mut v = Vec::new();
        v.extend(self.regions().filter_map(|r| r.clean(cx))
                     .map(RegionBound));
        v.extend(self.types().map(|t| TraitBound(PolyTrait {
            trait_: t.clean(cx),
            lifetimes: vec![]
        }, hir::TraitBoundModifier::None)));
        if !v.is_empty() {Some(v)} else {None}
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
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
        let def = cx.tcx.named_region_map.defs.get(&self.id).cloned();
        match def {
            Some(DefEarlyBoundRegion(_, node_id)) |
            Some(DefLateBoundRegion(_, node_id)) |
            Some(DefFreeRegion(_, node_id)) => {
                if let Some(lt) = cx.lt_substs.borrow().get(&node_id).cloned() {
                    return lt;
                }
            }
            _ => {}
        }
        Lifetime(self.name.to_string())
    }
}

impl Clean<Lifetime> for hir::LifetimeDef {
    fn clean(&self, _: &DocContext) -> Lifetime {
        if self.bounds.len() > 0 {
            let mut s = format!("{}: {}",
                                self.lifetime.name.to_string(),
                                self.bounds[0].name.to_string());
            for bound in self.bounds.iter().skip(1) {
                s.push_str(&format!(" + {}", bound.name.to_string()));
            }
            Lifetime(s)
        } else {
            Lifetime(self.lifetime.name.to_string())
        }
    }
}

impl<'tcx> Clean<Lifetime> for ty::RegionParameterDef<'tcx> {
    fn clean(&self, _: &DocContext) -> Lifetime {
        Lifetime(self.name.to_string())
    }
}

impl Clean<Option<Lifetime>> for ty::Region {
    fn clean(&self, cx: &DocContext) -> Option<Lifetime> {
        match *self {
            ty::ReStatic => Some(Lifetime::statik()),
            ty::ReLateBound(_, ty::BrNamed(_, name, _)) => Some(Lifetime(name.to_string())),
            ty::ReEarlyBound(ref data) => Some(Lifetime(data.name.clean(cx))),

            ty::ReLateBound(..) |
            ty::ReFree(..) |
            ty::ReScope(..) |
            ty::ReVar(..) |
            ty::ReSkolemized(..) |
            ty::ReEmpty |
            ty::ReErased => None
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub enum WherePredicate {
    BoundPredicate { ty: Type, bounds: Vec<TyParamBound> },
    RegionPredicate { lifetime: Lifetime, bounds: Vec<Lifetime>},
    EqPredicate { lhs: Type, rhs: Type }
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

            hir::WherePredicate::EqPredicate(_) => {
                unimplemented!() // FIXME(#20041)
            }
        }
    }
}

impl<'a> Clean<WherePredicate> for ty::Predicate<'a> {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        use rustc::ty::Predicate;

        match *self {
            Predicate::Trait(ref pred) => pred.clean(cx),
            Predicate::Equate(ref pred) => pred.clean(cx),
            Predicate::RegionOutlives(ref pred) => pred.clean(cx),
            Predicate::TypeOutlives(ref pred) => pred.clean(cx),
            Predicate::Projection(ref pred) => pred.clean(cx),
            Predicate::WellFormed(_) => panic!("not user writable"),
            Predicate::ObjectSafe(_) => panic!("not user writable"),
            Predicate::ClosureKind(..) => panic!("not user writable"),
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

impl<'tcx> Clean<WherePredicate> for ty::EquatePredicate<'tcx> {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        let ty::EquatePredicate(ref lhs, ref rhs) = *self;
        WherePredicate::EqPredicate {
            lhs: lhs.clean(cx),
            rhs: rhs.clean(cx)
        }
    }
}

impl<'tcx> Clean<WherePredicate> for ty::OutlivesPredicate<&'tcx ty::Region, &'tcx ty::Region> {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        let ty::OutlivesPredicate(ref a, ref b) = *self;
        WherePredicate::RegionPredicate {
            lifetime: a.clean(cx).unwrap(),
            bounds: vec![b.clean(cx).unwrap()]
        }
    }
}

impl<'tcx> Clean<WherePredicate> for ty::OutlivesPredicate<ty::Ty<'tcx>, &'tcx ty::Region> {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        let ty::OutlivesPredicate(ref ty, ref lt) = *self;

        WherePredicate::BoundPredicate {
            ty: ty.clean(cx),
            bounds: vec![TyParamBound::RegionBound(lt.clean(cx).unwrap())]
        }
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
        let trait_ = match self.trait_ref.clean(cx) {
            TyParamBound::TraitBound(t, _) => t.trait_,
            TyParamBound::RegionBound(_) => {
                panic!("cleaning a trait got a region")
            }
        };
        Type::QPath {
            name: self.item_name.clean(cx),
            self_type: box self.trait_ref.self_ty().clean(cx),
            trait_: box trait_
        }
    }
}

// maybe use a Generic enum and use Vec<Generic>?
#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct Generics {
    pub lifetimes: Vec<Lifetime>,
    pub type_params: Vec<TyParam>,
    pub where_predicates: Vec<WherePredicate>
}

impl Clean<Generics> for hir::Generics {
    fn clean(&self, cx: &DocContext) -> Generics {
        Generics {
            lifetimes: self.lifetimes.clean(cx),
            type_params: self.ty_params.clean(cx),
            where_predicates: self.where_clause.predicates.clean(cx)
        }
    }
}

impl<'a, 'tcx> Clean<Generics> for (&'a ty::Generics<'tcx>,
                                    &'a ty::GenericPredicates<'tcx>) {
    fn clean(&self, cx: &DocContext) -> Generics {
        use self::WherePredicate as WP;

        let (gens, preds) = *self;

        // Bounds in the type_params and lifetimes fields are repeated in the
        // predicates field (see rustc_typeck::collect::ty_generics), so remove
        // them.
        let stripped_typarams = gens.types.iter().filter_map(|tp| {
            if tp.name == keywords::SelfType.name() {
                assert_eq!(tp.index, 0);
                None
            } else {
                Some(tp.clean(cx))
            }
        }).collect::<Vec<_>>();
        let stripped_lifetimes = gens.regions.iter().map(|rp| {
            let mut srp = rp.clone();
            srp.bounds = Vec::new();
            srp.clean(cx)
        }).collect::<Vec<_>>();

        let mut where_predicates = preds.predicates.to_vec().clean(cx);

        // Type parameters and have a Sized bound by default unless removed with
        // ?Sized.  Scan through the predicates and mark any type parameter with
        // a Sized bound, removing the bounds as we find them.
        //
        // Note that associated types also have a sized bound by default, but we
        // don't actually know the set of associated types right here so that's
        // handled in cleaning associated types
        let mut sized_params = FxHashSet();
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
                    bounds: vec![TyParamBound::maybe_sized(cx)],
                })
            }
        }

        // It would be nice to collect all of the bounds on a type and recombine
        // them if possible, to avoid e.g. `where T: Foo, T: Bar, T: Sized, T: 'a`
        // and instead see `where T: Foo + Bar + Sized + 'a`

        Generics {
            type_params: simplify::ty_params(stripped_typarams),
            lifetimes: stripped_lifetimes,
            where_predicates: simplify::where_clauses(cx, where_predicates),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Method {
    pub generics: Generics,
    pub unsafety: hir::Unsafety,
    pub constness: hir::Constness,
    pub decl: FnDecl,
    pub abi: Abi,
}

impl<'a> Clean<Method> for (&'a hir::MethodSig, hir::BodyId) {
    fn clean(&self, cx: &DocContext) -> Method {
        Method {
            generics: self.0.generics.clean(cx),
            unsafety: self.0.unsafety,
            constness: self.0.constness,
            decl: (&*self.0.decl, self.1).clean(cx),
            abi: self.0.abi
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct TyMethod {
    pub unsafety: hir::Unsafety,
    pub decl: FnDecl,
    pub generics: Generics,
    pub abi: Abi,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Function {
    pub decl: FnDecl,
    pub generics: Generics,
    pub unsafety: hir::Unsafety,
    pub constness: hir::Constness,
    pub abi: Abi,
}

impl Clean<Item> for doctree::Function {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            inner: FunctionItem(Function {
                decl: (&self.decl, self.body).clean(cx),
                generics: self.generics.clean(cx),
                unsafety: self.unsafety,
                constness: self.constness,
                abi: self.abi,
            }),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct FnDecl {
    pub inputs: Arguments,
    pub output: FunctionRetTy,
    pub variadic: bool,
    pub attrs: Attributes,
}

impl FnDecl {
    pub fn has_self(&self) -> bool {
        self.inputs.values.len() > 0 && self.inputs.values[0].name == "self"
    }

    pub fn self_type(&self) -> Option<SelfTy> {
        self.inputs.values.get(0).and_then(|v| v.to_self())
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct Arguments {
    pub values: Vec<Argument>,
}

impl<'a> Clean<Arguments> for (&'a [P<hir::Ty>], &'a [Spanned<ast::Name>]) {
    fn clean(&self, cx: &DocContext) -> Arguments {
        Arguments {
            values: self.0.iter().enumerate().map(|(i, ty)| {
                let mut name = self.1.get(i).map(|n| n.node.to_string())
                                            .unwrap_or(String::new());
                if name.is_empty() {
                    name = "_".to_string();
                }
                Argument {
                    name: name,
                    type_: ty.clean(cx),
                }
            }).collect()
        }
    }
}

impl<'a> Clean<Arguments> for (&'a [P<hir::Ty>], hir::BodyId) {
    fn clean(&self, cx: &DocContext) -> Arguments {
        let body = cx.tcx.map.body(self.1);

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
    where (&'a [P<hir::Ty>], A): Clean<Arguments>
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

impl<'a, 'tcx> Clean<FnDecl> for (DefId, &'a ty::PolyFnSig<'tcx>) {
    fn clean(&self, cx: &DocContext) -> FnDecl {
        let (did, sig) = *self;
        let mut names = if cx.tcx.map.as_local_node_id(did).is_some() {
            vec![].into_iter()
        } else {
            cx.tcx.sess.cstore.fn_arg_names(did).into_iter()
        }.peekable();
        FnDecl {
            output: Return(sig.skip_binder().output().clean(cx)),
            attrs: Attributes::default(),
            variadic: sig.skip_binder().variadic,
            inputs: Arguments {
                values: sig.skip_binder().inputs().iter().map(|t| {
                    Argument {
                        type_: t.clean(cx),
                        name: names.next().map_or("".to_string(), |name| name.to_string()),
                    }
                }).collect(),
            },
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
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

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
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

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Trait {
    pub unsafety: hir::Unsafety,
    pub items: Vec<Item>,
    pub generics: Generics,
    pub bounds: Vec<TyParamBound>,
}

impl Clean<Item> for doctree::Trait {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: TraitItem(Trait {
                unsafety: self.unsafety,
                items: self.items.clean(cx),
                generics: self.generics.clean(cx),
                bounds: self.bounds.clean(cx),
            }),
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
            lifetimes: self.bound_lifetimes.clean(cx)
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
                MethodItem((sig, body).clean(cx))
            }
            hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Required(ref names)) => {
                TyMethodItem(TyMethod {
                    unsafety: sig.unsafety.clone(),
                    decl: (&*sig.decl, &names[..]).clean(cx),
                    generics: sig.generics.clean(cx),
                    abi: sig.abi
                })
            }
            hir::TraitItemKind::Type(ref bounds, ref default) => {
                AssociatedTypeItem(bounds.clean(cx), default.clean(cx))
            }
        };
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            visibility: None,
            stability: get_stability(cx, cx.tcx.map.local_def_id(self.id)),
            deprecation: get_deprecation(cx, cx.tcx.map.local_def_id(self.id)),
            inner: inner
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
                MethodItem((sig, body).clean(cx))
            }
            hir::ImplItemKind::Type(ref ty) => TypedefItem(Typedef {
                type_: ty.clean(cx),
                generics: Generics {
                    lifetimes: Vec::new(),
                    type_params: Vec::new(),
                    where_predicates: Vec::new()
                },
            }, true),
        };
        Item {
            name: Some(self.name.clean(cx)),
            source: self.span.clean(cx),
            attrs: self.attrs.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, cx.tcx.map.local_def_id(self.id)),
            deprecation: get_deprecation(cx, cx.tcx.map.local_def_id(self.id)),
            inner: inner
        }
    }
}

impl<'tcx> Clean<Item> for ty::AssociatedItem {
    fn clean(&self, cx: &DocContext) -> Item {
        let inner = match self.kind {
            ty::AssociatedKind::Const => {
                let ty = cx.tcx.item_type(self.def_id);
                AssociatedConstItem(ty.clean(cx), None)
            }
            ty::AssociatedKind::Method => {
                let generics = (cx.tcx.item_generics(self.def_id),
                                &cx.tcx.item_predicates(self.def_id)).clean(cx);
                let fty = match cx.tcx.item_type(self.def_id).sty {
                    ty::TyFnDef(_, _, f) => f,
                    _ => unreachable!()
                };
                let mut decl = (self.def_id, &fty.sig).clean(cx);

                if self.method_has_self_argument {
                    let self_ty = match self.container {
                        ty::ImplContainer(def_id) => {
                            cx.tcx.item_type(def_id)
                        }
                        ty::TraitContainer(_) => cx.tcx.mk_self_type()
                    };
                    let self_arg_ty = *fty.sig.input(0).skip_binder();
                    if self_arg_ty == self_ty {
                        decl.inputs.values[0].type_ = Generic(String::from("Self"));
                    } else if let ty::TyRef(_, mt) = self_arg_ty.sty {
                        if mt.ty == self_ty {
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
                    ty::ImplContainer(_) => false,
                    ty::TraitContainer(_) => self.defaultness.has_value()
                };
                if provided {
                    MethodItem(Method {
                        unsafety: fty.unsafety,
                        generics: generics,
                        decl: decl,
                        abi: fty.abi,

                        // trait methods canot (currently, at least) be const
                        constness: hir::Constness::NotConst,
                    })
                } else {
                    TyMethodItem(TyMethod {
                        unsafety: fty.unsafety,
                        generics: generics,
                        decl: decl,
                        abi: fty.abi,
                    })
                }
            }
            ty::AssociatedKind::Type => {
                let my_name = self.name.clean(cx);

                let mut bounds = if let ty::TraitContainer(did) = self.container {
                    // When loading a cross-crate associated type, the bounds for this type
                    // are actually located on the trait/impl itself, so we need to load
                    // all of the generics from there and then look for bounds that are
                    // applied to this associated type in question.
                    let predicates = cx.tcx.item_predicates(did);
                    let generics = (cx.tcx.item_generics(did), &predicates).clean(cx);
                    generics.where_predicates.iter().filter_map(|pred| {
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
                    }).flat_map(|i| i.iter().cloned()).collect::<Vec<_>>()
                } else {
                    vec![]
                };

                // Our Sized/?Sized bound didn't get handled when creating the generics
                // because we didn't actually get our whole set of bounds until just now
                // (some of them may have come from the trait). If we do have a sized
                // bound, we remove it, and if we don't then we add the `?Sized` bound
                // at the end.
                match bounds.iter().position(|b| b.is_sized_bound(cx)) {
                    Some(i) => { bounds.remove(i); }
                    None => bounds.push(TyParamBound::maybe_sized(cx)),
                }

                let ty = if self.defaultness.has_value() {
                    Some(cx.tcx.item_type(self.def_id))
                } else {
                    None
                };

                AssociatedTypeItem(bounds, ty.clean(cx))
            }
        };

        Item {
            name: Some(self.name.clean(cx)),
            visibility: Some(Inherited),
            stability: get_stability(cx, self.def_id),
            deprecation: get_deprecation(cx, self.def_id),
            def_id: self.def_id,
            attrs: inline::load_attrs(cx, self.def_id),
            source: cx.tcx.def_span(self.def_id).clean(cx),
            inner: inner,
        }
    }
}

/// A trait reference, which may have higher ranked lifetimes.
#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct PolyTrait {
    pub trait_: Type,
    pub lifetimes: Vec<Lifetime>
}

/// A representation of a Type suitable for hyperlinking purposes. Ideally one can get the original
/// type out of the AST/TyCtxt given one of these, if more information is needed. Most importantly
/// it does not preserve mutability or boxes.
#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub enum Type {
    /// structs/enums/traits (most that'd be an hir::TyPath)
    ResolvedPath {
        path: Path,
        typarams: Option<Vec<TyParamBound>>,
        did: DefId,
        /// true if is a `T::Name` path for associated types
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
    Vector(Box<Type>),
    FixedVector(Box<Type>, String),
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
    ImplTrait(Vec<TyParamBound>),
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
    RawPointer,
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
            Vector(..) | BorrowedRef{ type_: box Vector(..), ..  } => Some(PrimitiveType::Slice),
            FixedVector(..) | BorrowedRef { type_: box FixedVector(..), .. } => {
                Some(PrimitiveType::Array)
            }
            Tuple(..) => Some(PrimitiveType::Tuple),
            RawPointer(..) => Some(PrimitiveType::RawPointer),
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
}

impl GetDefId for Type {
    fn def_id(&self) -> Option<DefId> {
        match *self {
            ResolvedPath { did, .. } => Some(did),
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
            "pointer" => Some(PrimitiveType::RawPointer),
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
            RawPointer => "pointer",
        }
    }

    pub fn to_url_str(&self) -> &'static str {
        self.as_str()
    }
}

impl From<ast::IntTy> for PrimitiveType {
    fn from(int_ty: ast::IntTy) -> PrimitiveType {
        match int_ty {
            ast::IntTy::Is => PrimitiveType::Isize,
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
            ast::UintTy::Us => PrimitiveType::Usize,
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
            TyNever => Never,
            TyPtr(ref m) => RawPointer(m.mutbl.clean(cx), box m.ty.clean(cx)),
            TyRptr(ref l, ref m) =>
                BorrowedRef {lifetime: l.clean(cx), mutability: m.mutbl.clean(cx),
                             type_: box m.ty.clean(cx)},
            TySlice(ref ty) => Vector(box ty.clean(cx)),
            TyArray(ref ty, length) => {
                use rustc_const_eval::eval_length;
                let n = eval_length(cx.tcx, length, "array length").unwrap();
                FixedVector(box ty.clean(cx), n.to_string())
            },
            TyTup(ref tys) => Tuple(tys.clean(cx)),
            TyPath(hir::QPath::Resolved(None, ref path)) => {
                if let Some(new_ty) = cx.ty_substs.borrow().get(&path.def).cloned() {
                    return new_ty;
                }

                let mut alias = None;
                if let Def::TyAlias(def_id) = path.def {
                    // Substitute private type aliases
                    if let Some(node_id) = cx.tcx.map.as_local_node_id(def_id) {
                        if !cx.access_levels.borrow().is_exported(def_id) {
                            alias = Some(&cx.tcx.map.expect_item(node_id).node);
                        }
                    }
                };

                if let Some(&hir::ItemTy(ref ty, ref generics)) = alias {
                    let provided_params = &path.segments.last().unwrap().parameters;
                    let mut ty_substs = FxHashMap();
                    let mut lt_substs = FxHashMap();
                    for (i, ty_param) in generics.ty_params.iter().enumerate() {
                        let ty_param_def = Def::TyParam(cx.tcx.map.local_def_id(ty_param.id));
                        if let Some(ty) = provided_params.types().get(i).cloned()
                                                                        .cloned() {
                            ty_substs.insert(ty_param_def, ty.unwrap().clean(cx));
                        } else if let Some(default) = ty_param.default.clone() {
                            ty_substs.insert(ty_param_def, default.unwrap().clean(cx));
                        }
                    }
                    for (i, lt_param) in generics.lifetimes.iter().enumerate() {
                        if let Some(lt) = provided_params.lifetimes().get(i).cloned()
                                                                            .cloned() {
                            lt_substs.insert(lt_param.lifetime.id, lt.clean(cx));
                        }
                    }
                    return cx.enter_alias(ty_substs, lt_substs, || ty.clean(cx));
                }
                resolve_type(cx, path.clean(cx), self.id)
            }
            TyPath(hir::QPath::Resolved(Some(ref qself), ref p)) => {
                let mut segments: Vec<_> = p.segments.clone().into();
                segments.pop();
                let trait_path = hir::Path {
                    span: p.span,
                    def: Def::Trait(cx.tcx.associated_item(p.def.def_id()).container.id()),
                    segments: segments.into(),
                };
                Type::QPath {
                    name: p.segments.last().unwrap().name.clean(cx),
                    self_type: box qself.clean(cx),
                    trait_: box resolve_type(cx, trait_path.clean(cx), self.id)
                }
            }
            TyPath(hir::QPath::TypeRelative(ref qself, ref segment)) => {
                let mut def = Def::Err;
                if let Some(ty) = cx.hir_ty_to_ty.get(&self.id) {
                    if let ty::TyProjection(proj) = ty.sty {
                        def = Def::Trait(proj.trait_ref.def_id);
                    }
                }
                let trait_path = hir::Path {
                    span: self.span,
                    def: def,
                    segments: vec![].into(),
                };
                Type::QPath {
                    name: segment.name.clean(cx),
                    self_type: box qself.clean(cx),
                    trait_: box resolve_type(cx, trait_path.clean(cx), self.id)
                }
            }
            TyTraitObject(ref bounds) => {
                let lhs_ty = bounds[0].clean(cx);
                match lhs_ty {
                    TraitBound(poly_trait, ..) => {
                        match poly_trait.trait_ {
                            ResolvedPath { path, typarams: None, did, is_generic } => {
                                ResolvedPath {
                                    path: path,
                                    typarams: Some(bounds[1..].clean(cx)),
                                    did: did,
                                    is_generic: is_generic,
                                }
                            }
                            _ => Infer // shouldn't happen
                        }
                    }
                    _ => Infer // shouldn't happen
                }
            }
            TyBareFn(ref barefn) => BareFunction(box barefn.clean(cx)),
            TyImplTrait(ref bounds) => ImplTrait(bounds.clean(cx)),
            TyInfer => Infer,
            TyTypeof(..) => panic!("Unimplemented type {:?}", self.node),
        }
    }
}

impl<'tcx> Clean<Type> for ty::Ty<'tcx> {
    fn clean(&self, cx: &DocContext) -> Type {
        match self.sty {
            ty::TyNever => Never,
            ty::TyBool => Primitive(PrimitiveType::Bool),
            ty::TyChar => Primitive(PrimitiveType::Char),
            ty::TyInt(int_ty) => Primitive(int_ty.into()),
            ty::TyUint(uint_ty) => Primitive(uint_ty.into()),
            ty::TyFloat(float_ty) => Primitive(float_ty.into()),
            ty::TyStr => Primitive(PrimitiveType::Str),
            ty::TyBox(t) => {
                let box_did = cx.tcx.lang_items.owned_box();
                lang_struct(cx, box_did, t, "Box", Unique)
            }
            ty::TySlice(ty) => Vector(box ty.clean(cx)),
            ty::TyArray(ty, i) => FixedVector(box ty.clean(cx),
                                              format!("{}", i)),
            ty::TyRawPtr(mt) => RawPointer(mt.mutbl.clean(cx), box mt.ty.clean(cx)),
            ty::TyRef(r, mt) => BorrowedRef {
                lifetime: r.clean(cx),
                mutability: mt.mutbl.clean(cx),
                type_: box mt.ty.clean(cx),
            },
            ty::TyFnDef(.., ref fty) |
            ty::TyFnPtr(ref fty) => BareFunction(box BareFunctionDecl {
                unsafety: fty.unsafety,
                generics: Generics {
                    lifetimes: Vec::new(),
                    type_params: Vec::new(),
                    where_predicates: Vec::new()
                },
                decl: (cx.tcx.map.local_def_id(ast::CRATE_NODE_ID), &fty.sig).clean(cx),
                abi: fty.abi,
            }),
            ty::TyAdt(def, substs) => {
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
                    path: path,
                    typarams: None,
                    did: did,
                    is_generic: false,
                }
            }
            ty::TyDynamic(ref obj, ref reg) => {
                if let Some(principal) = obj.principal() {
                    let did = principal.def_id();
                    inline::record_extern_fqn(cx, did, TypeKind::Trait);

                    let mut typarams = vec![];
                    reg.clean(cx).map(|b| typarams.push(RegionBound(b)));
                    for did in obj.auto_traits() {
                        let empty = cx.tcx.intern_substs(&[]);
                        let path = external_path(cx, &cx.tcx.item_name(did).as_str(),
                            Some(did), false, vec![], empty);
                        inline::record_extern_fqn(cx, did, TypeKind::Trait);
                        let bound = TraitBound(PolyTrait {
                            trait_: ResolvedPath {
                                path: path,
                                typarams: None,
                                did: did,
                                is_generic: false,
                            },
                            lifetimes: vec![]
                        }, hir::TraitBoundModifier::None);
                        typarams.push(bound);
                    }

                    let mut bindings = vec![];
                    for ty::Binder(ref pb) in obj.projection_bounds() {
                        bindings.push(TypeBinding {
                            name: pb.item_name.clean(cx),
                            ty: pb.ty.clean(cx)
                        });
                    }

                    let path = external_path(cx, &cx.tcx.item_name(did).as_str(), Some(did),
                        false, bindings, principal.0.substs);
                    ResolvedPath {
                        path: path,
                        typarams: Some(typarams),
                        did: did,
                        is_generic: false,
                    }
                } else {
                    Never
                }
            }
            ty::TyTuple(ref t) => Tuple(t.clean(cx)),

            ty::TyProjection(ref data) => data.clean(cx),

            ty::TyParam(ref p) => Generic(p.name.to_string()),

            ty::TyAnon(def_id, substs) => {
                // Grab the "TraitA + TraitB" from `impl TraitA + TraitB`,
                // by looking up the projections associated with the def_id.
                let item_predicates = cx.tcx.item_predicates(def_id);
                let substs = cx.tcx.lift(&substs).unwrap();
                let bounds = item_predicates.instantiate(cx.tcx, substs);
                ImplTrait(bounds.predicates.into_iter().filter_map(|predicate| {
                    predicate.to_opt_poly_trait_ref().clean(cx)
                }).collect())
            }

            ty::TyClosure(..) => Tuple(vec![]), // FIXME(pcwalton)

            ty::TyInfer(..) => panic!("TyInfer"),
            ty::TyError => panic!("TyError"),
        }
    }
}

impl Clean<Item> for hir::StructField {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name).clean(cx),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, cx.tcx.map.local_def_id(self.id)),
            deprecation: get_deprecation(cx, cx.tcx.map.local_def_id(self.id)),
            def_id: cx.tcx.map.local_def_id(self.id),
            inner: StructFieldItem(self.ty.clean(cx)),
        }
    }
}

impl<'tcx> Clean<Item> for ty::FieldDef {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name).clean(cx),
            attrs: cx.tcx.get_attrs(self.did).clean(cx),
            source: cx.tcx.def_span(self.did).clean(cx),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, self.did),
            deprecation: get_deprecation(cx, self.did),
            def_id: self.did,
            inner: StructFieldItem(cx.tcx.item_type(self.did).clean(cx)),
        }
    }
}

#[derive(Clone, PartialEq, Eq, RustcDecodable, RustcEncodable, Debug)]
pub enum Visibility {
    Public,
    Inherited,
}

impl Clean<Option<Visibility>> for hir::Visibility {
    fn clean(&self, _: &DocContext) -> Option<Visibility> {
        Some(if *self == hir::Visibility::Public { Public } else { Inherited })
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
            def_id: cx.tcx.map.local_def_id(self.id),
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
            def_id: cx.tcx.map.local_def_id(self.id),
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
    pub variants: Vec<Item>,
    pub generics: Generics,
    pub variants_stripped: bool,
}

impl Clean<Item> for doctree::Enum {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: EnumItem(Enum {
                variants: self.variants.clean(cx),
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
            def_id: cx.tcx.map.local_def_id(self.def.id()),
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
                    self.fields.iter().map(|f| cx.tcx.item_type(f.did).clean(cx)).collect()
                )
            }
            CtorKind::Fictive => {
                VariantKind::Struct(VariantStruct {
                    struct_type: doctree::Plain,
                    fields_stripped: false,
                    fields: self.fields.iter().map(|field| {
                        Item {
                            source: cx.tcx.def_span(field.did).clean(cx),
                            name: Some(field.name.clean(cx)),
                            attrs: cx.tcx.get_attrs(field.did).clean(cx),
                            visibility: field.vis.clean(cx),
                            def_id: field.did,
                            stability: get_stability(cx, field.did),
                            deprecation: get_deprecation(cx, field.did),
                            inner: StructFieldItem(cx.tcx.item_type(field.did).clean(cx))
                        }
                    }).collect()
                })
            }
        };
        Item {
            name: Some(self.name.clean(cx)),
            attrs: inline::load_attrs(cx, self.did),
            source: cx.tcx.def_span(self.did).clean(cx),
            visibility: Some(Inherited),
            def_id: self.did,
            inner: VariantItem(Variant { kind: kind }),
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
    pub filename: String,
    pub loline: usize,
    pub locol: usize,
    pub hiline: usize,
    pub hicol: usize,
}

impl Span {
    fn empty() -> Span {
        Span {
            filename: "".to_string(),
            loline: 0, locol: 0,
            hiline: 0, hicol: 0,
        }
    }
}

impl Clean<Span> for syntax_pos::Span {
    fn clean(&self, cx: &DocContext) -> Span {
        if *self == DUMMY_SP {
            return Span::empty();
        }

        let cm = cx.sess().codemap();
        let filename = cm.span_to_filename(*self);
        let lo = cm.lookup_char_pos(self.lo);
        let hi = cm.lookup_char_pos(self.hi);
        Span {
            filename: filename.to_string(),
            loline: lo.line,
            locol: lo.col.to_usize(),
            hiline: hi.line,
            hicol: hi.col.to_usize(),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct Path {
    pub global: bool,
    pub def: Def,
    pub segments: Vec<PathSegment>,
}

impl Path {
    pub fn singleton(name: String) -> Path {
        Path {
            global: false,
            def: Def::Err,
            segments: vec![PathSegment {
                name: name,
                params: PathParameters::AngleBracketed {
                    lifetimes: Vec::new(),
                    types: Vec::new(),
                    bindings: Vec::new()
                }
            }]
        }
    }

    pub fn last_name(&self) -> &str {
        self.segments.last().unwrap().name.as_str()
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

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub enum PathParameters {
    AngleBracketed {
        lifetimes: Vec<Lifetime>,
        types: Vec<Type>,
        bindings: Vec<TypeBinding>
    },
    Parenthesized {
        inputs: Vec<Type>,
        output: Option<Type>
    }
}

impl Clean<PathParameters> for hir::PathParameters {
    fn clean(&self, cx: &DocContext) -> PathParameters {
        match *self {
            hir::AngleBracketedParameters(ref data) => {
                PathParameters::AngleBracketed {
                    lifetimes: data.lifetimes.clean(cx),
                    types: data.types.clean(cx),
                    bindings: data.bindings.clean(cx)
                }
            }

            hir::ParenthesizedParameters(ref data) => {
                PathParameters::Parenthesized {
                    inputs: data.inputs.clean(cx),
                    output: data.output.clean(cx)
                }
            }
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct PathSegment {
    pub name: String,
    pub params: PathParameters
}

impl Clean<PathSegment> for hir::PathSegment {
    fn clean(&self, cx: &DocContext) -> PathSegment {
        PathSegment {
            name: self.name.clean(cx),
            params: self.parameters.clean(cx)
        }
    }
}

fn qpath_to_string(p: &hir::QPath) -> String {
    let segments = match *p {
        hir::QPath::Resolved(_, ref path) => &path.segments,
        hir::QPath::TypeRelative(_, ref segment) => return segment.name.to_string(),
    };

    let mut s = String::new();
    for (i, seg) in segments.iter().enumerate() {
        if i > 0 {
            s.push_str("::");
        }
        if seg.name != keywords::CrateRoot.name() {
            s.push_str(&*seg.name.as_str());
        }
    }
    s
}

impl Clean<String> for ast::Name {
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
            def_id: cx.tcx.map.local_def_id(self.id.clone()),
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

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct BareFunctionDecl {
    pub unsafety: hir::Unsafety,
    pub generics: Generics,
    pub decl: FnDecl,
    pub abi: Abi,
}

impl Clean<BareFunctionDecl> for hir::BareFnTy {
    fn clean(&self, cx: &DocContext) -> BareFunctionDecl {
        BareFunctionDecl {
            unsafety: self.unsafety,
            generics: Generics {
                lifetimes: self.lifetimes.clean(cx),
                type_params: Vec::new(),
                where_predicates: Vec::new()
            },
            decl: (&*self.decl, &[][..]).clean(cx),
            abi: self.abi,
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
            def_id: cx.tcx.map.local_def_id(self.id),
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
            def_id: cx.tcx.map.local_def_id(self.id),
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

#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq, Copy)]
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

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Copy, Debug)]
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
}

impl Clean<Vec<Item>> for doctree::Impl {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {
        let mut ret = Vec::new();
        let trait_ = self.trait_.clean(cx);
        let items = self.items.clean(cx);

        // If this impl block is an implementation of the Deref trait, then we
        // need to try inlining the target's inherent impl blocks as well.
        if trait_.def_id() == cx.tcx.lang_items.deref_trait() {
            build_deref_target_impls(cx, &items, &mut ret);
        }

        let provided = trait_.def_id().map(|did| {
            cx.tcx.provided_trait_methods(did)
                  .into_iter()
                  .map(|meth| meth.name.to_string())
                  .collect()
        }).unwrap_or(FxHashSet());

        ret.push(Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            deprecation: self.depr.clean(cx),
            inner: ImplItem(Impl {
                unsafety: self.unsafety,
                generics: self.generics.clean(cx),
                provided_trait_methods: provided,
                trait_: trait_,
                for_: self.for_.clean(cx),
                items: items,
                polarity: Some(self.polarity.clean(cx)),
            }),
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
            Isize => tcx.lang_items.isize_impl(),
            I8 => tcx.lang_items.i8_impl(),
            I16 => tcx.lang_items.i16_impl(),
            I32 => tcx.lang_items.i32_impl(),
            I64 => tcx.lang_items.i64_impl(),
            I128 => tcx.lang_items.i128_impl(),
            Usize => tcx.lang_items.usize_impl(),
            U8 => tcx.lang_items.u8_impl(),
            U16 => tcx.lang_items.u16_impl(),
            U32 => tcx.lang_items.u32_impl(),
            U64 => tcx.lang_items.u64_impl(),
            U128 => tcx.lang_items.u128_impl(),
            F32 => tcx.lang_items.f32_impl(),
            F64 => tcx.lang_items.f64_impl(),
            Char => tcx.lang_items.char_impl(),
            Bool => None,
            Str => tcx.lang_items.str_impl(),
            Slice => tcx.lang_items.slice_impl(),
            Array => tcx.lang_items.slice_impl(),
            Tuple => None,
            RawPointer => tcx.lang_items.const_ptr_impl(),
        };
        if let Some(did) = did {
            if !did.is_local() {
                inline::build_impl(cx, did, ret);
            }
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct DefaultImpl {
    pub unsafety: hir::Unsafety,
    pub trait_: Type,
}

impl Clean<Item> for doctree::DefaultImpl {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            visibility: Some(Public),
            stability: None,
            deprecation: None,
            inner: DefaultImplItem(DefaultImpl {
                unsafety: self.unsafety,
                trait_: self.trait_.clean(cx),
            }),
        }
    }
}

impl Clean<Item> for doctree::ExternCrate {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: DefId { krate: self.cnum, index: CRATE_DEF_INDEX },
            visibility: self.vis.clean(cx),
            stability: None,
            deprecation: None,
            inner: ExternCrateItem(self.name.clean(cx), self.path.clone())
        }
    }
}

impl Clean<Vec<Item>> for doctree::Import {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {
        // We consider inlining the documentation of `pub use` statements, but we
        // forcefully don't inline if this is not public or if the
        // #[doc(no_inline)] attribute is present.
        // Don't inline doc(hidden) imports so they can be stripped at a later stage.
        let denied = self.vis != hir::Public || self.attrs.iter().any(|a| {
            a.name() == "doc" && match a.meta_item_list() {
                Some(l) => attr::list_contains_name(l, "no_inline") ||
                           attr::list_contains_name(l, "hidden"),
                None => false,
            }
        });
        let path = self.path.clean(cx);
        let inner = if self.glob {
            Import::Glob(resolve_use_source(cx, path))
        } else {
            let name = self.name;
            if !denied {
                if let Some(items) = inline::try_inline(cx, path.def, Some(name)) {
                    return items;
                }
            }
            Import::Simple(name.clean(cx), resolve_use_source(cx, path))
        };
        vec![Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.tcx.map.local_def_id(ast::CRATE_NODE_ID),
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
                f.abi = self.abi;
            }
        }
        items
    }
}

impl Clean<Item> for hir::ForeignItem {
    fn clean(&self, cx: &DocContext) -> Item {
        let inner = match self.node {
            hir::ForeignItemFn(ref decl, ref names, ref generics) => {
                ForeignFunctionItem(Function {
                    decl: (&**decl, &names[..]).clean(cx),
                    generics: generics.clean(cx),
                    unsafety: hir::Unsafety::Unsafe,
                    abi: Abi::Rust,
                    constness: hir::Constness::NotConst,
                })
            }
            hir::ForeignItemStatic(ref ty, mutbl) => {
                ForeignStaticItem(Static {
                    type_: ty.clean(cx),
                    mutability: if mutbl {Mutable} else {Immutable},
                    expr: "".to_string(),
                })
            }
        };
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.tcx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, cx.tcx.map.local_def_id(self.id)),
            deprecation: get_deprecation(cx, cx.tcx.map.local_def_id(self.id)),
            inner: inner,
        }
    }
}

// Utilities

trait ToSource {
    fn to_src(&self, cx: &DocContext) -> String;
}

impl ToSource for syntax_pos::Span {
    fn to_src(&self, cx: &DocContext) -> String {
        debug!("converting span {:?} to snippet", self.clean(cx));
        let sn = match cx.sess().codemap().span_to_snippet(*self) {
            Ok(x) => x.to_string(),
            Err(_) => "".to_string()
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
        PatKind::Binding(_, _, ref p, _) => p.node.to_string(),
        PatKind::TupleStruct(ref p, ..) | PatKind::Path(ref p) => qpath_to_string(p),
        PatKind::Struct(ref name, ref fields, etc) => {
            format!("{} {{ {}{} }}", qpath_to_string(name),
                fields.iter().map(|&Spanned { node: ref fp, .. }|
                                  format!("{}: {}", fp.name, name_from_pat(&*fp.pat)))
                             .collect::<Vec<String>>().join(", "),
                if etc { ", ..." } else { "" }
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

fn print_const_expr(cx: &DocContext, body: hir::BodyId) -> String {
    cx.tcx.map.node_to_pretty_string(body.node_id)
}

/// Given a type Path, resolve it to a Type using the TyCtxt
fn resolve_type(cx: &DocContext,
                path: Path,
                id: ast::NodeId) -> Type {
    debug!("resolve_type({:?},{:?})", path, id);

    let is_generic = match path.def {
        Def::PrimTy(p) => match p {
            hir::TyStr => return Primitive(PrimitiveType::Str),
            hir::TyBool => return Primitive(PrimitiveType::Bool),
            hir::TyChar => return Primitive(PrimitiveType::Char),
            hir::TyInt(int_ty) => return Primitive(int_ty.into()),
            hir::TyUint(uint_ty) => return Primitive(uint_ty.into()),
            hir::TyFloat(float_ty) => return Primitive(float_ty.into()),
        },
        Def::SelfTy(..) if path.segments.len() == 1 => {
            return Generic(keywords::SelfType.name().to_string());
        }
        Def::SelfTy(..) | Def::TyParam(..) | Def::AssociatedTy(..) => true,
        _ => false,
    };
    let did = register_def(&*cx, path.def);
    ResolvedPath { path: path, typarams: None, did: did, is_generic: is_generic }
}

fn register_def(cx: &DocContext, def: Def) -> DefId {
    debug!("register_def({:?})", def);

    let (did, kind) = match def {
        Def::Fn(i) => (i, TypeKind::Function),
        Def::TyAlias(i) => (i, TypeKind::Typedef),
        Def::Enum(i) => (i, TypeKind::Enum),
        Def::Trait(i) => (i, TypeKind::Trait),
        Def::Struct(i) => (i, TypeKind::Struct),
        Def::Union(i) => (i, TypeKind::Union),
        Def::Mod(i) => (i, TypeKind::Module),
        Def::Static(i, _) => (i, TypeKind::Static),
        Def::Variant(i) => (cx.tcx.parent_def_id(i).unwrap(), TypeKind::Enum),
        Def::SelfTy(Some(def_id), _) => (def_id, TypeKind::Trait),
        Def::SelfTy(_, Some(impl_def_id)) => {
            return impl_def_id
        }
        _ => return def.def_id()
    };
    if did.is_local() { return did }
    inline::record_extern_fqn(cx, did, kind);
    if let TypeKind::Trait = kind {
        let t = inline::build_external_trait(cx, did);
        cx.external_traits.borrow_mut().insert(did, t);
    }
    did
}

fn resolve_use_source(cx: &DocContext, path: Path) -> ImportSource {
    ImportSource {
        did: if path.def == Def::Err {
            None
        } else {
            Some(register_def(cx, path.def))
        },
        path: path,
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
pub struct Stability {
    pub level: stability::StabilityLevel,
    pub feature: String,
    pub since: String,
    pub deprecated_since: String,
    pub deprecated_reason: String,
    pub unstable_reason: String,
    pub issue: Option<u32>
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Deprecation {
    pub since: String,
    pub note: String,
}

impl Clean<Stability> for attr::Stability {
    fn clean(&self, _: &DocContext) -> Stability {
        Stability {
            level: stability::StabilityLevel::from_attr_level(&self.level),
            feature: self.feature.to_string(),
            since: match self.level {
                attr::Stable {ref since} => since.to_string(),
                _ => "".to_string(),
            },
            deprecated_since: match self.rustc_depr {
                Some(attr::RustcDeprecation {ref since, ..}) => since.to_string(),
                _=> "".to_string(),
            },
            deprecated_reason: match self.rustc_depr {
                Some(ref depr) => depr.reason.to_string(),
                _ => "".to_string(),
            },
            unstable_reason: match self.level {
                attr::Unstable { reason: Some(ref reason), .. } => reason.to_string(),
                _ => "".to_string(),
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
            since: self.since.as_ref().map_or("".to_string(), |s| s.to_string()),
            note: self.note.as_ref().map_or("".to_string(), |s| s.to_string()),
        }
    }
}

fn lang_struct(cx: &DocContext, did: Option<DefId>,
               t: ty::Ty, name: &str,
               fallback: fn(Box<Type>) -> Type) -> Type {
    let did = match did {
        Some(did) => did,
        None => return fallback(box t.clean(cx)),
    };
    inline::record_extern_fqn(cx, did, TypeKind::Struct);
    ResolvedPath {
        typarams: None,
        did: did,
        path: Path {
            global: false,
            def: Def::Err,
            segments: vec![PathSegment {
                name: name.to_string(),
                params: PathParameters::AngleBracketed {
                    lifetimes: vec![],
                    types: vec![t.clean(cx)],
                    bindings: vec![]
                }
            }],
        },
        is_generic: false,
    }
}

/// An equality constraint on an associated type, e.g. `A=Bar` in `Foo<A=Bar>`
#[derive(Clone, PartialEq, RustcDecodable, RustcEncodable, Debug)]
pub struct TypeBinding {
    pub name: String,
    pub ty: Type
}

impl Clean<TypeBinding> for hir::TypeBinding {
    fn clean(&self, cx: &DocContext) -> TypeBinding {
        TypeBinding {
            name: self.name.clean(cx),
            ty: self.ty.clean(cx)
        }
    }
}
