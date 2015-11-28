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
pub use self::PrimitiveType::*;
pub use self::TypeKind::*;
pub use self::StructField::*;
pub use self::VariantKind::*;
pub use self::Mutability::*;
pub use self::Import::*;
pub use self::ItemEnum::*;
pub use self::Attribute::*;
pub use self::TyParamBound::*;
pub use self::SelfTy::*;
pub use self::FunctionRetTy::*;

use syntax;
use syntax::abi;
use syntax::ast;
use syntax::attr;
use syntax::attr::{AttributeMethods, AttrMetaMethods};
use syntax::codemap;
use syntax::codemap::{DUMMY_SP, Pos, Spanned};
use syntax::parse::token::{self, InternedString, special_idents};
use syntax::ptr::P;

use rustc_trans::back::link;
use rustc::middle::cstore::{self, CrateStore};
use rustc::middle::def;
use rustc::middle::def_id::{DefId, DefIndex};
use rustc::middle::subst::{self, ParamSpace, VecPerParamSpace};
use rustc::middle::ty;
use rustc::middle::stability;

use rustc_front::hir;

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::u32;

use core::DocContext;
use doctree;
use visit_ast;

/// A stable identifier to the particular version of JSON output.
/// Increment this when the `Crate` and related structures change.
pub const SCHEMA_VERSION: &'static str = "0.8.3";

mod inline;
mod simplify;

// extract the stability index for a node from tcx, if possible
fn get_stability(cx: &DocContext, def_id: DefId) -> Option<Stability> {
    cx.tcx_opt().and_then(|tcx| stability::lookup(tcx, def_id)).clean(cx)
}

pub trait Clean<T> {
    fn clean(&self, cx: &DocContext) -> T;
}

impl<T: Clean<U>, U> Clean<Vec<U>> for [T] {
    fn clean(&self, cx: &DocContext) -> Vec<U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

impl<T: Clean<U>, U> Clean<VecPerParamSpace<U>> for VecPerParamSpace<T> {
    fn clean(&self, cx: &DocContext) -> VecPerParamSpace<U> {
        self.map(|x| x.clean(cx))
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
        match self {
            &None => None,
            &Some(ref v) => Some(v.clean(cx))
        }
    }
}

impl<T, U> Clean<U> for ty::Binder<T> where T: Clean<U> {
    fn clean(&self, cx: &DocContext) -> U {
        self.0.clean(cx)
    }
}

impl<T: Clean<U>, U> Clean<Vec<U>> for syntax::owned_slice::OwnedSlice<T> {
    fn clean(&self, cx: &DocContext) -> Vec<U> {
        self.iter().map(|x| x.clean(cx)).collect()
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Crate {
    pub name: String,
    pub src: PathBuf,
    pub module: Option<Item>,
    pub externs: Vec<(ast::CrateNum, ExternalCrate)>,
    pub primitives: Vec<PrimitiveType>,
    pub external_traits: HashMap<DefId, Trait>,
}

struct CrateNum(ast::CrateNum);

impl<'a, 'tcx> Clean<Crate> for visit_ast::RustdocVisitor<'a, 'tcx> {
    fn clean(&self, cx: &DocContext) -> Crate {
        use rustc::session::config::Input;

        if let Some(t) = cx.tcx_opt() {
            cx.deref_trait_did.set(t.lang_items.deref_trait());
        }

        let mut externs = Vec::new();
        for cnum in cx.sess().cstore.crates() {
            externs.push((cnum, CrateNum(cnum).clean(cx)));
        }
        externs.sort_by(|&(a, _), &(b, _)| a.cmp(&b));

        // Figure out the name of this crate
        let input = &cx.input;
        let name = link::find_crate_name(None, &self.attrs, input);

        // Clean the crate, translating the entire libsyntax AST to one that is
        // understood by rustdoc.
        let mut module = self.module.clean(cx);

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
        let mut primitives = Vec::new();
        {
            let m = match module.inner {
                ModuleItem(ref mut m) => m,
                _ => unreachable!(),
            };
            let mut tmp = Vec::new();
            for child in &mut m.items {
                match child.inner {
                    ModuleItem(..) => {}
                    _ => continue,
                }
                let prim = match PrimitiveType::find(&child.attrs) {
                    Some(prim) => prim,
                    None => continue,
                };
                primitives.push(prim);
                tmp.push(Item {
                    source: Span::empty(),
                    name: Some(prim.to_url_str().to_string()),
                    attrs: child.attrs.clone(),
                    visibility: Some(hir::Public),
                    stability: None,
                    def_id: DefId::local(prim.to_def_index()),
                    inner: PrimitiveItem(prim),
                });
            }
            m.items.extend(tmp);
        }

        let src = match cx.input {
            Input::File(ref path) => path.clone(),
            Input::Str(_) => PathBuf::new() // FIXME: this is wrong
        };

        Crate {
            name: name.to_string(),
            src: src,
            module: Some(module),
            externs: externs,
            primitives: primitives,
            external_traits: cx.external_traits.borrow_mut().take()
                               .unwrap_or(HashMap::new()),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ExternalCrate {
    pub name: String,
    pub attrs: Vec<Attribute>,
    pub primitives: Vec<PrimitiveType>,
}

impl Clean<ExternalCrate> for CrateNum {
    fn clean(&self, cx: &DocContext) -> ExternalCrate {
        let mut primitives = Vec::new();
        cx.tcx_opt().map(|tcx| {
            for item in tcx.sess.cstore.crate_top_level_items(self.0) {
                let did = match item.def {
                    cstore::DlDef(def::DefMod(did)) => did,
                    _ => continue
                };
                let attrs = inline::load_attrs(cx, tcx, did);
                PrimitiveType::find(&attrs).map(|prim| primitives.push(prim));
            }
        });
        ExternalCrate {
            name: cx.sess().cstore.crate_name(self.0),
            attrs: cx.sess().cstore.crate_attrs(self.0).clean(cx),
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
    pub attrs: Vec<Attribute> ,
    pub inner: ItemEnum,
    pub visibility: Option<Visibility>,
    pub def_id: DefId,
    pub stability: Option<Stability>,
}

impl Item {
    /// Finds the `doc` attribute as a List and returns the list of attributes
    /// nested inside.
    pub fn doc_list<'a>(&'a self) -> Option<&'a [Attribute]> {
        for attr in &self.attrs {
            match *attr {
                List(ref x, ref list) if "doc" == *x => {
                    return Some(list);
                }
                _ => {}
            }
        }
        return None;
    }

    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    pub fn doc_value<'a>(&'a self) -> Option<&'a str> {
        for attr in &self.attrs {
            match *attr {
                NameValue(ref x, ref v) if "doc" == *x => {
                    return Some(v);
                }
                _ => {}
            }
        }
        return None;
    }

    pub fn is_hidden_from_doc(&self) -> bool {
        match self.doc_list() {
            Some(l) => {
                for innerattr in l {
                    match *innerattr {
                        Word(ref s) if "hidden" == *s => {
                            return true
                        }
                        _ => (),
                    }
                }
            },
            None => ()
        }
        return false;
    }

    pub fn is_mod(&self) -> bool {
        match self.inner { ModuleItem(..) => true, _ => false }
    }
    pub fn is_trait(&self) -> bool {
        match self.inner { TraitItem(..) => true, _ => false }
    }
    pub fn is_struct(&self) -> bool {
        match self.inner { StructItem(..) => true, _ => false }
    }
    pub fn is_enum(&self) -> bool {
        match self.inner { EnumItem(..) => true, _ => false }
    }
    pub fn is_fn(&self) -> bool {
        match self.inner { FunctionItem(..) => true, _ => false }
    }

    pub fn stability_class(&self) -> String {
        match self.stability {
            Some(ref s) => {
                let mut base = match s.level {
                    stability::Unstable => "unstable".to_string(),
                    stability::Stable => String::new(),
                };
                if !s.deprecated_since.is_empty() {
                    base.push_str(" deprecated");
                }
                base
            }
            _ => String::new(),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum ItemEnum {
    ExternCrateItem(String, Option<String>),
    ImportItem(Import),
    StructItem(Struct),
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
    StructFieldItem(StructField),
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
            def_id: cx.map.local_def_id(self.id),
            inner: ModuleItem(Module {
               is_crate: self.is_crate,
               items: items
            })
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub enum Attribute {
    Word(String),
    List(String, Vec<Attribute> ),
    NameValue(String, String)
}

impl Clean<Attribute> for ast::MetaItem {
    fn clean(&self, cx: &DocContext) -> Attribute {
        match self.node {
            ast::MetaWord(ref s) => Word(s.to_string()),
            ast::MetaList(ref s, ref l) => {
                List(s.to_string(), l.clean(cx))
            }
            ast::MetaNameValue(ref s, ref v) => {
                NameValue(s.to_string(), lit_to_string(v))
            }
        }
    }
}

impl Clean<Attribute> for ast::Attribute {
    fn clean(&self, cx: &DocContext) -> Attribute {
        self.with_desugared_doc(|a| a.node.value.clean(cx))
    }
}

// This is a rough approximation that gets us what we want.
impl attr::AttrMetaMethods for Attribute {
    fn name(&self) -> InternedString {
        match *self {
            Word(ref n) | List(ref n, _) | NameValue(ref n, _) => {
                token::intern_and_get_ident(n)
            }
        }
    }

    fn value_str(&self) -> Option<InternedString> {
        match *self {
            NameValue(_, ref v) => {
                Some(token::intern_and_get_ident(v))
            }
            _ => None,
        }
    }
    fn meta_item_list<'a>(&'a self) -> Option<&'a [P<ast::MetaItem>]> { None }
    fn span(&self) -> codemap::Span { unimplemented!() }
}
impl<'a> attr::AttrMetaMethods for &'a Attribute {
    fn name(&self) -> InternedString { (**self).name() }
    fn value_str(&self) -> Option<InternedString> { (**self).value_str() }
    fn meta_item_list(&self) -> Option<&[P<ast::MetaItem>]> { None }
    fn span(&self) -> codemap::Span { unimplemented!() }
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
            did: cx.map.local_def_id(self.id),
            bounds: self.bounds.clean(cx),
            default: self.default.clean(cx),
        }
    }
}

impl<'tcx> Clean<TyParam> for ty::TypeParameterDef<'tcx> {
    fn clean(&self, cx: &DocContext) -> TyParam {
        cx.external_typarams.borrow_mut().as_mut().unwrap()
          .insert(self.def_id, self.name.clean(cx));
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
        use rustc_front::hir::TraitBoundModifier as TBM;
        let mut sized_bound = ty::BoundSized.clean(cx);
        if let TyParamBound::TraitBound(_, ref mut tbm) = sized_bound {
            *tbm = TBM::Maybe
        };
        sized_bound
    }

    fn is_sized_bound(&self, cx: &DocContext) -> bool {
        use rustc_front::hir::TraitBoundModifier as TBM;
        if let Some(tcx) = cx.tcx_opt() {
            let sized_did = match tcx.lang_items.sized_trait() {
                Some(did) => did,
                None => return false
            };
            if let TyParamBound::TraitBound(PolyTrait {
                trait_: Type::ResolvedPath { did, .. }, ..
            }, TBM::None) = *self {
                if did == sized_did {
                    return true
                }
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

impl<'tcx> Clean<(Vec<TyParamBound>, Vec<TypeBinding>)> for ty::ExistentialBounds<'tcx> {
    fn clean(&self, cx: &DocContext) -> (Vec<TyParamBound>, Vec<TypeBinding>) {
        let mut tp_bounds = vec![];
        self.region_bound.clean(cx).map(|b| tp_bounds.push(RegionBound(b)));
        for bb in &self.builtin_bounds {
            tp_bounds.push(bb.clean(cx));
        }

        let mut bindings = vec![];
        for &ty::Binder(ref pb) in &self.projection_bounds {
            bindings.push(TypeBinding {
                name: pb.projection_ty.item_name.clean(cx),
                ty: pb.ty.clean(cx)
            });
        }

        (tp_bounds, bindings)
    }
}

fn external_path_params(cx: &DocContext, trait_did: Option<DefId>,
                        bindings: Vec<TypeBinding>, substs: &subst::Substs) -> PathParameters {
    let lifetimes = substs.regions().get_slice(subst::TypeSpace)
                    .iter()
                    .filter_map(|v| v.clean(cx))
                    .collect();
    let types = substs.types.get_slice(subst::TypeSpace).to_vec();

    match (trait_did, cx.tcx_opt()) {
        // Attempt to sugar an external path like Fn<(A, B,), C> to Fn(A, B) -> C
        (Some(did), Some(ref tcx)) if tcx.lang_items.fn_trait_kind(did).is_some() => {
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
        (_, _) => {
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
fn external_path(cx: &DocContext, name: &str, trait_did: Option<DefId>,
                 bindings: Vec<TypeBinding>, substs: &subst::Substs) -> Path {
    Path {
        global: false,
        segments: vec![PathSegment {
            name: name.to_string(),
            params: external_path_params(cx, trait_did, bindings, substs)
        }],
    }
}

impl Clean<TyParamBound> for ty::BuiltinBound {
    fn clean(&self, cx: &DocContext) -> TyParamBound {
        let tcx = match cx.tcx_opt() {
            Some(tcx) => tcx,
            None => return RegionBound(Lifetime::statik())
        };
        let empty = subst::Substs::empty();
        let (did, path) = match *self {
            ty::BoundSend =>
                (tcx.lang_items.send_trait().unwrap(),
                 external_path(cx, "Send", None, vec![], &empty)),
            ty::BoundSized =>
                (tcx.lang_items.sized_trait().unwrap(),
                 external_path(cx, "Sized", None, vec![], &empty)),
            ty::BoundCopy =>
                (tcx.lang_items.copy_trait().unwrap(),
                 external_path(cx, "Copy", None, vec![], &empty)),
            ty::BoundSync =>
                (tcx.lang_items.sync_trait().unwrap(),
                 external_path(cx, "Sync", None, vec![], &empty)),
        };
        let fqn = tcx.sess.cstore.item_path(did);
        let fqn = fqn.into_iter().map(|i| i.to_string()).collect();
        cx.external_paths.borrow_mut().as_mut().unwrap().insert(did,
                                                                (fqn, TypeTrait));
        TraitBound(PolyTrait {
            trait_: ResolvedPath {
                path: path,
                typarams: None,
                did: did,
                is_generic: false,
            },
            lifetimes: vec![]
        }, hir::TraitBoundModifier::None)
    }
}

impl<'tcx> Clean<TyParamBound> for ty::TraitRef<'tcx> {
    fn clean(&self, cx: &DocContext) -> TyParamBound {
        let tcx = match cx.tcx_opt() {
            Some(tcx) => tcx,
            None => return RegionBound(Lifetime::statik())
        };
        let fqn = tcx.sess.cstore.item_path(self.def_id);
        let fqn = fqn.into_iter().map(|i| i.to_string())
                     .collect::<Vec<String>>();
        let path = external_path(cx, fqn.last().unwrap(),
                                 Some(self.def_id), vec![], self.substs);
        cx.external_paths.borrow_mut().as_mut().unwrap().insert(self.def_id,
                                                            (fqn, TypeTrait));

        debug!("ty::TraitRef\n  substs.types(TypeSpace): {:?}\n",
               self.substs.types.get_slice(ParamSpace::TypeSpace));

        // collect any late bound regions
        let mut late_bounds = vec![];
        for &ty_s in self.substs.types.get_slice(ParamSpace::TypeSpace) {
            if let ty::TyTuple(ref ts) = ty_s.sty {
                for &ty_s in ts {
                    if let ty::TyRef(ref reg, _) = ty_s.sty {
                        if let &ty::Region::ReLateBound(_, _) = *reg {
                            debug!("  hit an ReLateBound {:?}", reg);
                            if let Some(lt) = reg.clean(cx) {
                                late_bounds.push(lt)
                            }
                        }
                    }
                }
            }
        }

        TraitBound(PolyTrait {
            trait_: ResolvedPath {
                path: path,
                typarams: None,
                did: self.def_id,
                is_generic: false,
            },
            lifetimes: late_bounds
        }, hir::TraitBoundModifier::None)
    }
}

impl<'tcx> Clean<Option<Vec<TyParamBound>>> for subst::Substs<'tcx> {
    fn clean(&self, cx: &DocContext) -> Option<Vec<TyParamBound>> {
        let mut v = Vec::new();
        v.extend(self.regions().iter().filter_map(|r| r.clean(cx)).map(RegionBound));
        v.extend(self.types.iter().map(|t| TraitBound(PolyTrait {
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
        return s;
    }

    pub fn statik() -> Lifetime {
        Lifetime("'static".to_string())
    }
}

impl Clean<Lifetime> for hir::Lifetime {
    fn clean(&self, _: &DocContext) -> Lifetime {
        Lifetime(self.name.to_string())
    }
}

impl Clean<Lifetime> for hir::LifetimeDef {
    fn clean(&self, _: &DocContext) -> Lifetime {
        Lifetime(self.lifetime.name.to_string())
    }
}

impl Clean<Lifetime> for ty::RegionParameterDef {
    fn clean(&self, _: &DocContext) -> Lifetime {
        Lifetime(self.name.to_string())
    }
}

impl Clean<Option<Lifetime>> for ty::Region {
    fn clean(&self, cx: &DocContext) -> Option<Lifetime> {
        match *self {
            ty::ReStatic => Some(Lifetime::statik()),
            ty::ReLateBound(_, ty::BrNamed(_, name)) =>
                Some(Lifetime(name.to_string())),
            ty::ReEarlyBound(ref data) => Some(Lifetime(data.name.clean(cx))),

            ty::ReLateBound(..) |
            ty::ReFree(..) |
            ty::ReScope(..) |
            ty::ReVar(..) |
            ty::ReSkolemized(..) |
            ty::ReEmpty => None
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
        use rustc::middle::ty::Predicate;

        match *self {
            Predicate::Trait(ref pred) => pred.clean(cx),
            Predicate::Equate(ref pred) => pred.clean(cx),
            Predicate::RegionOutlives(ref pred) => pred.clean(cx),
            Predicate::TypeOutlives(ref pred) => pred.clean(cx),
            Predicate::Projection(ref pred) => pred.clean(cx),
            Predicate::WellFormed(_) => panic!("not user writable"),
            Predicate::ObjectSafe(_) => panic!("not user writable"),
        }
    }
}

impl<'a> Clean<WherePredicate> for ty::TraitPredicate<'a> {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        WherePredicate::BoundPredicate {
            ty: self.trait_ref.substs.self_ty().clean(cx).unwrap(),
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

impl Clean<WherePredicate> for ty::OutlivesPredicate<ty::Region, ty::Region> {
    fn clean(&self, cx: &DocContext) -> WherePredicate {
        let ty::OutlivesPredicate(ref a, ref b) = *self;
        WherePredicate::RegionPredicate {
            lifetime: a.clean(cx).unwrap(),
            bounds: vec![b.clean(cx).unwrap()]
        }
    }
}

impl<'tcx> Clean<WherePredicate> for ty::OutlivesPredicate<ty::Ty<'tcx>, ty::Region> {
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
                                    &'a ty::GenericPredicates<'tcx>,
                                    subst::ParamSpace) {
    fn clean(&self, cx: &DocContext) -> Generics {
        use std::collections::HashSet;
        use self::WherePredicate as WP;

        let (gens, preds, space) = *self;

        // Bounds in the type_params and lifetimes fields are repeated in the
        // predicates field (see rustc_typeck::collect::ty_generics), so remove
        // them.
        let stripped_typarams = gens.types.get_slice(space).iter().map(|tp| {
            tp.clean(cx)
        }).collect::<Vec<_>>();
        let stripped_lifetimes = gens.regions.get_slice(space).iter().map(|rp| {
            let mut srp = rp.clone();
            srp.bounds = Vec::new();
            srp.clean(cx)
        }).collect::<Vec<_>>();

        let mut where_predicates = preds.predicates.get_slice(space)
                                                   .to_vec().clean(cx);

        // Type parameters and have a Sized bound by default unless removed with
        // ?Sized.  Scan through the predicates and mark any type parameter with
        // a Sized bound, removing the bounds as we find them.
        //
        // Note that associated types also have a sized bound by default, but we
        // don't actually know the set of associated types right here so that's
        // handled in cleaning associated types
        let mut sized_params = HashSet::new();
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
    pub self_: SelfTy,
    pub unsafety: hir::Unsafety,
    pub constness: hir::Constness,
    pub decl: FnDecl,
    pub abi: abi::Abi
}

impl Clean<Method> for hir::MethodSig {
    fn clean(&self, cx: &DocContext) -> Method {
        let all_inputs = &self.decl.inputs;
        let inputs = match self.explicit_self.node {
            hir::SelfStatic => &**all_inputs,
            _ => &all_inputs[1..]
        };
        let decl = FnDecl {
            inputs: Arguments {
                values: inputs.clean(cx),
            },
            output: self.decl.output.clean(cx),
            variadic: false,
            attrs: Vec::new()
        };
        Method {
            generics: self.generics.clean(cx),
            self_: self.explicit_self.node.clean(cx),
            unsafety: self.unsafety,
            constness: self.constness,
            decl: decl,
            abi: self.abi
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct TyMethod {
    pub unsafety: hir::Unsafety,
    pub decl: FnDecl,
    pub generics: Generics,
    pub self_: SelfTy,
    pub abi: abi::Abi
}

impl Clean<TyMethod> for hir::MethodSig {
    fn clean(&self, cx: &DocContext) -> TyMethod {
        let inputs = match self.explicit_self.node {
            hir::SelfStatic => &*self.decl.inputs,
            _ => &self.decl.inputs[1..]
        };
        let decl = FnDecl {
            inputs: Arguments {
                values: inputs.clean(cx),
            },
            output: self.decl.output.clean(cx),
            variadic: false,
            attrs: Vec::new()
        };
        TyMethod {
            unsafety: self.unsafety.clone(),
            decl: decl,
            self_: self.explicit_self.node.clean(cx),
            generics: self.generics.clean(cx),
            abi: self.abi
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub enum SelfTy {
    SelfStatic,
    SelfValue,
    SelfBorrowed(Option<Lifetime>, Mutability),
    SelfExplicit(Type),
}

impl Clean<SelfTy> for hir::ExplicitSelf_ {
    fn clean(&self, cx: &DocContext) -> SelfTy {
        match *self {
            hir::SelfStatic => SelfStatic,
            hir::SelfValue(_) => SelfValue,
            hir::SelfRegion(ref lt, ref mt, _) => {
                SelfBorrowed(lt.clean(cx), mt.clean(cx))
            }
            hir::SelfExplicit(ref typ, _) => SelfExplicit(typ.clean(cx)),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Function {
    pub decl: FnDecl,
    pub generics: Generics,
    pub unsafety: hir::Unsafety,
    pub constness: hir::Constness,
    pub abi: abi::Abi,
}

impl Clean<Item> for doctree::Function {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            def_id: cx.map.local_def_id(self.id),
            inner: FunctionItem(Function {
                decl: self.decl.clean(cx),
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
    pub attrs: Vec<Attribute>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub struct Arguments {
    pub values: Vec<Argument>,
}

impl Clean<FnDecl> for hir::FnDecl {
    fn clean(&self, cx: &DocContext) -> FnDecl {
        FnDecl {
            inputs: Arguments {
                values: self.inputs.clean(cx),
            },
            output: self.output.clean(cx),
            variadic: self.variadic,
            attrs: Vec::new()
        }
    }
}

impl<'tcx> Clean<Type> for ty::FnOutput<'tcx> {
    fn clean(&self, cx: &DocContext) -> Type {
        match *self {
            ty::FnConverging(ty) => ty.clean(cx),
            ty::FnDiverging => Bottom
        }
    }
}

impl<'a, 'tcx> Clean<FnDecl> for (DefId, &'a ty::PolyFnSig<'tcx>) {
    fn clean(&self, cx: &DocContext) -> FnDecl {
        let (did, sig) = *self;
        let mut names = if let Some(_) = cx.map.as_local_node_id(did) {
            vec![].into_iter()
        } else {
            cx.tcx().sess.cstore.method_arg_names(did).into_iter()
        }.peekable();
        if names.peek().map(|s| &**s) == Some("self") {
            let _ = names.next();
        }
        FnDecl {
            output: Return(sig.0.output.clean(cx)),
            attrs: Vec::new(),
            variadic: sig.0.variadic,
            inputs: Arguments {
                values: sig.0.inputs.iter().map(|t| {
                    Argument {
                        type_: t.clean(cx),
                        id: 0,
                        name: names.next().unwrap_or("".to_string()),
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
    pub id: ast::NodeId,
}

impl Clean<Argument> for hir::Arg {
    fn clean(&self, cx: &DocContext) -> Argument {
        Argument {
            name: name_from_pat(&*self.pat),
            type_: (self.ty.clean(cx)),
            id: self.id
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug)]
pub enum FunctionRetTy {
    Return(Type),
    DefaultReturn,
    NoReturn
}

impl Clean<FunctionRetTy> for hir::FunctionRetTy {
    fn clean(&self, cx: &DocContext) -> FunctionRetTy {
        match *self {
            hir::Return(ref typ) => Return(typ.clean(cx)),
            hir::DefaultReturn(..) => DefaultReturn,
            hir::NoReturn(..) => NoReturn
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
            def_id: cx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
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
            hir::ConstTraitItem(ref ty, ref default) => {
                AssociatedConstItem(ty.clean(cx),
                                    default.as_ref().map(|expr|
                                                         expr.span.to_src(cx)))
            }
            hir::MethodTraitItem(ref sig, Some(_)) => {
                MethodItem(sig.clean(cx))
            }
            hir::MethodTraitItem(ref sig, None) => {
                TyMethodItem(sig.clean(cx))
            }
            hir::TypeTraitItem(ref bounds, ref default) => {
                AssociatedTypeItem(bounds.clean(cx), default.clean(cx))
            }
        };
        Item {
            name: Some(self.name.clean(cx)),
            attrs: self.attrs.clean(cx),
            source: self.span.clean(cx),
            def_id: cx.map.local_def_id(self.id),
            visibility: None,
            stability: get_stability(cx, cx.map.local_def_id(self.id)),
            inner: inner
        }
    }
}

impl Clean<Item> for hir::ImplItem {
    fn clean(&self, cx: &DocContext) -> Item {
        let inner = match self.node {
            hir::ImplItemKind::Const(ref ty, ref expr) => {
                ConstantItem(Constant{
                    type_: ty.clean(cx),
                    expr: expr.span.to_src(cx),
                })
            }
            hir::ImplItemKind::Method(ref sig, _) => {
                MethodItem(sig.clean(cx))
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
            def_id: cx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, cx.map.local_def_id(self.id)),
            inner: inner
        }
    }
}

impl<'tcx> Clean<Item> for ty::Method<'tcx> {
    fn clean(&self, cx: &DocContext) -> Item {
        let (self_, sig) = match self.explicit_self {
            ty::StaticExplicitSelfCategory => (hir::SelfStatic.clean(cx),
                                               self.fty.sig.clone()),
            s => {
                let sig = ty::Binder(ty::FnSig {
                    inputs: self.fty.sig.0.inputs[1..].to_vec(),
                    ..self.fty.sig.0.clone()
                });
                let s = match s {
                    ty::ByValueExplicitSelfCategory => SelfValue,
                    ty::ByReferenceExplicitSelfCategory(..) => {
                        match self.fty.sig.0.inputs[0].sty {
                            ty::TyRef(r, mt) => {
                                SelfBorrowed(r.clean(cx), mt.mutbl.clean(cx))
                            }
                            _ => unreachable!(),
                        }
                    }
                    ty::ByBoxExplicitSelfCategory => {
                        SelfExplicit(self.fty.sig.0.inputs[0].clean(cx))
                    }
                    ty::StaticExplicitSelfCategory => unreachable!(),
                };
                (s, sig)
            }
        };

        let generics = (&self.generics, &self.predicates,
                        subst::FnSpace).clean(cx);
        let decl = (self.def_id, &sig).clean(cx);
        let provided = match self.container {
            ty::ImplContainer(..) => false,
            ty::TraitContainer(did) => {
                cx.tcx().provided_trait_methods(did).iter().any(|m| {
                    m.def_id == self.def_id
                })
            }
        };
        let inner = if provided {
            MethodItem(Method {
                unsafety: self.fty.unsafety,
                generics: generics,
                self_: self_,
                decl: decl,
                abi: self.fty.abi,

                // trait methods canot (currently, at least) be const
                constness: hir::Constness::NotConst,
            })
        } else {
            TyMethodItem(TyMethod {
                unsafety: self.fty.unsafety,
                generics: generics,
                self_: self_,
                decl: decl,
                abi: self.fty.abi,
            })
        };

        Item {
            name: Some(self.name.clean(cx)),
            visibility: Some(hir::Inherited),
            stability: get_stability(cx, self.def_id),
            def_id: self.def_id,
            attrs: inline::load_attrs(cx, cx.tcx(), self.def_id),
            source: Span::empty(),
            inner: inner,
        }
    }
}

impl<'tcx> Clean<Item> for ty::ImplOrTraitItem<'tcx> {
    fn clean(&self, cx: &DocContext) -> Item {
        match *self {
            ty::ConstTraitItem(ref cti) => cti.clean(cx),
            ty::MethodTraitItem(ref mti) => mti.clean(cx),
            ty::TypeTraitItem(ref tti) => tti.clean(cx),
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
/// type out of the AST/ty::ctxt given one of these, if more information is needed. Most importantly
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
    /// aka TyBot
    Bottom,
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

    // for<'a> Foo(&'a)
    PolyTraitRef(Vec<TyParamBound>),
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Copy, Debug)]
pub enum PrimitiveType {
    Isize, I8, I16, I32, I64,
    Usize, U8, U16, U32, U64,
    F32, F64,
    Char,
    Bool,
    Str,
    Slice,
    Array,
    PrimitiveTuple,
    PrimitiveRawPointer,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Copy, Debug)]
pub enum TypeKind {
    TypeEnum,
    TypeFunction,
    TypeModule,
    TypeConst,
    TypeStatic,
    TypeStruct,
    TypeTrait,
    TypeVariant,
    TypeTypedef,
}

impl Type {
    pub fn primitive_type(&self) -> Option<PrimitiveType> {
        match *self {
            Primitive(p) | BorrowedRef { type_: box Primitive(p), ..} => Some(p),
            Vector(..) | BorrowedRef{ type_: box Vector(..), ..  } => Some(Slice),
            FixedVector(..) | BorrowedRef { type_: box FixedVector(..), .. } => {
                Some(Array)
            }
            Tuple(..) => Some(PrimitiveTuple),
            RawPointer(..) => Some(PrimitiveRawPointer),
            _ => None,
        }
    }
}

impl PrimitiveType {
    fn from_str(s: &str) -> Option<PrimitiveType> {
        match s {
            "isize" => Some(Isize),
            "i8" => Some(I8),
            "i16" => Some(I16),
            "i32" => Some(I32),
            "i64" => Some(I64),
            "usize" => Some(Usize),
            "u8" => Some(U8),
            "u16" => Some(U16),
            "u32" => Some(U32),
            "u64" => Some(U64),
            "bool" => Some(Bool),
            "char" => Some(Char),
            "str" => Some(Str),
            "f32" => Some(F32),
            "f64" => Some(F64),
            "array" => Some(Array),
            "slice" => Some(Slice),
            "tuple" => Some(PrimitiveTuple),
            "pointer" => Some(PrimitiveRawPointer),
            _ => None,
        }
    }

    fn find(attrs: &[Attribute]) -> Option<PrimitiveType> {
        for attr in attrs {
            let list = match *attr {
                List(ref k, ref l) if *k == "doc" => l,
                _ => continue,
            };
            for sub_attr in list {
                let value = match *sub_attr {
                    NameValue(ref k, ref v)
                        if *k == "primitive" => v,
                    _ => continue,
                };
                match PrimitiveType::from_str(value) {
                    Some(p) => return Some(p),
                    None => {}
                }
            }
        }
        return None
    }

    pub fn to_string(&self) -> &'static str {
        match *self {
            Isize => "isize",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            Usize => "usize",
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            F32 => "f32",
            F64 => "f64",
            Str => "str",
            Bool => "bool",
            Char => "char",
            Array => "array",
            Slice => "slice",
            PrimitiveTuple => "tuple",
            PrimitiveRawPointer => "pointer",
        }
    }

    pub fn to_url_str(&self) -> &'static str {
        self.to_string()
    }

    /// Creates a rustdoc-specific node id for primitive types.
    ///
    /// These node ids are generally never used by the AST itself.
    pub fn to_def_index(&self) -> DefIndex {
        let x = u32::MAX - 1 - (*self as u32);
        DefIndex::new(x as usize)
    }
}

impl Clean<Type> for hir::Ty {
    fn clean(&self, cx: &DocContext) -> Type {
        use rustc_front::hir::*;
        match self.node {
            TyPtr(ref m) => RawPointer(m.mutbl.clean(cx), box m.ty.clean(cx)),
            TyRptr(ref l, ref m) =>
                BorrowedRef {lifetime: l.clean(cx), mutability: m.mutbl.clean(cx),
                             type_: box m.ty.clean(cx)},
            TyVec(ref ty) => Vector(box ty.clean(cx)),
            TyFixedLengthVec(ref ty, ref e) => FixedVector(box ty.clean(cx),
                                                           e.span.to_src(cx)),
            TyTup(ref tys) => Tuple(tys.clean(cx)),
            TyPath(None, ref p) => {
                resolve_type(cx, p.clean(cx), self.id)
            }
            TyPath(Some(ref qself), ref p) => {
                let mut trait_path = p.clone();
                trait_path.segments.pop();
                Type::QPath {
                    name: p.segments.last().unwrap().identifier.name.clean(cx),
                    self_type: box qself.ty.clean(cx),
                    trait_: box resolve_type(cx, trait_path.clean(cx), self.id)
                }
            }
            TyObjectSum(ref lhs, ref bounds) => {
                let lhs_ty = lhs.clean(cx);
                match lhs_ty {
                    ResolvedPath { path, typarams: None, did, is_generic } => {
                        ResolvedPath {
                            path: path,
                            typarams: Some(bounds.clean(cx)),
                            did: did,
                            is_generic: is_generic,
                        }
                    }
                    _ => {
                        lhs_ty // shouldn't happen
                    }
                }
            }
            TyBareFn(ref barefn) => BareFunction(box barefn.clean(cx)),
            TyPolyTraitRef(ref bounds) => {
                PolyTraitRef(bounds.clean(cx))
            },
            TyInfer => {
                Infer
            },
            TyTypeof(..) => {
                panic!("Unimplemented type {:?}", self.node)
            },
        }
    }
}

impl<'tcx> Clean<Type> for ty::Ty<'tcx> {
    fn clean(&self, cx: &DocContext) -> Type {
        match self.sty {
            ty::TyBool => Primitive(Bool),
            ty::TyChar => Primitive(Char),
            ty::TyInt(ast::TyIs) => Primitive(Isize),
            ty::TyInt(ast::TyI8) => Primitive(I8),
            ty::TyInt(ast::TyI16) => Primitive(I16),
            ty::TyInt(ast::TyI32) => Primitive(I32),
            ty::TyInt(ast::TyI64) => Primitive(I64),
            ty::TyUint(ast::TyUs) => Primitive(Usize),
            ty::TyUint(ast::TyU8) => Primitive(U8),
            ty::TyUint(ast::TyU16) => Primitive(U16),
            ty::TyUint(ast::TyU32) => Primitive(U32),
            ty::TyUint(ast::TyU64) => Primitive(U64),
            ty::TyFloat(ast::TyF32) => Primitive(F32),
            ty::TyFloat(ast::TyF64) => Primitive(F64),
            ty::TyStr => Primitive(Str),
            ty::TyBox(t) => {
                let box_did = cx.tcx_opt().and_then(|tcx| {
                    tcx.lang_items.owned_box()
                });
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
            ty::TyBareFn(_, ref fty) => BareFunction(box BareFunctionDecl {
                unsafety: fty.unsafety,
                generics: Generics {
                    lifetimes: Vec::new(),
                    type_params: Vec::new(),
                    where_predicates: Vec::new()
                },
                decl: (cx.map.local_def_id(0), &fty.sig).clean(cx),
                abi: fty.abi.to_string(),
            }),
            ty::TyStruct(def, substs) |
            ty::TyEnum(def, substs) => {
                let did = def.did;
                let fqn = cx.tcx().sess.cstore.item_path(did);
                let fqn: Vec<_> = fqn.into_iter().map(|i| i.to_string()).collect();
                let kind = match self.sty {
                    ty::TyStruct(..) => TypeStruct,
                    _ => TypeEnum,
                };
                let path = external_path(cx, &fqn.last().unwrap().to_string(),
                                         None, vec![], substs);
                cx.external_paths.borrow_mut().as_mut().unwrap().insert(did, (fqn, kind));
                ResolvedPath {
                    path: path,
                    typarams: None,
                    did: did,
                    is_generic: false,
                }
            }
            ty::TyTrait(box ty::TraitTy { ref principal, ref bounds }) => {
                let did = principal.def_id();
                let fqn = cx.tcx().sess.cstore.item_path(did);
                let fqn: Vec<_> = fqn.into_iter().map(|i| i.to_string()).collect();
                let (typarams, bindings) = bounds.clean(cx);
                let path = external_path(cx, &fqn.last().unwrap().to_string(),
                                         Some(did), bindings, principal.substs());
                cx.external_paths.borrow_mut().as_mut().unwrap().insert(did, (fqn, TypeTrait));
                ResolvedPath {
                    path: path,
                    typarams: Some(typarams),
                    did: did,
                    is_generic: false,
                }
            }
            ty::TyTuple(ref t) => Tuple(t.clean(cx)),

            ty::TyProjection(ref data) => data.clean(cx),

            ty::TyParam(ref p) => Generic(p.name.to_string()),

            ty::TyClosure(..) => Tuple(vec![]), // FIXME(pcwalton)

            ty::TyInfer(..) => panic!("TyInfer"),
            ty::TyError => panic!("TyError"),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum StructField {
    HiddenStructField, // inserted later by strip passes
    TypedStructField(Type),
}

impl Clean<Item> for hir::StructField {
    fn clean(&self, cx: &DocContext) -> Item {
        let (name, vis) = match self.node.kind {
            hir::NamedField(id, vis) => (Some(id), vis),
            hir::UnnamedField(vis) => (None, vis)
        };
        Item {
            name: name.clean(cx),
            attrs: self.node.attrs.clean(cx),
            source: self.span.clean(cx),
            visibility: Some(vis),
            stability: get_stability(cx, cx.map.local_def_id(self.node.id)),
            def_id: cx.map.local_def_id(self.node.id),
            inner: StructFieldItem(TypedStructField(self.node.ty.clean(cx))),
        }
    }
}

impl<'tcx> Clean<Item> for ty::FieldDefData<'tcx, 'static> {
    fn clean(&self, cx: &DocContext) -> Item {
        use syntax::parse::token::special_idents::unnamed_field;
        // FIXME: possible O(n^2)-ness! Not my fault.
        let attr_map =
            cx.tcx().sess.cstore.crate_struct_field_attrs(self.did.krate);

        let (name, attrs) = if self.name == unnamed_field.name {
            (None, None)
        } else {
            (Some(self.name), Some(attr_map.get(&self.did).unwrap()))
        };

        Item {
            name: name.clean(cx),
            attrs: attrs.unwrap_or(&Vec::new()).clean(cx),
            source: Span::empty(),
            visibility: Some(self.vis),
            stability: get_stability(cx, self.did),
            def_id: self.did,
            inner: StructFieldItem(TypedStructField(self.unsubst_ty().clean(cx))),
        }
    }
}

pub type Visibility = hir::Visibility;

impl Clean<Option<Visibility>> for hir::Visibility {
    fn clean(&self, _: &DocContext) -> Option<Visibility> {
        Some(*self)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Struct {
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
            def_id: cx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            inner: StructItem(Struct {
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

impl Clean<VariantStruct> for ::rustc_front::hir::VariantData {
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
            def_id: cx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
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
            def_id: cx.map.local_def_id(self.def.id()),
            inner: VariantItem(Variant {
                kind: struct_def_to_variant_kind(&self.def, cx),
            }),
        }
    }
}

impl<'tcx> Clean<Item> for ty::VariantDefData<'tcx, 'static> {
    fn clean(&self, cx: &DocContext) -> Item {
        // use syntax::parse::token::special_idents::unnamed_field;
        let kind = match self.kind() {
            ty::VariantKind::Unit => CLikeVariant,
            ty::VariantKind::Tuple => {
                TupleVariant(
                    self.fields.iter().map(|f| f.unsubst_ty().clean(cx)).collect()
                )
            }
            ty::VariantKind::Struct => {
                StructVariant(VariantStruct {
                    struct_type: doctree::Plain,
                    fields_stripped: false,
                    fields: self.fields.iter().map(|field| {
                        Item {
                            source: Span::empty(),
                            name: Some(field.name.clean(cx)),
                            attrs: Vec::new(),
                            visibility: Some(hir::Public),
                            // FIXME: this is not accurate, we need an id for
                            //        the specific field but we're using the id
                            //        for the whole variant. Thus we read the
                            //        stability from the whole variant as well.
                            //        Struct variants are experimental and need
                            //        more infrastructure work before we can get
                            //        at the needed information here.
                            def_id: self.did,
                            stability: get_stability(cx, self.did),
                            inner: StructFieldItem(
                                TypedStructField(field.unsubst_ty().clean(cx))
                            )
                        }
                    }).collect()
                })
            }
        };
        Item {
            name: Some(self.name.clean(cx)),
            attrs: inline::load_attrs(cx, cx.tcx(), self.did),
            source: Span::empty(),
            visibility: Some(hir::Public),
            def_id: self.did,
            inner: VariantItem(Variant { kind: kind }),
            stability: get_stability(cx, self.did),
        }
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum VariantKind {
    CLikeVariant,
    TupleVariant(Vec<Type>),
    StructVariant(VariantStruct),
}

fn struct_def_to_variant_kind(struct_def: &hir::VariantData, cx: &DocContext) -> VariantKind {
    if struct_def.is_struct() {
        StructVariant(struct_def.clean(cx))
    } else if struct_def.is_unit() {
        CLikeVariant
    } else {
        TupleVariant(struct_def.fields().iter().map(|x| x.node.ty.clean(cx)).collect())
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

impl Clean<Span> for syntax::codemap::Span {
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
    pub segments: Vec<PathSegment>,
}

impl Path {
    pub fn singleton(name: String) -> Path {
        Path {
            global: false,
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
}

impl Clean<Path> for hir::Path {
    fn clean(&self, cx: &DocContext) -> Path {
        Path {
            global: self.global,
            segments: self.segments.clean(cx),
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
            name: self.identifier.name.clean(cx),
            params: self.parameters.clean(cx)
        }
    }
}

fn path_to_string(p: &hir::Path) -> String {
    let mut s = String::new();
    let mut first = true;
    for i in p.segments.iter().map(|x| x.identifier.name.as_str()) {
        if !first || p.global {
            s.push_str("::");
        } else {
            first = false;
        }
        s.push_str(&i);
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
            def_id: cx.map.local_def_id(self.id.clone()),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
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
    pub abi: String,
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
            decl: self.decl.clean(cx),
            abi: self.abi.to_string(),
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
            def_id: cx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            inner: StaticItem(Static {
                type_: self.type_.clean(cx),
                mutability: self.mutability.clean(cx),
                expr: self.expr.span.to_src(cx),
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
            def_id: cx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            inner: ConstantItem(Constant {
                type_: self.type_.clean(cx),
                expr: self.expr.span.to_src(cx),
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
    pub trait_: Option<Type>,
    pub for_: Type,
    pub items: Vec<Item>,
    pub derived: bool,
    pub polarity: Option<ImplPolarity>,
}

fn detect_derived<M: AttrMetaMethods>(attrs: &[M]) -> bool {
    attr::contains_name(attrs, "automatically_derived")
}

impl Clean<Vec<Item>> for doctree::Impl {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {
        let mut ret = Vec::new();
        let trait_ = self.trait_.clean(cx);
        let items = self.items.clean(cx);

        // If this impl block is an implementation of the Deref trait, then we
        // need to try inlining the target's inherent impl blocks as well.
        if let Some(ResolvedPath { did, .. }) = trait_ {
            if Some(did) == cx.deref_trait_did.get() {
                build_deref_target_impls(cx, &items, &mut ret);
            }
        }

        ret.push(Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: self.stab.clean(cx),
            inner: ImplItem(Impl {
                unsafety: self.unsafety,
                generics: self.generics.clean(cx),
                trait_: trait_,
                for_: self.for_.clean(cx),
                items: items,
                derived: detect_derived(&self.attrs),
                polarity: Some(self.polarity.clean(cx)),
            }),
        });
        return ret;
    }
}

fn build_deref_target_impls(cx: &DocContext,
                            items: &[Item],
                            ret: &mut Vec<Item>) {
    let tcx = match cx.tcx_opt() {
        Some(t) => t,
        None => return,
    };

    for item in items {
        let target = match item.inner {
            TypedefItem(ref t, true) => &t.type_,
            _ => continue,
        };
        let primitive = match *target {
            ResolvedPath { did, .. } if did.is_local() => continue,
            ResolvedPath { did, .. } => {
                ret.extend(inline::build_impls(cx, tcx, did));
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
            Usize => tcx.lang_items.usize_impl(),
            U8 => tcx.lang_items.u8_impl(),
            U16 => tcx.lang_items.u16_impl(),
            U32 => tcx.lang_items.u32_impl(),
            U64 => tcx.lang_items.u64_impl(),
            F32 => tcx.lang_items.f32_impl(),
            F64 => tcx.lang_items.f64_impl(),
            Char => tcx.lang_items.char_impl(),
            Bool => None,
            Str => tcx.lang_items.str_impl(),
            Slice => tcx.lang_items.slice_impl(),
            Array => tcx.lang_items.slice_impl(),
            PrimitiveTuple => None,
            PrimitiveRawPointer => tcx.lang_items.const_ptr_impl(),
        };
        if let Some(did) = did {
            if !did.is_local() {
                inline::build_impl(cx, tcx, did, ret);
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
            def_id: cx.map.local_def_id(self.id),
            visibility: Some(hir::Public),
            stability: None,
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
            def_id: cx.map.local_def_id(0),
            visibility: self.vis.clean(cx),
            stability: None,
            inner: ExternCrateItem(self.name.clean(cx), self.path.clone())
        }
    }
}

impl Clean<Vec<Item>> for doctree::Import {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {
        // We consider inlining the documentation of `pub use` statements, but we
        // forcefully don't inline if this is not public or if the
        // #[doc(no_inline)] attribute is present.
        let denied = self.vis != hir::Public || self.attrs.iter().any(|a| {
            &a.name()[..] == "doc" && match a.meta_item_list() {
                Some(l) => attr::contains_name(l, "no_inline"),
                None => false,
            }
        });
        let (mut ret, inner) = match self.node {
            hir::ViewPathGlob(ref p) => {
                (vec![], GlobImport(resolve_use_source(cx, p.clean(cx), self.id)))
            }
            hir::ViewPathList(ref p, ref list) => {
                // Attempt to inline all reexported items, but be sure
                // to keep any non-inlineable reexports so they can be
                // listed in the documentation.
                let mut ret = vec![];
                let remaining = if !denied {
                    let mut remaining = vec![];
                    for path in list {
                        match inline::try_inline(cx, path.node.id(), path.node.rename()) {
                            Some(items) => {
                                ret.extend(items);
                            }
                            None => {
                                remaining.push(path.clean(cx));
                            }
                        }
                    }
                    remaining
                } else {
                    list.clean(cx)
                };
                if remaining.is_empty() {
                    return ret;
                }
                (ret, ImportList(resolve_use_source(cx, p.clean(cx), self.id),
                                 remaining))
            }
            hir::ViewPathSimple(name, ref p) => {
                if !denied {
                    match inline::try_inline(cx, self.id, Some(name)) {
                        Some(items) => return items,
                        None => {}
                    }
                }
                (vec![], SimpleImport(name.clean(cx),
                                      resolve_use_source(cx, p.clean(cx), self.id)))
            }
        };
        ret.push(Item {
            name: None,
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            def_id: cx.map.local_def_id(0),
            visibility: self.vis.clean(cx),
            stability: None,
            inner: ImportItem(inner)
        });
        ret
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum Import {
    // use source as str;
    SimpleImport(String, ImportSource),
    // use source::*;
    GlobImport(ImportSource),
    // use source::{a, b, c};
    ImportList(ImportSource, Vec<ViewListIdent>),
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ImportSource {
    pub path: Path,
    pub did: Option<DefId>,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ViewListIdent {
    pub name: String,
    pub rename: Option<String>,
    pub source: Option<DefId>,
}

impl Clean<ViewListIdent> for hir::PathListItem {
    fn clean(&self, cx: &DocContext) -> ViewListIdent {
        match self.node {
            hir::PathListIdent { id, name, rename } => ViewListIdent {
                name: name.clean(cx),
                rename: rename.map(|r| r.clean(cx)),
                source: resolve_def(cx, id)
            },
            hir::PathListMod { id, rename } => ViewListIdent {
                name: "self".to_string(),
                rename: rename.map(|r| r.clean(cx)),
                source: resolve_def(cx, id)
            }
        }
    }
}

impl Clean<Vec<Item>> for hir::ForeignMod {
    fn clean(&self, cx: &DocContext) -> Vec<Item> {
        let mut items = self.items.clean(cx);
        for item in &mut items {
            match item.inner {
                ForeignFunctionItem(ref mut f) => f.abi = self.abi,
                _ => {}
            }
        }
        items
    }
}

impl Clean<Item> for hir::ForeignItem {
    fn clean(&self, cx: &DocContext) -> Item {
        let inner = match self.node {
            hir::ForeignItemFn(ref decl, ref generics) => {
                ForeignFunctionItem(Function {
                    decl: decl.clean(cx),
                    generics: generics.clean(cx),
                    unsafety: hir::Unsafety::Unsafe,
                    abi: abi::Rust,
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
            def_id: cx.map.local_def_id(self.id),
            visibility: self.vis.clean(cx),
            stability: get_stability(cx, cx.map.local_def_id(self.id)),
            inner: inner,
        }
    }
}

// Utilities

trait ToSource {
    fn to_src(&self, cx: &DocContext) -> String;
}

impl ToSource for syntax::codemap::Span {
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

fn lit_to_string(lit: &ast::Lit) -> String {
    match lit.node {
        ast::LitStr(ref st, _) => st.to_string(),
        ast::LitByteStr(ref data) => format!("{:?}", data),
        ast::LitByte(b) => {
            let mut res = String::from("b'");
            for c in (b as char).escape_default() {
                res.push(c);
            }
            res.push('\'');
            res
        },
        ast::LitChar(c) => format!("'{}'", c),
        ast::LitInt(i, _t) => i.to_string(),
        ast::LitFloat(ref f, _t) => f.to_string(),
        ast::LitFloatUnsuffixed(ref f) => f.to_string(),
        ast::LitBool(b) => b.to_string(),
    }
}

fn name_from_pat(p: &hir::Pat) -> String {
    use rustc_front::hir::*;
    debug!("Trying to get a name from pattern: {:?}", p);

    match p.node {
        PatWild => "_".to_string(),
        PatIdent(_, ref p, _) => p.node.to_string(),
        PatEnum(ref p, _) => path_to_string(p),
        PatQPath(..) => panic!("tried to get argument name from PatQPath, \
                                which is not allowed in function arguments"),
        PatStruct(ref name, ref fields, etc) => {
            format!("{} {{ {}{} }}", path_to_string(name),
                fields.iter().map(|&Spanned { node: ref fp, .. }|
                                  format!("{}: {}", fp.name, name_from_pat(&*fp.pat)))
                             .collect::<Vec<String>>().join(", "),
                if etc { ", ..." } else { "" }
            )
        },
        PatTup(ref elts) => format!("({})", elts.iter().map(|p| name_from_pat(&**p))
                                            .collect::<Vec<String>>().join(", ")),
        PatBox(ref p) => name_from_pat(&**p),
        PatRegion(ref p, _) => name_from_pat(&**p),
        PatLit(..) => {
            warn!("tried to get argument name from PatLit, \
                  which is silly in function arguments");
            "()".to_string()
        },
        PatRange(..) => panic!("tried to get argument name from PatRange, \
                              which is not allowed in function arguments"),
        PatVec(ref begin, ref mid, ref end) => {
            let begin = begin.iter().map(|p| name_from_pat(&**p));
            let mid = mid.as_ref().map(|p| format!("..{}", name_from_pat(&**p))).into_iter();
            let end = end.iter().map(|p| name_from_pat(&**p));
            format!("[{}]", begin.chain(mid).chain(end).collect::<Vec<_>>().join(", "))
        },
    }
}

/// Given a Type, resolve it using the def_map
fn resolve_type(cx: &DocContext,
                path: Path,
                id: ast::NodeId) -> Type {
    debug!("resolve_type({:?},{:?})", path, id);
    let tcx = match cx.tcx_opt() {
        Some(tcx) => tcx,
        // If we're extracting tests, this return value's accuracy is not
        // important, all we want is a string representation to help people
        // figure out what doctests are failing.
        None => {
            let did = DefId::local(DefIndex::from_u32(0));
            return ResolvedPath {
                path: path,
                typarams: None,
                did: did,
                is_generic: false
            };
        }
    };
    let def = match tcx.def_map.borrow().get(&id) {
        Some(k) => k.full_def(),
        None => panic!("unresolved id not in defmap")
    };

    debug!("resolve_type: def={:?}", def);

    let is_generic = match def {
        def::DefPrimTy(p) => match p {
            hir::TyStr => return Primitive(Str),
            hir::TyBool => return Primitive(Bool),
            hir::TyChar => return Primitive(Char),
            hir::TyInt(ast::TyIs) => return Primitive(Isize),
            hir::TyInt(ast::TyI8) => return Primitive(I8),
            hir::TyInt(ast::TyI16) => return Primitive(I16),
            hir::TyInt(ast::TyI32) => return Primitive(I32),
            hir::TyInt(ast::TyI64) => return Primitive(I64),
            hir::TyUint(ast::TyUs) => return Primitive(Usize),
            hir::TyUint(ast::TyU8) => return Primitive(U8),
            hir::TyUint(ast::TyU16) => return Primitive(U16),
            hir::TyUint(ast::TyU32) => return Primitive(U32),
            hir::TyUint(ast::TyU64) => return Primitive(U64),
            hir::TyFloat(ast::TyF32) => return Primitive(F32),
            hir::TyFloat(ast::TyF64) => return Primitive(F64),
        },
        def::DefSelfTy(..) if path.segments.len() == 1 => {
            return Generic(special_idents::type_self.name.to_string());
        }
        def::DefSelfTy(..) | def::DefTyParam(..) => true,
        _ => false,
    };
    let did = register_def(&*cx, def);
    ResolvedPath { path: path, typarams: None, did: did, is_generic: is_generic }
}

fn register_def(cx: &DocContext, def: def::Def) -> DefId {
    debug!("register_def({:?})", def);

    let (did, kind) = match def {
        def::DefFn(i, _) => (i, TypeFunction),
        def::DefTy(i, false) => (i, TypeTypedef),
        def::DefTy(i, true) => (i, TypeEnum),
        def::DefTrait(i) => (i, TypeTrait),
        def::DefStruct(i) => (i, TypeStruct),
        def::DefMod(i) => (i, TypeModule),
        def::DefStatic(i, _) => (i, TypeStatic),
        def::DefVariant(i, _, _) => (i, TypeEnum),
        def::DefSelfTy(Some(def_id), _) => (def_id, TypeTrait),
        def::DefSelfTy(_, Some((impl_id, _))) => return cx.map.local_def_id(impl_id),
        _ => return def.def_id()
    };
    if did.is_local() { return did }
    let tcx = match cx.tcx_opt() {
        Some(tcx) => tcx,
        None => return did
    };
    inline::record_extern_fqn(cx, did, kind);
    if let TypeTrait = kind {
        let t = inline::build_external_trait(cx, tcx, did);
        cx.external_traits.borrow_mut().as_mut().unwrap().insert(did, t);
    }
    return did;
}

fn resolve_use_source(cx: &DocContext, path: Path, id: ast::NodeId) -> ImportSource {
    ImportSource {
        path: path,
        did: resolve_def(cx, id),
    }
}

fn resolve_def(cx: &DocContext, id: ast::NodeId) -> Option<DefId> {
    cx.tcx_opt().and_then(|tcx| {
        tcx.def_map.borrow().get(&id).map(|d| register_def(cx, d.full_def()))
    })
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Macro {
    pub source: String,
    pub imported_from: Option<String>,
}

impl Clean<Item> for doctree::Macro {
    fn clean(&self, cx: &DocContext) -> Item {
        let name = format!("{}!", self.name.clean(cx));
        Item {
            name: Some(name.clone()),
            attrs: self.attrs.clean(cx),
            source: self.whence.clean(cx),
            visibility: hir::Public.clean(cx),
            stability: self.stab.clean(cx),
            def_id: cx.map.local_def_id(self.id),
            inner: MacroItem(Macro {
                source: format!("macro_rules! {} {{\n{}}}",
                    name.trim_right_matches('!'), self.matchers.iter().map(|span|
                        format!("    {} => {{ ... }};\n", span.to_src(cx))).collect::<String>()),
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
    pub reason: String,
    pub issue: Option<u32>
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
            deprecated_since: match self.depr {
                Some(attr::Deprecation {ref since, ..}) => since.to_string(),
                _=> "".to_string(),
            },
            reason: {
                if let Some(ref depr) = self.depr {
                    depr.reason.to_string()
                } else if let attr::Unstable {reason: Some(ref reason), ..} = self.level {
                    reason.to_string()
                } else {
                    "".to_string()
                }
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

impl<'tcx> Clean<Item> for ty::AssociatedConst<'tcx> {
    fn clean(&self, cx: &DocContext) -> Item {
        Item {
            source: DUMMY_SP.clean(cx),
            name: Some(self.name.clean(cx)),
            attrs: Vec::new(),
            inner: AssociatedConstItem(self.ty.clean(cx), None),
            visibility: None,
            def_id: self.def_id,
            stability: None,
        }
    }
}

impl<'tcx> Clean<Item> for ty::AssociatedType<'tcx> {
    fn clean(&self, cx: &DocContext) -> Item {
        let my_name = self.name.clean(cx);

        let mut bounds = if let ty::TraitContainer(did) = self.container {
            // When loading a cross-crate associated type, the bounds for this type
            // are actually located on the trait/impl itself, so we need to load
            // all of the generics from there and then look for bounds that are
            // applied to this associated type in question.
            let def = cx.tcx().lookup_trait_def(did);
            let predicates = cx.tcx().lookup_predicates(did);
            let generics = (&def.generics, &predicates, subst::TypeSpace).clean(cx);
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

        Item {
            source: DUMMY_SP.clean(cx),
            name: Some(self.name.clean(cx)),
            attrs: inline::load_attrs(cx, cx.tcx(), self.def_id),
            inner: AssociatedTypeItem(bounds, self.ty.clean(cx)),
            visibility: self.vis.clean(cx),
            def_id: self.def_id,
            stability: stability::lookup(cx.tcx(), self.def_id).clean(cx),
        }
    }
}

impl<'a> Clean<Typedef> for (ty::TypeScheme<'a>, ty::GenericPredicates<'a>,
                             ParamSpace) {
    fn clean(&self, cx: &DocContext) -> Typedef {
        let (ref ty_scheme, ref predicates, ps) = *self;
        Typedef {
            type_: ty_scheme.ty.clean(cx),
            generics: (&ty_scheme.generics, predicates, ps).clean(cx)
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
    let fqn = cx.tcx().sess.cstore.item_path(did);
    let fqn: Vec<String> = fqn.into_iter().map(|i| {
        i.to_string()
    }).collect();
    cx.external_paths.borrow_mut().as_mut().unwrap().insert(did, (fqn, TypeStruct));
    ResolvedPath {
        typarams: None,
        did: did,
        path: Path {
            global: false,
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
