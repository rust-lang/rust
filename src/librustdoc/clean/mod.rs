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

use syntax;
use syntax::ast;
use syntax::ast_util;
use syntax::attr;
use syntax::attr::{AttributeMethods, AttrMetaMethods};
use syntax::codemap::Pos;
use syntax::parse::token::InternedString;
use syntax::parse::token;

use rustc::back::link;
use rustc::driver::driver;
use rustc::metadata::cstore;
use rustc::metadata::csearch;
use rustc::metadata::decoder;
use rustc::middle::def;
use rustc::middle::subst;
use rustc::middle::subst::VecPerParamSpace;
use rustc::middle::ty;
use rustc::middle::stability;

use std::rc::Rc;
use std::u32;
use std::gc::{Gc, GC};

use core;
use doctree;
use visit_ast;

/// A stable identifier to the particular version of JSON output.
/// Increment this when the `Crate` and related structures change.
pub static SCHEMA_VERSION: &'static str = "0.8.3";

mod inline;

// load the current DocContext from TLD
fn get_cx() -> Gc<core::DocContext> {
    *super::ctxtkey.get().unwrap()
}

// extract the stability index for a node from TLD, if possible
fn get_stability(def_id: ast::DefId) -> Option<Stability> {
    get_cx().tcx_opt().and_then(|tcx| stability::lookup(tcx, def_id))
            .map(|stab| stab.clean())
}

pub trait Clean<T> {
    fn clean(&self) -> T;
}

impl<T: Clean<U>, U> Clean<Vec<U>> for Vec<T> {
    fn clean(&self) -> Vec<U> {
        self.iter().map(|x| x.clean()).collect()
    }
}

impl<T: Clean<U>, U> Clean<VecPerParamSpace<U>> for VecPerParamSpace<T> {
    fn clean(&self) -> VecPerParamSpace<U> {
        self.map(|x| x.clean())
    }
}

impl<T: 'static + Clean<U>, U> Clean<U> for Gc<T> {
    fn clean(&self) -> U {
        (**self).clean()
    }
}

impl<T: Clean<U>, U> Clean<U> for Rc<T> {
    fn clean(&self) -> U {
        (**self).clean()
    }
}

impl<T: Clean<U>, U> Clean<Option<U>> for Option<T> {
    fn clean(&self) -> Option<U> {
        match self {
            &None => None,
            &Some(ref v) => Some(v.clean())
        }
    }
}

impl<T: Clean<U>, U> Clean<Vec<U>> for syntax::owned_slice::OwnedSlice<T> {
    fn clean(&self) -> Vec<U> {
        self.iter().map(|x| x.clean()).collect()
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Crate {
    pub name: String,
    pub module: Option<Item>,
    pub externs: Vec<(ast::CrateNum, ExternalCrate)>,
    pub primitives: Vec<Primitive>,
}

impl<'a> Clean<Crate> for visit_ast::RustdocVisitor<'a> {
    fn clean(&self) -> Crate {
        let cx = get_cx();

        let mut externs = Vec::new();
        cx.sess().cstore.iter_crate_data(|n, meta| {
            externs.push((n, meta.clean()));
        });
        externs.sort_by(|&(a, _), &(b, _)| a.cmp(&b));

        // Figure out the name of this crate
        let input = driver::FileInput(cx.src.clone());
        let t_outputs = driver::build_output_filenames(&input,
                                                       &None,
                                                       &None,
                                                       self.attrs.as_slice(),
                                                       cx.sess());
        let id = link::find_crate_id(self.attrs.as_slice(),
                                     t_outputs.out_filestem.as_slice());

        // Clean the crate, translating the entire libsyntax AST to one that is
        // understood by rustdoc.
        let mut module = self.module.clean();

        // Collect all inner modules which are tagged as implementations of
        // primitives.
        //
        // Note that this loop only searches the top-level items of the crate,
        // and this is intentional. If we were to search the entire crate for an
        // item tagged with `#[doc(primitive)]` then we we would also have to
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
            for child in m.items.mut_iter() {
                let inner = match child.inner {
                    ModuleItem(ref mut m) => m,
                    _ => continue,
                };
                let prim = match Primitive::find(child.attrs.as_slice()) {
                    Some(prim) => prim,
                    None => continue,
                };
                primitives.push(prim);
                let mut i = Item {
                    source: Span::empty(),
                    name: Some(prim.to_url_str().to_string()),
                    attrs: Vec::new(),
                    visibility: None,
                    stability: None,
                    def_id: ast_util::local_def(prim.to_node_id()),
                    inner: PrimitiveItem(prim),
                };
                // Push one copy to get indexed for the whole crate, and push a
                // another copy in the proper location which will actually get
                // documented. The first copy will also serve as a redirect to
                // the other copy.
                tmp.push(i.clone());
                i.visibility = Some(ast::Public);
                i.attrs = child.attrs.clone();
                inner.items.push(i);

            }
            m.items.extend(tmp.move_iter());
        }

        Crate {
            name: id.name.to_string(),
            module: Some(module),
            externs: externs,
            primitives: primitives,
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ExternalCrate {
    pub name: String,
    pub attrs: Vec<Attribute>,
    pub primitives: Vec<Primitive>,
}

impl Clean<ExternalCrate> for cstore::crate_metadata {
    fn clean(&self) -> ExternalCrate {
        let mut primitives = Vec::new();
        get_cx().tcx_opt().map(|tcx| {
            csearch::each_top_level_item_of_crate(&tcx.sess.cstore,
                                                  self.cnum,
                                                  |def, _, _| {
                let did = match def {
                    decoder::DlDef(def::DefMod(did)) => did,
                    _ => return
                };
                let attrs = inline::load_attrs(tcx, did);
                Primitive::find(attrs.as_slice()).map(|prim| primitives.push(prim));
            })
        });
        ExternalCrate {
            name: self.name.to_string(),
            attrs: decoder::get_crate_attributes(self.data()).clean(),
            primitives: primitives,
        }
    }
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[deriving(Clone, Encodable, Decodable)]
pub struct Item {
    /// Stringified span
    pub source: Span,
    /// Not everything has a name. E.g., impls
    pub name: Option<String>,
    pub attrs: Vec<Attribute> ,
    pub inner: ItemEnum,
    pub visibility: Option<Visibility>,
    pub def_id: ast::DefId,
    pub stability: Option<Stability>,
}

impl Item {
    /// Finds the `doc` attribute as a List and returns the list of attributes
    /// nested inside.
    pub fn doc_list<'a>(&'a self) -> Option<&'a [Attribute]> {
        for attr in self.attrs.iter() {
            match *attr {
                List(ref x, ref list) if "doc" == x.as_slice() => {
                    return Some(list.as_slice());
                }
                _ => {}
            }
        }
        return None;
    }

    /// Finds the `doc` attribute as a NameValue and returns the corresponding
    /// value found.
    pub fn doc_value<'a>(&'a self) -> Option<&'a str> {
        for attr in self.attrs.iter() {
            match *attr {
                NameValue(ref x, ref v) if "doc" == x.as_slice() => {
                    return Some(v.as_slice());
                }
                _ => {}
            }
        }
        return None;
    }

    pub fn is_hidden_from_doc(&self) -> bool {
        match self.doc_list() {
            Some(ref l) => {
                for innerattr in l.iter() {
                    match *innerattr {
                        Word(ref s) if "hidden" == s.as_slice() => {
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
}

#[deriving(Clone, Encodable, Decodable)]
pub enum ItemEnum {
    StructItem(Struct),
    EnumItem(Enum),
    FunctionItem(Function),
    ModuleItem(Module),
    TypedefItem(Typedef),
    StaticItem(Static),
    TraitItem(Trait),
    ImplItem(Impl),
    /// `use` and `extern crate`
    ViewItemItem(ViewItem),
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
    PrimitiveItem(Primitive),
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Module {
    pub items: Vec<Item>,
    pub is_crate: bool,
}

impl Clean<Item> for doctree::Module {
    fn clean(&self) -> Item {
        let name = if self.name.is_some() {
            self.name.unwrap().clean()
        } else {
            "".to_string()
        };
        let mut foreigns = Vec::new();
        for subforeigns in self.foreigns.clean().move_iter() {
            for foreign in subforeigns.move_iter() {
                foreigns.push(foreign)
            }
        }
        let items: Vec<Vec<Item> > = vec!(
            self.structs.clean().move_iter().collect(),
            self.enums.clean().move_iter().collect(),
            self.fns.clean().move_iter().collect(),
            foreigns,
            self.mods.clean().move_iter().collect(),
            self.typedefs.clean().move_iter().collect(),
            self.statics.clean().move_iter().collect(),
            self.traits.clean().move_iter().collect(),
            self.impls.clean().move_iter().collect(),
            self.view_items.clean().move_iter()
                           .flat_map(|s| s.move_iter()).collect(),
            self.macros.clean().move_iter().collect()
        );

        // determine if we should display the inner contents or
        // the outer `mod` item for the source code.
        let where = {
            let ctxt = super::ctxtkey.get().unwrap();
            let cm = ctxt.sess().codemap();
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
            attrs: self.attrs.clean(),
            source: where.clean(),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            def_id: ast_util::local_def(self.id),
            inner: ModuleItem(Module {
               is_crate: self.is_crate,
               items: items.iter()
                           .flat_map(|x| x.iter().map(|x| (*x).clone()))
                           .collect(),
            })
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum Attribute {
    Word(String),
    List(String, Vec<Attribute> ),
    NameValue(String, String)
}

impl Clean<Attribute> for ast::MetaItem {
    fn clean(&self) -> Attribute {
        match self.node {
            ast::MetaWord(ref s) => Word(s.get().to_string()),
            ast::MetaList(ref s, ref l) => {
                List(s.get().to_string(), l.clean().move_iter().collect())
            }
            ast::MetaNameValue(ref s, ref v) => {
                NameValue(s.get().to_string(), lit_to_str(v))
            }
        }
    }
}

impl Clean<Attribute> for ast::Attribute {
    fn clean(&self) -> Attribute {
        self.desugar_doc().node.value.clean()
    }
}

// This is a rough approximation that gets us what we want.
impl attr::AttrMetaMethods for Attribute {
    fn name(&self) -> InternedString {
        match *self {
            Word(ref n) | List(ref n, _) | NameValue(ref n, _) => {
                token::intern_and_get_ident(n.as_slice())
            }
        }
    }

    fn value_str(&self) -> Option<InternedString> {
        match *self {
            NameValue(_, ref v) => {
                Some(token::intern_and_get_ident(v.as_slice()))
            }
            _ => None,
        }
    }
    fn meta_item_list<'a>(&'a self) -> Option<&'a [Gc<ast::MetaItem>]> { None }
}
impl<'a> attr::AttrMetaMethods for &'a Attribute {
    fn name(&self) -> InternedString { (**self).name() }
    fn value_str(&self) -> Option<InternedString> { (**self).value_str() }
    fn meta_item_list<'a>(&'a self) -> Option<&'a [Gc<ast::MetaItem>]> { None }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct TyParam {
    pub name: String,
    pub did: ast::DefId,
    pub bounds: Vec<TyParamBound>,
    pub default: Option<Type>
}

impl Clean<TyParam> for ast::TyParam {
    fn clean(&self) -> TyParam {
        TyParam {
            name: self.ident.clean(),
            did: ast::DefId { krate: ast::LOCAL_CRATE, node: self.id },
            bounds: self.bounds.clean().move_iter().collect(),
            default: self.default.clean()
        }
    }
}

impl Clean<TyParam> for ty::TypeParameterDef {
    fn clean(&self) -> TyParam {
        get_cx().external_typarams.borrow_mut().get_mut_ref()
                .insert(self.def_id, self.ident.clean());
        TyParam {
            name: self.ident.clean(),
            did: self.def_id,
            bounds: self.bounds.clean(),
            default: self.default.clean()
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum TyParamBound {
    RegionBound,
    TraitBound(Type)
}

impl Clean<TyParamBound> for ast::TyParamBound {
    fn clean(&self) -> TyParamBound {
        match *self {
            ast::StaticRegionTyParamBound => RegionBound,
            ast::OtherRegionTyParamBound(_) => RegionBound,
            ast::UnboxedFnTyParamBound(_) => {
                // FIXME(pcwalton): Wrong.
                RegionBound
            }
            ast::TraitTyParamBound(ref t) => TraitBound(t.clean()),
        }
    }
}

fn external_path(name: &str, substs: &subst::Substs) -> Path {
    let lifetimes = substs.regions().get_vec(subst::TypeSpace)
                    .iter()
                    .filter_map(|v| v.clean())
                    .collect();
    let types = substs.types.get_vec(subst::TypeSpace).clean();
    Path {
        global: false,
        segments: vec![PathSegment {
            name: name.to_string(),
            lifetimes: lifetimes,
            types: types,
        }],
    }
}

impl Clean<TyParamBound> for ty::BuiltinBound {
    fn clean(&self) -> TyParamBound {
        let cx = get_cx();
        let tcx = match cx.maybe_typed {
            core::Typed(ref tcx) => tcx,
            core::NotTyped(_) => return RegionBound,
        };
        let empty = subst::Substs::empty();
        let (did, path) = match *self {
            ty::BoundStatic => return RegionBound,
            ty::BoundSend =>
                (tcx.lang_items.send_trait().unwrap(),
                 external_path("Send", &empty)),
            ty::BoundSized =>
                (tcx.lang_items.sized_trait().unwrap(),
                 external_path("Sized", &empty)),
            ty::BoundCopy =>
                (tcx.lang_items.copy_trait().unwrap(),
                 external_path("Copy", &empty)),
            ty::BoundShare =>
                (tcx.lang_items.share_trait().unwrap(),
                 external_path("Share", &empty)),
        };
        let fqn = csearch::get_item_path(tcx, did);
        let fqn = fqn.move_iter().map(|i| i.to_str()).collect();
        cx.external_paths.borrow_mut().get_mut_ref().insert(did,
                                                            (fqn, TypeTrait));
        TraitBound(ResolvedPath {
            path: path,
            typarams: None,
            did: did,
        })
    }
}

impl Clean<TyParamBound> for ty::TraitRef {
    fn clean(&self) -> TyParamBound {
        let cx = get_cx();
        let tcx = match cx.maybe_typed {
            core::Typed(ref tcx) => tcx,
            core::NotTyped(_) => return RegionBound,
        };
        let fqn = csearch::get_item_path(tcx, self.def_id);
        let fqn = fqn.move_iter().map(|i| i.to_str())
                     .collect::<Vec<String>>();
        let path = external_path(fqn.last().unwrap().as_slice(),
                                 &self.substs);
        cx.external_paths.borrow_mut().get_mut_ref().insert(self.def_id,
                                                            (fqn, TypeTrait));
        TraitBound(ResolvedPath {
            path: path,
            typarams: None,
            did: self.def_id,
        })
    }
}

impl Clean<Vec<TyParamBound>> for ty::ParamBounds {
    fn clean(&self) -> Vec<TyParamBound> {
        let mut v = Vec::new();
        for b in self.builtin_bounds.iter() {
            if b != ty::BoundSized {
                v.push(b.clean());
            }
        }
        for t in self.trait_bounds.iter() {
            v.push(t.clean());
        }
        return v;
    }
}

impl Clean<Option<Vec<TyParamBound>>> for subst::Substs {
    fn clean(&self) -> Option<Vec<TyParamBound>> {
        let mut v = Vec::new();
        v.extend(self.regions().iter().map(|_| RegionBound));
        v.extend(self.types.iter().map(|t| TraitBound(t.clean())));
        if v.len() > 0 {Some(v)} else {None}
    }
}

#[deriving(Clone, Encodable, Decodable, PartialEq)]
pub struct Lifetime(String);

impl Lifetime {
    pub fn get_ref<'a>(&'a self) -> &'a str {
        let Lifetime(ref s) = *self;
        let s: &'a str = s.as_slice();
        return s;
    }
}

impl Clean<Lifetime> for ast::Lifetime {
    fn clean(&self) -> Lifetime {
        Lifetime(token::get_name(self.name).get().to_string())
    }
}

impl Clean<Lifetime> for ty::RegionParameterDef {
    fn clean(&self) -> Lifetime {
        Lifetime(token::get_name(self.name).get().to_string())
    }
}

impl Clean<Option<Lifetime>> for ty::Region {
    fn clean(&self) -> Option<Lifetime> {
        match *self {
            ty::ReStatic => Some(Lifetime("'static".to_string())),
            ty::ReLateBound(_, ty::BrNamed(_, name)) =>
                Some(Lifetime(token::get_name(name).get().to_string())),
            ty::ReEarlyBound(_, _, _, name) => Some(Lifetime(name.clean())),

            ty::ReLateBound(..) |
            ty::ReFree(..) |
            ty::ReScope(..) |
            ty::ReInfer(..) |
            ty::ReEmpty(..) => None
        }
    }
}

// maybe use a Generic enum and use ~[Generic]?
#[deriving(Clone, Encodable, Decodable)]
pub struct Generics {
    pub lifetimes: Vec<Lifetime>,
    pub type_params: Vec<TyParam>,
}

impl Clean<Generics> for ast::Generics {
    fn clean(&self) -> Generics {
        Generics {
            lifetimes: self.lifetimes.clean(),
            type_params: self.ty_params.clean(),
        }
    }
}

impl Clean<Generics> for ty::Generics {
    fn clean(&self) -> Generics {
        // In the type space, generics can come in one of multiple
        // namespaces.  This means that e.g. for fn items the type
        // parameters will live in FnSpace, but for types the
        // parameters will live in TypeSpace (trait definitions also
        // define a parameter in SelfSpace). *Method* definitions are
        // the one exception: they combine the TypeSpace parameters
        // from the enclosing impl/trait with their own FnSpace
        // parameters.
        //
        // In general, when we clean, we are trying to produce the
        // "user-facing" generics. Hence we select the most specific
        // namespace that is occupied, ignoring SelfSpace because it
        // is implicit.

        let space = {
            if !self.types.get_vec(subst::FnSpace).is_empty() ||
                !self.regions.get_vec(subst::FnSpace).is_empty()
            {
                subst::FnSpace
            } else {
                subst::TypeSpace
            }
        };

        Generics {
            type_params: self.types.get_vec(space).clean(),
            lifetimes: self.regions.get_vec(space).clean(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Method {
    pub generics: Generics,
    pub self_: SelfTy,
    pub fn_style: ast::FnStyle,
    pub decl: FnDecl,
}

impl Clean<Item> for ast::Method {
    fn clean(&self) -> Item {
        let inputs = match self.explicit_self.node {
            ast::SelfStatic => self.decl.inputs.as_slice(),
            _ => self.decl.inputs.slice_from(1)
        };
        let decl = FnDecl {
            inputs: Arguments {
                values: inputs.iter().map(|x| x.clean()).collect(),
            },
            output: (self.decl.output.clean()),
            cf: self.decl.cf.clean(),
            attrs: Vec::new()
        };
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            def_id: ast_util::local_def(self.id),
            visibility: self.vis.clean(),
            stability: get_stability(ast_util::local_def(self.id)),
            inner: MethodItem(Method {
                generics: self.generics.clean(),
                self_: self.explicit_self.node.clean(),
                fn_style: self.fn_style.clone(),
                decl: decl,
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct TyMethod {
    pub fn_style: ast::FnStyle,
    pub decl: FnDecl,
    pub generics: Generics,
    pub self_: SelfTy,
}

impl Clean<Item> for ast::TypeMethod {
    fn clean(&self) -> Item {
        let inputs = match self.explicit_self.node {
            ast::SelfStatic => self.decl.inputs.as_slice(),
            _ => self.decl.inputs.slice_from(1)
        };
        let decl = FnDecl {
            inputs: Arguments {
                values: inputs.iter().map(|x| x.clean()).collect(),
            },
            output: (self.decl.output.clean()),
            cf: self.decl.cf.clean(),
            attrs: Vec::new()
        };
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            def_id: ast_util::local_def(self.id),
            visibility: None,
            stability: get_stability(ast_util::local_def(self.id)),
            inner: TyMethodItem(TyMethod {
                fn_style: self.fn_style.clone(),
                decl: decl,
                self_: self.explicit_self.node.clean(),
                generics: self.generics.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable, PartialEq)]
pub enum SelfTy {
    SelfStatic,
    SelfValue,
    SelfBorrowed(Option<Lifetime>, Mutability),
    SelfOwned,
}

impl Clean<SelfTy> for ast::ExplicitSelf_ {
    fn clean(&self) -> SelfTy {
        match *self {
            ast::SelfStatic => SelfStatic,
            ast::SelfValue => SelfValue,
            ast::SelfUniq => SelfOwned,
            ast::SelfRegion(lt, mt) => SelfBorrowed(lt.clean(), mt.clean()),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Function {
    pub decl: FnDecl,
    pub generics: Generics,
    pub fn_style: ast::FnStyle,
}

impl Clean<Item> for doctree::Function {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            def_id: ast_util::local_def(self.id),
            inner: FunctionItem(Function {
                decl: self.decl.clean(),
                generics: self.generics.clean(),
                fn_style: self.fn_style,
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ClosureDecl {
    pub lifetimes: Vec<Lifetime>,
    pub decl: FnDecl,
    pub onceness: ast::Onceness,
    pub fn_style: ast::FnStyle,
    pub bounds: Vec<TyParamBound>,
}

impl Clean<ClosureDecl> for ast::ClosureTy {
    fn clean(&self) -> ClosureDecl {
        ClosureDecl {
            lifetimes: self.lifetimes.clean(),
            decl: self.decl.clean(),
            onceness: self.onceness,
            fn_style: self.fn_style,
            bounds: match self.bounds {
                Some(ref x) => x.clean().move_iter().collect(),
                None        => Vec::new()
            },
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct FnDecl {
    pub inputs: Arguments,
    pub output: Type,
    pub cf: RetStyle,
    pub attrs: Vec<Attribute>,
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Arguments {
    pub values: Vec<Argument>,
}

impl Clean<FnDecl> for ast::FnDecl {
    fn clean(&self) -> FnDecl {
        FnDecl {
            inputs: Arguments {
                values: self.inputs.iter().map(|x| x.clean()).collect(),
            },
            output: (self.output.clean()),
            cf: self.cf.clean(),
            attrs: Vec::new()
        }
    }
}

impl<'a> Clean<FnDecl> for (ast::DefId, &'a ty::FnSig) {
    fn clean(&self) -> FnDecl {
        let cx = get_cx();
        let (did, sig) = *self;
        let mut names = if did.node != 0 {
            csearch::get_method_arg_names(&cx.tcx().sess.cstore, did).move_iter()
        } else {
            Vec::new().move_iter()
        }.peekable();
        if names.peek().map(|s| s.as_slice()) == Some("self") {
            let _ = names.next();
        }
        FnDecl {
            output: sig.output.clean(),
            cf: Return,
            attrs: Vec::new(),
            inputs: Arguments {
                values: sig.inputs.iter().map(|t| {
                    Argument {
                        type_: t.clean(),
                        id: 0,
                        name: names.next().unwrap_or("".to_string()),
                    }
                }).collect(),
            },
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Argument {
    pub type_: Type,
    pub name: String,
    pub id: ast::NodeId,
}

impl Clean<Argument> for ast::Arg {
    fn clean(&self) -> Argument {
        Argument {
            name: name_from_pat(&*self.pat),
            type_: (self.ty.clean()),
            id: self.id
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum RetStyle {
    NoReturn,
    Return
}

impl Clean<RetStyle> for ast::RetStyle {
    fn clean(&self) -> RetStyle {
        match *self {
            ast::Return => Return,
            ast::NoReturn => NoReturn
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Trait {
    pub methods: Vec<TraitMethod>,
    pub generics: Generics,
    pub parents: Vec<Type>,
}

impl Clean<Item> for doctree::Trait {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            def_id: ast_util::local_def(self.id),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            inner: TraitItem(Trait {
                methods: self.methods.clean(),
                generics: self.generics.clean(),
                parents: self.parents.clean(),
            }),
        }
    }
}

impl Clean<Type> for ast::TraitRef {
    fn clean(&self) -> Type {
        resolve_type(self.path.clean(), None, self.ref_id)
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum TraitMethod {
    Required(Item),
    Provided(Item),
}

impl TraitMethod {
    pub fn is_req(&self) -> bool {
        match self {
            &Required(..) => true,
            _ => false,
        }
    }
    pub fn is_def(&self) -> bool {
        match self {
            &Provided(..) => true,
            _ => false,
        }
    }
    pub fn item<'a>(&'a self) -> &'a Item {
        match *self {
            Required(ref item) => item,
            Provided(ref item) => item,
        }
    }
}

impl Clean<TraitMethod> for ast::TraitMethod {
    fn clean(&self) -> TraitMethod {
        match self {
            &ast::Required(ref t) => Required(t.clean()),
            &ast::Provided(ref t) => Provided(t.clean()),
        }
    }
}

impl Clean<Item> for ty::Method {
    fn clean(&self) -> Item {
        let cx = get_cx();
        let (self_, sig) = match self.explicit_self {
            ast::SelfStatic => (ast::SelfStatic.clean(), self.fty.sig.clone()),
            s => {
                let sig = ty::FnSig {
                    inputs: Vec::from_slice(self.fty.sig.inputs.slice_from(1)),
                    ..self.fty.sig.clone()
                };
                let s = match s {
                    ast::SelfRegion(..) => {
                        match ty::get(*self.fty.sig.inputs.get(0)).sty {
                            ty::ty_rptr(r, mt) => {
                                SelfBorrowed(r.clean(), mt.mutbl.clean())
                            }
                            _ => s.clean(),
                        }
                    }
                    s => s.clean(),
                };
                (s, sig)
            }
        };

        Item {
            name: Some(self.ident.clean()),
            visibility: Some(ast::Inherited),
            stability: get_stability(self.def_id),
            def_id: self.def_id,
            attrs: inline::load_attrs(cx.tcx(), self.def_id),
            source: Span::empty(),
            inner: TyMethodItem(TyMethod {
                fn_style: self.fty.fn_style,
                generics: self.generics.clean(),
                self_: self_,
                decl: (self.def_id, &sig).clean(),
            })
        }
    }
}

/// A representation of a Type suitable for hyperlinking purposes. Ideally one can get the original
/// type out of the AST/ty::ctxt given one of these, if more information is needed. Most importantly
/// it does not preserve mutability or boxes.
#[deriving(Clone, Encodable, Decodable)]
pub enum Type {
    /// structs/enums/traits (anything that'd be an ast::TyPath)
    ResolvedPath {
        pub path: Path,
        pub typarams: Option<Vec<TyParamBound>>,
        pub did: ast::DefId,
    },
    // I have no idea how to usefully use this.
    TyParamBinder(ast::NodeId),
    /// For parameterized types, so the consumer of the JSON don't go looking
    /// for types which don't exist anywhere.
    Generic(ast::DefId),
    /// For references to self
    Self(ast::DefId),
    /// Primitives are just the fixed-size numeric types (plus int/uint/float), and char.
    Primitive(Primitive),
    Closure(Box<ClosureDecl>, Option<Lifetime>),
    Proc(Box<ClosureDecl>),
    /// extern "ABI" fn
    BareFunction(Box<BareFunctionDecl>),
    Tuple(Vec<Type>),
    Vector(Box<Type>),
    FixedVector(Box<Type>, String),
    /// aka TyBot
    Bottom,
    Unique(Box<Type>),
    Managed(Box<Type>),
    RawPointer(Mutability, Box<Type>),
    BorrowedRef {
        pub lifetime: Option<Lifetime>,
        pub mutability: Mutability,
        pub type_: Box<Type>,
    },
    // region, raw, other boxes, mutable
}

#[deriving(Clone, Encodable, Decodable, PartialEq, Eq, Hash)]
pub enum Primitive {
    Int, I8, I16, I32, I64,
    Uint, U8, U16, U32, U64,
    F32, F64,
    Char,
    Bool,
    Nil,
    Str,
    Slice,
    PrimitiveTuple,
}

#[deriving(Clone, Encodable, Decodable)]
pub enum TypeKind {
    TypeEnum,
    TypeFunction,
    TypeModule,
    TypeStatic,
    TypeStruct,
    TypeTrait,
    TypeVariant,
}

impl Primitive {
    fn from_str(s: &str) -> Option<Primitive> {
        match s.as_slice() {
            "int" => Some(Int),
            "i8" => Some(I8),
            "i16" => Some(I16),
            "i32" => Some(I32),
            "i64" => Some(I64),
            "uint" => Some(Uint),
            "u8" => Some(U8),
            "u16" => Some(U16),
            "u32" => Some(U32),
            "u64" => Some(U64),
            "bool" => Some(Bool),
            "nil" => Some(Nil),
            "char" => Some(Char),
            "str" => Some(Str),
            "f32" => Some(F32),
            "f64" => Some(F64),
            "slice" => Some(Slice),
            "tuple" => Some(PrimitiveTuple),
            _ => None,
        }
    }

    fn find(attrs: &[Attribute]) -> Option<Primitive> {
        for attr in attrs.iter() {
            let list = match *attr {
                List(ref k, ref l) if k.as_slice() == "doc" => l,
                _ => continue,
            };
            for sub_attr in list.iter() {
                let value = match *sub_attr {
                    NameValue(ref k, ref v)
                        if k.as_slice() == "primitive" => v.as_slice(),
                    _ => continue,
                };
                match Primitive::from_str(value) {
                    Some(p) => return Some(p),
                    None => {}
                }
            }
        }
        return None
    }

    pub fn to_str(&self) -> &'static str {
        match *self {
            Int => "int",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            Uint => "uint",
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            F32 => "f32",
            F64 => "f64",
            Str => "str",
            Bool => "bool",
            Char => "char",
            Nil => "()",
            Slice => "slice",
            PrimitiveTuple => "tuple",
        }
    }

    pub fn to_url_str(&self) -> &'static str {
        match *self {
            Nil => "nil",
            other => other.to_str(),
        }
    }

    /// Creates a rustdoc-specific node id for primitive types.
    ///
    /// These node ids are generally never used by the AST itself.
    pub fn to_node_id(&self) -> ast::NodeId {
        u32::MAX - 1 - (*self as u32)
    }
}

impl Clean<Type> for ast::Ty {
    fn clean(&self) -> Type {
        use syntax::ast::*;
        match self.node {
            TyNil => Primitive(Nil),
            TyPtr(ref m) => RawPointer(m.mutbl.clean(), box m.ty.clean()),
            TyRptr(ref l, ref m) =>
                BorrowedRef {lifetime: l.clean(), mutability: m.mutbl.clean(),
                             type_: box m.ty.clean()},
            TyBox(ty) => Managed(box ty.clean()),
            TyUniq(ty) => Unique(box ty.clean()),
            TyVec(ty) => Vector(box ty.clean()),
            TyFixedLengthVec(ty, ref e) => FixedVector(box ty.clean(),
                                                       e.span.to_src()),
            TyTup(ref tys) => Tuple(tys.iter().map(|x| x.clean()).collect()),
            TyPath(ref p, ref tpbs, id) => {
                resolve_type(p.clean(),
                             tpbs.clean().map(|x| x.move_iter().collect()),
                             id)
            }
            TyClosure(ref c, region) => Closure(box c.clean(), region.clean()),
            TyProc(ref c) => Proc(box c.clean()),
            TyBareFn(ref barefn) => BareFunction(box barefn.clean()),
            TyParen(ref ty) => ty.clean(),
            TyBot => Bottom,
            ref x => fail!("Unimplemented type {:?}", x),
        }
    }
}

impl Clean<Type> for ty::t {
    fn clean(&self) -> Type {
        match ty::get(*self).sty {
            ty::ty_bot => Bottom,
            ty::ty_nil => Primitive(Nil),
            ty::ty_bool => Primitive(Bool),
            ty::ty_char => Primitive(Char),
            ty::ty_int(ast::TyI) => Primitive(Int),
            ty::ty_int(ast::TyI8) => Primitive(I8),
            ty::ty_int(ast::TyI16) => Primitive(I16),
            ty::ty_int(ast::TyI32) => Primitive(I32),
            ty::ty_int(ast::TyI64) => Primitive(I64),
            ty::ty_uint(ast::TyU) => Primitive(Uint),
            ty::ty_uint(ast::TyU8) => Primitive(U8),
            ty::ty_uint(ast::TyU16) => Primitive(U16),
            ty::ty_uint(ast::TyU32) => Primitive(U32),
            ty::ty_uint(ast::TyU64) => Primitive(U64),
            ty::ty_float(ast::TyF32) => Primitive(F32),
            ty::ty_float(ast::TyF64) => Primitive(F64),
            ty::ty_str => Primitive(Str),
            ty::ty_box(t) => Managed(box t.clean()),
            ty::ty_uniq(t) => Unique(box t.clean()),
            ty::ty_vec(mt, None) => Vector(box mt.ty.clean()),
            ty::ty_vec(mt, Some(i)) => FixedVector(box mt.ty.clean(),
                                                   format!("{}", i)),
            ty::ty_ptr(mt) => RawPointer(mt.mutbl.clean(), box mt.ty.clean()),
            ty::ty_rptr(r, mt) => BorrowedRef {
                lifetime: r.clean(),
                mutability: mt.mutbl.clean(),
                type_: box mt.ty.clean(),
            },
            ty::ty_bare_fn(ref fty) => BareFunction(box BareFunctionDecl {
                fn_style: fty.fn_style,
                generics: Generics {
                    lifetimes: Vec::new(), type_params: Vec::new()
                },
                decl: (ast_util::local_def(0), &fty.sig).clean(),
                abi: fty.abi.to_str(),
            }),
            ty::ty_closure(ref fty) => {
                let decl = box ClosureDecl {
                    lifetimes: Vec::new(), // FIXME: this looks wrong...
                    decl: (ast_util::local_def(0), &fty.sig).clean(),
                    onceness: fty.onceness,
                    fn_style: fty.fn_style,
                    bounds: fty.bounds.iter().map(|i| i.clean()).collect(),
                };
                match fty.store {
                    ty::UniqTraitStore => Proc(decl),
                    ty::RegionTraitStore(ref r, _) => Closure(decl, r.clean()),
                }
            }
            ty::ty_struct(did, ref substs) |
            ty::ty_enum(did, ref substs) |
            ty::ty_trait(box ty::TyTrait { def_id: did, ref substs, .. }) => {
                let fqn = csearch::get_item_path(get_cx().tcx(), did);
                let fqn: Vec<String> = fqn.move_iter().map(|i| {
                    i.to_str()
                }).collect();
                let kind = match ty::get(*self).sty {
                    ty::ty_struct(..) => TypeStruct,
                    ty::ty_trait(..) => TypeTrait,
                    _ => TypeEnum,
                };
                let path = external_path(fqn.last().unwrap().to_str().as_slice(),
                                         substs);
                get_cx().external_paths.borrow_mut().get_mut_ref()
                                       .insert(did, (fqn, kind));
                ResolvedPath {
                    path: path,
                    typarams: None,
                    did: did,
                }
            }
            ty::ty_tup(ref t) => Tuple(t.iter().map(|t| t.clean()).collect()),

            ty::ty_param(ref p) => {
                if p.space == subst::SelfSpace {
                    Self(p.def_id)
                } else {
                    Generic(p.def_id)
                }
            }

            ty::ty_infer(..) => fail!("ty_infer"),
            ty::ty_err => fail!("ty_err"),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum StructField {
    HiddenStructField, // inserted later by strip passes
    TypedStructField(Type),
}

impl Clean<Item> for ast::StructField {
    fn clean(&self) -> Item {
        let (name, vis) = match self.node.kind {
            ast::NamedField(id, vis) => (Some(id), vis),
            ast::UnnamedField(vis) => (None, vis)
        };
        Item {
            name: name.clean(),
            attrs: self.node.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            visibility: Some(vis),
            stability: get_stability(ast_util::local_def(self.node.id)),
            def_id: ast_util::local_def(self.node.id),
            inner: StructFieldItem(TypedStructField(self.node.ty.clean())),
        }
    }
}

impl Clean<Item> for ty::field_ty {
    fn clean(&self) -> Item {
        use syntax::parse::token::special_idents::unnamed_field;
        let name = if self.name == unnamed_field.name {
            None
        } else {
            Some(self.name)
        };
        let cx = get_cx();
        let ty = ty::lookup_item_type(cx.tcx(), self.id);
        Item {
            name: name.clean(),
            attrs: inline::load_attrs(cx.tcx(), self.id),
            source: Span::empty(),
            visibility: Some(self.vis),
            stability: get_stability(self.id),
            def_id: self.id,
            inner: StructFieldItem(TypedStructField(ty.ty.clean())),
        }
    }
}

pub type Visibility = ast::Visibility;

impl Clean<Option<Visibility>> for ast::Visibility {
    fn clean(&self) -> Option<Visibility> {
        Some(*self)
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Struct {
    pub struct_type: doctree::StructType,
    pub generics: Generics,
    pub fields: Vec<Item>,
    pub fields_stripped: bool,
}

impl Clean<Item> for doctree::Struct {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            def_id: ast_util::local_def(self.id),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            inner: StructItem(Struct {
                struct_type: self.struct_type,
                generics: self.generics.clean(),
                fields: self.fields.clean(),
                fields_stripped: false,
            }),
        }
    }
}

/// This is a more limited form of the standard Struct, different in that
/// it lacks the things most items have (name, id, parameterization). Found
/// only as a variant in an enum.
#[deriving(Clone, Encodable, Decodable)]
pub struct VariantStruct {
    pub struct_type: doctree::StructType,
    pub fields: Vec<Item>,
    pub fields_stripped: bool,
}

impl Clean<VariantStruct> for syntax::ast::StructDef {
    fn clean(&self) -> VariantStruct {
        VariantStruct {
            struct_type: doctree::struct_type_from_def(self),
            fields: self.fields.clean().move_iter().collect(),
            fields_stripped: false,
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Enum {
    pub variants: Vec<Item>,
    pub generics: Generics,
    pub variants_stripped: bool,
}

impl Clean<Item> for doctree::Enum {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            def_id: ast_util::local_def(self.id),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            inner: EnumItem(Enum {
                variants: self.variants.clean(),
                generics: self.generics.clean(),
                variants_stripped: false,
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Variant {
    pub kind: VariantKind,
}

impl Clean<Item> for doctree::Variant {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            def_id: ast_util::local_def(self.id),
            inner: VariantItem(Variant {
                kind: self.kind.clean(),
            }),
        }
    }
}

impl Clean<Item> for ty::VariantInfo {
    fn clean(&self) -> Item {
        // use syntax::parse::token::special_idents::unnamed_field;
        let cx = get_cx();
        let kind = match self.arg_names.as_ref().map(|s| s.as_slice()) {
            None | Some([]) if self.args.len() == 0 => CLikeVariant,
            None | Some([]) => {
                TupleVariant(self.args.iter().map(|t| t.clean()).collect())
            }
            Some(s) => {
                StructVariant(VariantStruct {
                    struct_type: doctree::Plain,
                    fields_stripped: false,
                    fields: s.iter().zip(self.args.iter()).map(|(name, ty)| {
                        Item {
                            source: Span::empty(),
                            name: Some(name.clean()),
                            attrs: Vec::new(),
                            visibility: Some(ast::Public),
                            stability: get_stability(self.id),
                            // FIXME: this is not accurate, we need an id for
                            //        the specific field but we're using the id
                            //        for the whole variant. Nothing currently
                            //        uses this so we should be good for now.
                            def_id: self.id,
                            inner: StructFieldItem(
                                TypedStructField(ty.clean())
                            )
                        }
                    }).collect()
                })
            }
        };
        Item {
            name: Some(self.name.clean()),
            attrs: inline::load_attrs(cx.tcx(), self.id),
            source: Span::empty(),
            visibility: Some(ast::Public),
            def_id: self.id,
            inner: VariantItem(Variant { kind: kind }),
            stability: None,
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum VariantKind {
    CLikeVariant,
    TupleVariant(Vec<Type>),
    StructVariant(VariantStruct),
}

impl Clean<VariantKind> for ast::VariantKind {
    fn clean(&self) -> VariantKind {
        match self {
            &ast::TupleVariantKind(ref args) => {
                if args.len() == 0 {
                    CLikeVariant
                } else {
                    TupleVariant(args.iter().map(|x| x.ty.clean()).collect())
                }
            },
            &ast::StructVariantKind(ref sd) => StructVariant(sd.clean()),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Span {
    pub filename: String,
    pub loline: uint,
    pub locol: uint,
    pub hiline: uint,
    pub hicol: uint,
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
    fn clean(&self) -> Span {
        let ctxt = super::ctxtkey.get().unwrap();
        let cm = ctxt.sess().codemap();
        let filename = cm.span_to_filename(*self);
        let lo = cm.lookup_char_pos(self.lo);
        let hi = cm.lookup_char_pos(self.hi);
        Span {
            filename: filename.to_string(),
            loline: lo.line,
            locol: lo.col.to_uint(),
            hiline: hi.line,
            hicol: hi.col.to_uint(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Path {
    pub global: bool,
    pub segments: Vec<PathSegment>,
}

impl Clean<Path> for ast::Path {
    fn clean(&self) -> Path {
        Path {
            global: self.global,
            segments: self.segments.clean().move_iter().collect(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct PathSegment {
    pub name: String,
    pub lifetimes: Vec<Lifetime>,
    pub types: Vec<Type>,
}

impl Clean<PathSegment> for ast::PathSegment {
    fn clean(&self) -> PathSegment {
        PathSegment {
            name: self.identifier.clean(),
            lifetimes: self.lifetimes.clean().move_iter().collect(),
            types: self.types.clean().move_iter().collect()
        }
    }
}

fn path_to_str(p: &ast::Path) -> String {
    use syntax::parse::token;

    let mut s = String::new();
    let mut first = true;
    for i in p.segments.iter().map(|x| token::get_ident(x.identifier)) {
        if !first || p.global {
            s.push_str("::");
        } else {
            first = false;
        }
        s.push_str(i.get());
    }
    s
}

impl Clean<String> for ast::Ident {
    fn clean(&self) -> String {
        token::get_ident(*self).get().to_string()
    }
}

impl Clean<String> for ast::Name {
    fn clean(&self) -> String {
        token::get_name(*self).get().to_string()
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Typedef {
    pub type_: Type,
    pub generics: Generics,
}

impl Clean<Item> for doctree::Typedef {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            def_id: ast_util::local_def(self.id.clone()),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            inner: TypedefItem(Typedef {
                type_: self.ty.clean(),
                generics: self.gen.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct BareFunctionDecl {
    pub fn_style: ast::FnStyle,
    pub generics: Generics,
    pub decl: FnDecl,
    pub abi: String,
}

impl Clean<BareFunctionDecl> for ast::BareFnTy {
    fn clean(&self) -> BareFunctionDecl {
        BareFunctionDecl {
            fn_style: self.fn_style,
            generics: Generics {
                lifetimes: self.lifetimes.clean().move_iter().collect(),
                type_params: Vec::new(),
            },
            decl: self.decl.clean(),
            abi: self.abi.to_str(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Static {
    pub type_: Type,
    pub mutability: Mutability,
    /// It's useful to have the value of a static documented, but I have no
    /// desire to represent expressions (that'd basically be all of the AST,
    /// which is huge!). So, have a string.
    pub expr: String,
}

impl Clean<Item> for doctree::Static {
    fn clean(&self) -> Item {
        debug!("claning static {}: {:?}", self.name.clean(), self);
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            def_id: ast_util::local_def(self.id),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            inner: StaticItem(Static {
                type_: self.type_.clean(),
                mutability: self.mutability.clean(),
                expr: self.expr.span.to_src(),
            }),
        }
    }
}

#[deriving(Show, Clone, Encodable, Decodable, PartialEq)]
pub enum Mutability {
    Mutable,
    Immutable,
}

impl Clean<Mutability> for ast::Mutability {
    fn clean(&self) -> Mutability {
        match self {
            &ast::MutMutable => Mutable,
            &ast::MutImmutable => Immutable,
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Impl {
    pub generics: Generics,
    pub trait_: Option<Type>,
    pub for_: Type,
    pub methods: Vec<Item>,
    pub derived: bool,
}

fn detect_derived<M: AttrMetaMethods>(attrs: &[M]) -> bool {
    attr::contains_name(attrs, "automatically_derived")
}

impl Clean<Item> for doctree::Impl {
    fn clean(&self) -> Item {
        Item {
            name: None,
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            def_id: ast_util::local_def(self.id),
            visibility: self.vis.clean(),
            stability: self.stab.clean(),
            inner: ImplItem(Impl {
                generics: self.generics.clean(),
                trait_: self.trait_.clean(),
                for_: self.for_.clean(),
                methods: self.methods.clean(),
                derived: detect_derived(self.attrs.as_slice()),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ViewItem {
    pub inner: ViewItemInner,
}

impl Clean<Vec<Item>> for ast::ViewItem {
    fn clean(&self) -> Vec<Item> {
        // We consider inlining the documentation of `pub use` statments, but we
        // forcefully don't inline if this is not public or if the
        // #[doc(no_inline)] attribute is present.
        let denied = self.vis != ast::Public || self.attrs.iter().any(|a| {
            a.name().get() == "doc" && match a.meta_item_list() {
                Some(l) => attr::contains_name(l, "no_inline"),
                None => false,
            }
        });
        let convert = |node: &ast::ViewItem_| {
            Item {
                name: None,
                attrs: self.attrs.clean().move_iter().collect(),
                source: self.span.clean(),
                def_id: ast_util::local_def(0),
                visibility: self.vis.clean(),
                stability: None,
                inner: ViewItemItem(ViewItem { inner: node.clean() }),
            }
        };
        let mut ret = Vec::new();
        match self.node {
            ast::ViewItemUse(ref path) if !denied => {
                match path.node {
                    ast::ViewPathGlob(..) => ret.push(convert(&self.node)),
                    ast::ViewPathList(ref a, ref list, ref b) => {
                        // Attempt to inline all reexported items, but be sure
                        // to keep any non-inlineable reexports so they can be
                        // listed in the documentation.
                        let remaining = list.iter().filter(|path| {
                            match inline::try_inline(path.node.id) {
                                Some(items) => {
                                    ret.extend(items.move_iter()); false
                                }
                                None => true,
                            }
                        }).map(|a| a.clone()).collect::<Vec<ast::PathListIdent>>();
                        if remaining.len() > 0 {
                            let path = ast::ViewPathList(a.clone(),
                                                         remaining,
                                                         b.clone());
                            let path = syntax::codemap::dummy_spanned(path);
                            ret.push(convert(&ast::ViewItemUse(box(GC) path)));
                        }
                    }
                    ast::ViewPathSimple(_, _, id) => {
                        match inline::try_inline(id) {
                            Some(items) => ret.extend(items.move_iter()),
                            None => ret.push(convert(&self.node)),
                        }
                    }
                }
            }
            ref n => ret.push(convert(n)),
        }
        return ret;
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum ViewItemInner {
    ExternCrate(String, Option<String>, ast::NodeId),
    Import(ViewPath)
}

impl Clean<ViewItemInner> for ast::ViewItem_ {
    fn clean(&self) -> ViewItemInner {
        match self {
            &ast::ViewItemExternCrate(ref i, ref p, ref id) => {
                let string = match *p {
                    None => None,
                    Some((ref x, _)) => Some(x.get().to_string()),
                };
                ExternCrate(i.clean(), string, *id)
            }
            &ast::ViewItemUse(ref vp) => {
                Import(vp.clean())
            }
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum ViewPath {
    // use str = source;
    SimpleImport(String, ImportSource),
    // use source::*;
    GlobImport(ImportSource),
    // use source::{a, b, c};
    ImportList(ImportSource, Vec<ViewListIdent>),
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ImportSource {
    pub path: Path,
    pub did: Option<ast::DefId>,
}

impl Clean<ViewPath> for ast::ViewPath {
    fn clean(&self) -> ViewPath {
        match self.node {
            ast::ViewPathSimple(ref i, ref p, id) =>
                SimpleImport(i.clean(), resolve_use_source(p.clean(), id)),
            ast::ViewPathGlob(ref p, id) =>
                GlobImport(resolve_use_source(p.clean(), id)),
            ast::ViewPathList(ref p, ref pl, id) => {
                ImportList(resolve_use_source(p.clean(), id),
                           pl.clean().move_iter().collect())
            }
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ViewListIdent {
    pub name: String,
    pub source: Option<ast::DefId>,
}

impl Clean<ViewListIdent> for ast::PathListIdent {
    fn clean(&self) -> ViewListIdent {
        ViewListIdent {
            name: self.node.name.clean(),
            source: resolve_def(self.node.id),
        }
    }
}

impl Clean<Vec<Item>> for ast::ForeignMod {
    fn clean(&self) -> Vec<Item> {
        self.items.clean()
    }
}

impl Clean<Item> for ast::ForeignItem {
    fn clean(&self) -> Item {
        let inner = match self.node {
            ast::ForeignItemFn(ref decl, ref generics) => {
                ForeignFunctionItem(Function {
                    decl: decl.clean(),
                    generics: generics.clean(),
                    fn_style: ast::UnsafeFn,
                })
            }
            ast::ForeignItemStatic(ref ty, mutbl) => {
                ForeignStaticItem(Static {
                    type_: ty.clean(),
                    mutability: if mutbl {Mutable} else {Immutable},
                    expr: "".to_string(),
                })
            }
        };
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            def_id: ast_util::local_def(self.id),
            visibility: self.vis.clean(),
            stability: None,
            inner: inner,
        }
    }
}

// Utilities

trait ToSource {
    fn to_src(&self) -> String;
}

impl ToSource for syntax::codemap::Span {
    fn to_src(&self) -> String {
        debug!("converting span {:?} to snippet", self.clean());
        let ctxt = super::ctxtkey.get().unwrap();
        let cm = ctxt.sess().codemap().clone();
        let sn = match cm.span_to_snippet(*self) {
            Some(x) => x.to_string(),
            None    => "".to_string()
        };
        debug!("got snippet {}", sn);
        sn
    }
}

fn lit_to_str(lit: &ast::Lit) -> String {
    match lit.node {
        ast::LitStr(ref st, _) => st.get().to_string(),
        ast::LitBinary(ref data) => format!("{:?}", data.as_slice()),
        ast::LitByte(b) => {
            let mut res = String::from_str("b'");
            (b as char).escape_default(|c| {
                res.push_char(c);
            });
            res.push_char('\'');
            res
        },
        ast::LitChar(c) => format!("'{}'", c),
        ast::LitInt(i, _t) => i.to_str(),
        ast::LitUint(u, _t) => u.to_str(),
        ast::LitIntUnsuffixed(i) => i.to_str(),
        ast::LitFloat(ref f, _t) => f.get().to_string(),
        ast::LitFloatUnsuffixed(ref f) => f.get().to_string(),
        ast::LitBool(b) => b.to_str(),
        ast::LitNil => "".to_string(),
    }
}

fn name_from_pat(p: &ast::Pat) -> String {
    use syntax::ast::*;
    debug!("Trying to get a name from pattern: {:?}", p);

    match p.node {
        PatWild => "_".to_string(),
        PatWildMulti => "..".to_string(),
        PatIdent(_, ref p, _) => path_to_str(p),
        PatEnum(ref p, _) => path_to_str(p),
        PatStruct(..) => fail!("tried to get argument name from pat_struct, \
                                which is not allowed in function arguments"),
        PatTup(..) => "(tuple arg NYI)".to_string(),
        PatBox(p) => name_from_pat(&*p),
        PatRegion(p) => name_from_pat(&*p),
        PatLit(..) => {
            warn!("tried to get argument name from PatLit, \
                  which is silly in function arguments");
            "()".to_string()
        },
        PatRange(..) => fail!("tried to get argument name from PatRange, \
                              which is not allowed in function arguments"),
        PatVec(..) => fail!("tried to get argument name from pat_vec, \
                             which is not allowed in function arguments"),
        PatMac(..) => {
            warn!("can't document the name of a function argument \
                   produced by a pattern macro");
            "(argument produced by macro)".to_string()
        }
    }
}

/// Given a Type, resolve it using the def_map
fn resolve_type(path: Path, tpbs: Option<Vec<TyParamBound>>,
                id: ast::NodeId) -> Type {
    let cx = get_cx();
    let tycx = match cx.maybe_typed {
        core::Typed(ref tycx) => tycx,
        // If we're extracting tests, this return value doesn't matter.
        core::NotTyped(_) => return Primitive(Bool),
    };
    debug!("searching for {:?} in defmap", id);
    let def = match tycx.def_map.borrow().find(&id) {
        Some(&k) => k,
        None => fail!("unresolved id not in defmap")
    };

    match def {
        def::DefSelfTy(i) => return Self(ast_util::local_def(i)),
        def::DefPrimTy(p) => match p {
            ast::TyStr => return Primitive(Str),
            ast::TyBool => return Primitive(Bool),
            ast::TyChar => return Primitive(Char),
            ast::TyInt(ast::TyI) => return Primitive(Int),
            ast::TyInt(ast::TyI8) => return Primitive(I8),
            ast::TyInt(ast::TyI16) => return Primitive(I16),
            ast::TyInt(ast::TyI32) => return Primitive(I32),
            ast::TyInt(ast::TyI64) => return Primitive(I64),
            ast::TyUint(ast::TyU) => return Primitive(Uint),
            ast::TyUint(ast::TyU8) => return Primitive(U8),
            ast::TyUint(ast::TyU16) => return Primitive(U16),
            ast::TyUint(ast::TyU32) => return Primitive(U32),
            ast::TyUint(ast::TyU64) => return Primitive(U64),
            ast::TyFloat(ast::TyF32) => return Primitive(F32),
            ast::TyFloat(ast::TyF64) => return Primitive(F64),
        },
        def::DefTyParam(_, i, _) => return Generic(i),
        def::DefTyParamBinder(i) => return TyParamBinder(i),
        _ => {}
    };
    let did = register_def(&*cx, def);
    ResolvedPath { path: path, typarams: tpbs, did: did }
}

fn register_def(cx: &core::DocContext, def: def::Def) -> ast::DefId {
    let (did, kind) = match def {
        def::DefFn(i, _) => (i, TypeFunction),
        def::DefTy(i) => (i, TypeEnum),
        def::DefTrait(i) => (i, TypeTrait),
        def::DefStruct(i) => (i, TypeStruct),
        def::DefMod(i) => (i, TypeModule),
        def::DefStatic(i, _) => (i, TypeStatic),
        def::DefVariant(i, _, _) => (i, TypeEnum),
        _ => return def.def_id()
    };
    if ast_util::is_local(did) { return did }
    let tcx = match cx.maybe_typed {
        core::Typed(ref t) => t,
        core::NotTyped(_) => return did
    };
    inline::record_extern_fqn(cx, did, kind);
    match kind {
        TypeTrait => {
            let t = inline::build_external_trait(tcx, did);
            cx.external_traits.borrow_mut().get_mut_ref().insert(did, t);
        }
        _ => {}
    }
    return did;
}

fn resolve_use_source(path: Path, id: ast::NodeId) -> ImportSource {
    ImportSource {
        path: path,
        did: resolve_def(id),
    }
}

fn resolve_def(id: ast::NodeId) -> Option<ast::DefId> {
    get_cx().tcx_opt().and_then(|tcx| {
        tcx.def_map.borrow().find(&id).map(|&def| register_def(&*get_cx(), def))
    })
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Macro {
    pub source: String,
}

impl Clean<Item> for doctree::Macro {
    fn clean(&self) -> Item {
        Item {
            name: Some(format!("{}!", self.name.clean())),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: ast::Public.clean(),
            stability: self.stab.clean(),
            def_id: ast_util::local_def(self.id),
            inner: MacroItem(Macro {
                source: self.where.to_src(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Stability {
    pub level: attr::StabilityLevel,
    pub text: String
}

impl Clean<Stability> for attr::Stability {
    fn clean(&self) -> Stability {
        Stability {
            level: self.level,
            text: self.text.as_ref().map_or("".to_string(),
                                            |interned| interned.get().to_string()),
        }
    }
}
