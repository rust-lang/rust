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
use syntax::attr::AttributeMethods;
use syntax::codemap::Pos;
use syntax::parse::token::InternedString;
use syntax::parse::token;

use rustc::metadata::cstore;
use rustc::metadata::csearch;
use rustc::metadata::decoder;

use std;

use doctree;
use visit_ast;
use std::local_data;
use std::vec_ng::Vec;

pub trait Clean<T> {
    fn clean(&self) -> T;
}

impl<T: Clean<U>, U> Clean<~[U]> for ~[T] {
    fn clean(&self) -> ~[U] {
        self.iter().map(|x| x.clean()).collect()
    }
}

impl<T: Clean<U>, U> Clean<Vec<U>> for Vec<T> {
    fn clean(&self) -> Vec<U> {
        self.iter().map(|x| x.clean()).collect()
    }
}

impl<T: Clean<U>, U> Clean<U> for @T {
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

impl<T: Clean<U>, U> Clean<Vec<U>> for syntax::opt_vec::OptVec<T> {
    fn clean(&self) -> Vec<U> {
        match self {
            &syntax::opt_vec::Empty => Vec::new(),
            &syntax::opt_vec::Vec(ref v) => v.clean()
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Crate {
    name: ~str,
    module: Option<Item>,
    externs: ~[(ast::CrateNum, ExternalCrate)],
}

impl<'a> Clean<Crate> for visit_ast::RustdocVisitor<'a> {
    fn clean(&self) -> Crate {
        use syntax::attr::find_crateid;
        let cx = local_data::get(super::ctxtkey, |x| *x.unwrap());

        let mut externs = ~[];
        cx.sess.cstore.iter_crate_data(|n, meta| {
            externs.push((n, meta.clean()));
        });

        Crate {
            name: match find_crateid(self.attrs) {
                Some(n) => n.name,
                None => fail!("rustdoc requires a `crate_id` crate attribute"),
            },
            module: Some(self.module.clean()),
            externs: externs,
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ExternalCrate {
    name: ~str,
    attrs: ~[Attribute],
}

impl Clean<ExternalCrate> for cstore::crate_metadata {
    fn clean(&self) -> ExternalCrate {
        ExternalCrate {
            name: self.name.to_owned(),
            attrs: decoder::get_crate_attributes(self.data()).clean()
        }
    }
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[deriving(Clone, Encodable, Decodable)]
pub struct Item {
    /// Stringified span
    source: Span,
    /// Not everything has a name. E.g., impls
    name: Option<~str>,
    attrs: ~[Attribute],
    inner: ItemEnum,
    visibility: Option<Visibility>,
    id: ast::NodeId,
}

impl Item {
    /// Finds the `doc` attribute as a List and returns the list of attributes
    /// nested inside.
    pub fn doc_list<'a>(&'a self) -> Option<&'a [Attribute]> {
        for attr in self.attrs.iter() {
            match *attr {
                List(~"doc", ref list) => { return Some(list.as_slice()); }
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
                NameValue(~"doc", ref v) => { return Some(v.as_slice()); }
                _ => {}
            }
        }
        return None;
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
    ViewItemItem(ViewItem),
    TyMethodItem(TyMethod),
    MethodItem(Method),
    StructFieldItem(StructField),
    VariantItem(Variant),
    ForeignFunctionItem(Function),
    ForeignStaticItem(Static),
    MacroItem(Macro),
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Module {
    items: ~[Item],
    is_crate: bool,
}

impl Clean<Item> for doctree::Module {
    fn clean(&self) -> Item {
        let name = if self.name.is_some() {
            self.name.unwrap().clean()
        } else {
            ~""
        };
        let mut foreigns = ~[];
        for subforeigns in self.foreigns.clean().move_iter() {
            for foreign in subforeigns.move_iter() {
                foreigns.push(foreign)
            }
        }
        let items: ~[~[Item]] = ~[
            self.structs.clean().move_iter().collect(),
            self.enums.clean().move_iter().collect(),
            self.fns.clean().move_iter().collect(),
            foreigns,
            self.mods.clean().move_iter().collect(),
            self.typedefs.clean().move_iter().collect(),
            self.statics.clean().move_iter().collect(),
            self.traits.clean().move_iter().collect(),
            self.impls.clean().move_iter().collect(),
            self.view_items.clean().move_iter().collect(),
            self.macros.clean().move_iter().collect()
        ];
        Item {
            name: Some(name),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            id: self.id,
            inner: ModuleItem(Module {
               is_crate: self.is_crate,
               items: items.concat_vec(),
            })
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum Attribute {
    Word(~str),
    List(~str, ~[Attribute]),
    NameValue(~str, ~str)
}

impl Clean<Attribute> for ast::MetaItem {
    fn clean(&self) -> Attribute {
        match self.node {
            ast::MetaWord(ref s) => Word(s.get().to_owned()),
            ast::MetaList(ref s, ref l) => {
                List(s.get().to_owned(), l.clean().move_iter().collect())
            }
            ast::MetaNameValue(ref s, ref v) => {
                NameValue(s.get().to_owned(), lit_to_str(v))
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
impl<'a> attr::AttrMetaMethods for &'a Attribute {
    fn name(&self) -> InternedString {
        match **self {
            Word(ref n) | List(ref n, _) | NameValue(ref n, _) => {
                token::intern_and_get_ident(*n)
            }
        }
    }

    fn value_str(&self) -> Option<InternedString> {
        match **self {
            NameValue(_, ref v) => Some(token::intern_and_get_ident(*v)),
            _ => None,
        }
    }
    fn meta_item_list<'a>(&'a self) -> Option<&'a [@ast::MetaItem]> { None }
    fn name_str_pair(&self) -> Option<(InternedString, InternedString)> {
        None
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct TyParam {
    name: ~str,
    id: ast::NodeId,
    bounds: ~[TyParamBound]
}

impl Clean<TyParam> for ast::TyParam {
    fn clean(&self) -> TyParam {
        TyParam {
            name: self.ident.clean(),
            id: self.id,
            bounds: self.bounds.clean().move_iter().collect(),
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
            ast::RegionTyParamBound => RegionBound,
            ast::TraitTyParamBound(ref t) => TraitBound(t.clean()),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Lifetime(~str);

impl Lifetime {
    pub fn get_ref<'a>(&'a self) -> &'a str {
        let Lifetime(ref s) = *self;
        let s: &'a str = *s;
        return s;
    }
}

impl Clean<Lifetime> for ast::Lifetime {
    fn clean(&self) -> Lifetime {
        Lifetime(token::get_name(self.ident).get().to_owned())
    }
}

// maybe use a Generic enum and use ~[Generic]?
#[deriving(Clone, Encodable, Decodable)]
pub struct Generics {
    lifetimes: ~[Lifetime],
    type_params: ~[TyParam]
}

impl Clean<Generics> for ast::Generics {
    fn clean(&self) -> Generics {
        Generics {
            lifetimes: self.lifetimes.clean().move_iter().collect(),
            type_params: self.ty_params.clean().move_iter().collect(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Method {
    generics: Generics,
    self_: SelfTy,
    purity: ast::Purity,
    decl: FnDecl,
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
            attrs: ~[]
        };
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            id: self.id.clone(),
            visibility: self.vis.clean(),
            inner: MethodItem(Method {
                generics: self.generics.clean(),
                self_: self.explicit_self.clean(),
                purity: self.purity.clone(),
                decl: decl,
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct TyMethod {
    purity: ast::Purity,
    decl: FnDecl,
    generics: Generics,
    self_: SelfTy,
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
            attrs: ~[]
        };
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            id: self.id,
            visibility: None,
            inner: TyMethodItem(TyMethod {
                purity: self.purity.clone(),
                decl: decl,
                self_: self.explicit_self.clean(),
                generics: self.generics.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum SelfTy {
    SelfStatic,
    SelfValue,
    SelfBorrowed(Option<Lifetime>, Mutability),
    SelfOwned,
}

impl Clean<SelfTy> for ast::ExplicitSelf {
    fn clean(&self) -> SelfTy {
        match self.node {
            ast::SelfStatic => SelfStatic,
            ast::SelfValue => SelfValue,
            ast::SelfUniq => SelfOwned,
            ast::SelfRegion(lt, mt) => SelfBorrowed(lt.clean(), mt.clean()),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Function {
    decl: FnDecl,
    generics: Generics,
    purity: ast::Purity,
}

impl Clean<Item> for doctree::Function {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            id: self.id,
            inner: FunctionItem(Function {
                decl: self.decl.clean(),
                generics: self.generics.clean(),
                purity: self.purity,
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ClosureDecl {
    sigil: ast::Sigil,
    region: Option<Lifetime>,
    lifetimes: ~[Lifetime],
    decl: FnDecl,
    onceness: ast::Onceness,
    purity: ast::Purity,
    bounds: ~[TyParamBound]
}

impl Clean<ClosureDecl> for ast::ClosureTy {
    fn clean(&self) -> ClosureDecl {
        ClosureDecl {
            sigil: self.sigil,
            region: self.region.clean(),
            lifetimes: self.lifetimes.clean().move_iter().collect(),
            decl: self.decl.clean(),
            onceness: self.onceness,
            purity: self.purity,
            bounds: match self.bounds {
                Some(ref x) => x.clean().move_iter().collect(),
                None        => ~[]
            },
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct FnDecl {
    inputs: Arguments,
    output: Type,
    cf: RetStyle,
    attrs: ~[Attribute]
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Arguments {
    values: ~[Argument],
}

impl Clean<FnDecl> for ast::FnDecl {
    fn clean(&self) -> FnDecl {
        FnDecl {
            inputs: Arguments {
                values: self.inputs.iter().map(|x| x.clean()).collect(),
            },
            output: (self.output.clean()),
            cf: self.cf.clean(),
            attrs: ~[]
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Argument {
    type_: Type,
    name: ~str,
    id: ast::NodeId
}

impl Clean<Argument> for ast::Arg {
    fn clean(&self) -> Argument {
        Argument {
            name: name_from_pat(self.pat),
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
    methods: ~[TraitMethod],
    generics: Generics,
    parents: ~[Type],
}

impl Clean<Item> for doctree::Trait {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
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

/// A representation of a Type suitable for hyperlinking purposes. Ideally one can get the original
/// type out of the AST/ty::ctxt given one of these, if more information is needed. Most importantly
/// it does not preserve mutability or boxes.
#[deriving(Clone, Encodable, Decodable)]
pub enum Type {
    /// structs/enums/traits (anything that'd be an ast::TyPath)
    ResolvedPath {
        path: Path,
        typarams: Option<~[TyParamBound]>,
        id: ast::NodeId,
    },
    /// Same as above, but only external variants
    ExternalPath {
        path: Path,
        typarams: Option<~[TyParamBound]>,
        fqn: ~[~str],
        kind: TypeKind,
        krate: ast::CrateNum,
    },
    // I have no idea how to usefully use this.
    TyParamBinder(ast::NodeId),
    /// For parameterized types, so the consumer of the JSON don't go looking
    /// for types which don't exist anywhere.
    Generic(ast::NodeId),
    /// For references to self
    Self(ast::NodeId),
    /// Primitives are just the fixed-size numeric types (plus int/uint/float), and char.
    Primitive(ast::PrimTy),
    Closure(~ClosureDecl),
    /// extern "ABI" fn
    BareFunction(~BareFunctionDecl),
    Tuple(~[Type]),
    Vector(~Type),
    FixedVector(~Type, ~str),
    String,
    Bool,
    /// aka TyNil
    Unit,
    /// aka TyBot
    Bottom,
    Unique(~Type),
    Managed(~Type),
    RawPointer(Mutability, ~Type),
    BorrowedRef { lifetime: Option<Lifetime>, mutability: Mutability, type_: ~Type},
    // region, raw, other boxes, mutable
}

#[deriving(Clone, Encodable, Decodable)]
pub enum TypeKind {
    TypeStruct,
    TypeEnum,
    TypeTrait,
    TypeFunction,
}

impl Clean<Type> for ast::Ty {
    fn clean(&self) -> Type {
        use syntax::ast::*;
        debug!("cleaning type `{:?}`", self);
        let codemap = local_data::get(super::ctxtkey, |x| *x.unwrap()).sess.codemap;
        debug!("span corresponds to `{}`", codemap.span_to_str(self.span));
        match self.node {
            TyNil => Unit,
            TyPtr(ref m) => RawPointer(m.mutbl.clean(), ~m.ty.clean()),
            TyRptr(ref l, ref m) =>
                BorrowedRef {lifetime: l.clean(), mutability: m.mutbl.clean(),
                             type_: ~m.ty.clean()},
            TyBox(ty) => Managed(~ty.clean()),
            TyUniq(ty) => Unique(~ty.clean()),
            TyVec(ty) => Vector(~ty.clean()),
            TyFixedLengthVec(ty, ref e) => FixedVector(~ty.clean(),
                                                       e.span.to_src()),
            TyTup(ref tys) => Tuple(tys.iter().map(|x| x.clean()).collect()),
            TyPath(ref p, ref tpbs, id) => {
                resolve_type(p.clean(),
                             tpbs.clean().map(|x| x.move_iter().collect()),
                             id)
            }
            TyClosure(ref c) => Closure(~c.clean()),
            TyBareFn(ref barefn) => BareFunction(~barefn.clean()),
            TyBot => Bottom,
            ref x => fail!("Unimplemented type {:?}", x),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct StructField {
    type_: Type,
}

impl Clean<Item> for ast::StructField {
    fn clean(&self) -> Item {
        let (name, vis) = match self.node.kind {
            ast::NamedField(id, vis) => (Some(id), Some(vis)),
            _ => (None, None)
        };
        Item {
            name: name.clean(),
            attrs: self.node.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            visibility: vis,
            id: self.node.id,
            inner: StructFieldItem(StructField {
                type_: self.node.ty.clean(),
            }),
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
    struct_type: doctree::StructType,
    generics: Generics,
    fields: ~[Item],
    fields_stripped: bool,
}

impl Clean<Item> for doctree::Struct {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
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
    struct_type: doctree::StructType,
    fields: ~[Item],
    fields_stripped: bool,
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
    variants: ~[Item],
    generics: Generics,
    variants_stripped: bool,
}

impl Clean<Item> for doctree::Enum {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
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
    kind: VariantKind,
}

impl Clean<Item> for doctree::Variant {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            id: self.id,
            inner: VariantItem(Variant {
                kind: self.kind.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum VariantKind {
    CLikeVariant,
    TupleVariant(~[Type]),
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
    filename: ~str,
    loline: uint,
    locol: uint,
    hiline: uint,
    hicol: uint,
}

impl Clean<Span> for syntax::codemap::Span {
    fn clean(&self) -> Span {
        let cm = local_data::get(super::ctxtkey, |x| *x.unwrap()).sess.codemap;
        let filename = cm.span_to_filename(*self);
        let lo = cm.lookup_char_pos(self.lo);
        let hi = cm.lookup_char_pos(self.hi);
        Span {
            filename: filename.to_owned(),
            loline: lo.line,
            locol: lo.col.to_uint(),
            hiline: hi.line,
            hicol: hi.col.to_uint(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Path {
    global: bool,
    segments: ~[PathSegment],
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
    name: ~str,
    lifetimes: ~[Lifetime],
    types: ~[Type],
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

fn path_to_str(p: &ast::Path) -> ~str {
    use syntax::parse::token;

    let mut s = ~"";
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

impl Clean<~str> for ast::Ident {
    fn clean(&self) -> ~str {
        token::get_ident(*self).get().to_owned()
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Typedef {
    type_: Type,
    generics: Generics,
}

impl Clean<Item> for doctree::Typedef {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id.clone(),
            visibility: self.vis.clean(),
            inner: TypedefItem(Typedef {
                type_: self.ty.clean(),
                generics: self.gen.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct BareFunctionDecl {
    purity: ast::Purity,
    generics: Generics,
    decl: FnDecl,
    abi: ~str
}

impl Clean<BareFunctionDecl> for ast::BareFnTy {
    fn clean(&self) -> BareFunctionDecl {
        BareFunctionDecl {
            purity: self.purity,
            generics: Generics {
                lifetimes: self.lifetimes.clean().move_iter().collect(),
                type_params: ~[],
            },
            decl: self.decl.clean(),
            abi: self.abis.to_str(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Static {
    type_: Type,
    mutability: Mutability,
    /// It's useful to have the value of a static documented, but I have no
    /// desire to represent expressions (that'd basically be all of the AST,
    /// which is huge!). So, have a string.
    expr: ~str,
}

impl Clean<Item> for doctree::Static {
    fn clean(&self) -> Item {
        debug!("claning static {}: {:?}", self.name.clean(), self);
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
            inner: StaticItem(Static {
                type_: self.type_.clean(),
                mutability: self.mutability.clean(),
                expr: self.expr.span.to_src(),
            }),
        }
    }
}

#[deriving(Show, Clone, Encodable, Decodable)]
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
    generics: Generics,
    trait_: Option<Type>,
    for_: Type,
    methods: ~[Item],
}

impl Clean<Item> for doctree::Impl {
    fn clean(&self) -> Item {
        Item {
            name: None,
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            id: self.id,
            visibility: self.vis.clean(),
            inner: ImplItem(Impl {
                generics: self.generics.clean(),
                trait_: self.trait_.clean(),
                for_: self.for_.clean(),
                methods: self.methods.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ViewItem {
    inner: ViewItemInner
}

impl Clean<Item> for ast::ViewItem {
    fn clean(&self) -> Item {
        Item {
            name: None,
            attrs: self.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            id: 0,
            visibility: self.vis.clean(),
            inner: ViewItemItem(ViewItem {
                inner: self.node.clean()
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum ViewItemInner {
    ExternMod(~str, Option<~str>, ast::NodeId),
    Import(~[ViewPath])
}

impl Clean<ViewItemInner> for ast::ViewItem_ {
    fn clean(&self) -> ViewItemInner {
        match self {
            &ast::ViewItemExternMod(ref i, ref p, ref id) => {
                let string = match *p {
                    None => None,
                    Some((ref x, _)) => Some(x.get().to_owned()),
                };
                ExternMod(i.clean(), string, *id)
            }
            &ast::ViewItemUse(ref vp) => {
                Import(vp.clean().move_iter().collect())
            }
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub enum ViewPath {
    // use str = source;
    SimpleImport(~str, ImportSource),
    // use source::*;
    GlobImport(ImportSource),
    // use source::{a, b, c};
    ImportList(ImportSource, ~[ViewListIdent]),
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ImportSource {
    path: Path,
    did: Option<ast::DefId>,
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
    name: ~str,
    source: Option<ast::DefId>,
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
                    purity: ast::ExternFn,
                })
            }
            ast::ForeignItemStatic(ref ty, mutbl) => {
                ForeignStaticItem(Static {
                    type_: ty.clean(),
                    mutability: if mutbl {Mutable} else {Immutable},
                    expr: ~"",
                })
            }
        };
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean().move_iter().collect(),
            source: self.span.clean(),
            id: self.id,
            visibility: self.vis.clean(),
            inner: inner,
        }
    }
}

// Utilities

trait ToSource {
    fn to_src(&self) -> ~str;
}

impl ToSource for syntax::codemap::Span {
    fn to_src(&self) -> ~str {
        debug!("converting span {:?} to snippet", self.clean());
        let cm = local_data::get(super::ctxtkey, |x| x.unwrap().clone()).sess.codemap.clone();
        let sn = match cm.span_to_snippet(*self) {
            Some(x) => x,
            None    => ~""
        };
        debug!("got snippet {}", sn);
        sn
    }
}

fn lit_to_str(lit: &ast::Lit) -> ~str {
    match lit.node {
        ast::LitStr(ref st, _) => st.get().to_owned(),
        ast::LitBinary(ref data) => format!("{:?}", data.borrow().as_slice()),
        ast::LitChar(c) => ~"'" + std::char::from_u32(c).unwrap().to_str() + "'",
        ast::LitInt(i, _t) => i.to_str(),
        ast::LitUint(u, _t) => u.to_str(),
        ast::LitIntUnsuffixed(i) => i.to_str(),
        ast::LitFloat(ref f, _t) => f.get().to_str(),
        ast::LitFloatUnsuffixed(ref f) => f.get().to_str(),
        ast::LitBool(b) => b.to_str(),
        ast::LitNil => ~"",
    }
}

fn name_from_pat(p: &ast::Pat) -> ~str {
    use syntax::ast::*;
    debug!("Trying to get a name from pattern: {:?}", p);

    match p.node {
        PatWild => ~"_",
        PatWildMulti => ~"..",
        PatIdent(_, ref p, _) => path_to_str(p),
        PatEnum(ref p, _) => path_to_str(p),
        PatStruct(..) => fail!("tried to get argument name from pat_struct, \
                                which is not allowed in function arguments"),
        PatTup(..) => ~"(tuple arg NYI)",
        PatUniq(p) => name_from_pat(p),
        PatRegion(p) => name_from_pat(p),
        PatLit(..) => {
            warn!("tried to get argument name from PatLit, \
                  which is silly in function arguments");
            ~"()"
        },
        PatRange(..) => fail!("tried to get argument name from PatRange, \
                              which is not allowed in function arguments"),
        PatVec(..) => fail!("tried to get argument name from pat_vec, \
                             which is not allowed in function arguments")
    }
}

/// Given a Type, resolve it using the def_map
fn resolve_type(path: Path, tpbs: Option<~[TyParamBound]>,
                id: ast::NodeId) -> Type {
    let cx = local_data::get(super::ctxtkey, |x| *x.unwrap());
    let tycx = match cx.tycx {
        Some(tycx) => tycx,
        // If we're extracting tests, this return value doesn't matter.
        None => return Bool
    };
    debug!("searching for {:?} in defmap", id);
    let def_map = tycx.def_map.borrow();
    let d = match def_map.get().find(&id) {
        Some(k) => k,
        None => {
            debug!("could not find {:?} in defmap (`{}`)", id, tycx.map.node_to_str(id));
            fail!("Unexpected failure: unresolved id not in defmap (this is a bug!)")
        }
    };

    let (def_id, kind) = match *d {
        ast::DefFn(i, _) => (i, TypeFunction),
        ast::DefSelfTy(i) => return Self(i),
        ast::DefTy(i) => (i, TypeEnum),
        ast::DefTrait(i) => {
            debug!("saw DefTrait in def_to_id");
            (i, TypeTrait)
        },
        ast::DefPrimTy(p) => match p {
            ast::TyStr => return String,
            ast::TyBool => return Bool,
            _ => return Primitive(p)
        },
        ast::DefTyParam(i, _) => return Generic(i.node),
        ast::DefStruct(i) => (i, TypeStruct),
        ast::DefTyParamBinder(i) => {
            debug!("found a typaram_binder, what is it? {}", i);
            return TyParamBinder(i);
        },
        x => fail!("resolved type maps to a weird def {:?}", x),
    };
    if ast_util::is_local(def_id) {
        ResolvedPath{ path: path, typarams: tpbs, id: def_id.node }
    } else {
        let fqn = csearch::get_item_path(tycx, def_id);
        let fqn = fqn.move_iter().map(|i| i.to_str()).to_owned_vec();
        ExternalPath{ path: path, typarams: tpbs, fqn: fqn, kind: kind,
                      krate: def_id.krate }
    }
}

fn resolve_use_source(path: Path, id: ast::NodeId) -> ImportSource {
    ImportSource {
        path: path,
        did: resolve_def(id),
    }
}

fn resolve_def(id: ast::NodeId) -> Option<ast::DefId> {
    let cx = local_data::get(super::ctxtkey, |x| *x.unwrap());
    match cx.tycx {
        Some(tcx) => {
            let def_map = tcx.def_map.borrow();
            def_map.get().find(&id).map(|&d| ast_util::def_id_of_def(d))
        }
        None => None
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Macro {
    source: ~str,
}

impl Clean<Item> for doctree::Macro {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.name.clean()),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: ast::Public.clean(),
            id: self.id,
            inner: MacroItem(Macro {
                source: self.where.to_src(),
            }),
        }
    }
}
