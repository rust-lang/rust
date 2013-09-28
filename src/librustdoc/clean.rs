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

use its = syntax::parse::token::ident_to_str;

use syntax;
use syntax::ast;
use syntax::ast_util;
use syntax::attr;
use syntax::attr::AttributeMethods;

use std;
use doctree;
use visit_ast;
use std::local_data;

pub trait Clean<T> {
    fn clean(&self) -> T;
}

impl<T: Clean<U>, U> Clean<~[U]> for ~[T] {
    fn clean(&self) -> ~[U] {
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

impl<T: Clean<U>, U> Clean<~[U]> for syntax::opt_vec::OptVec<T> {
    fn clean(&self) -> ~[U] {
        match self {
            &syntax::opt_vec::Empty => ~[],
            &syntax::opt_vec::Vec(ref v) => v.clean()
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Crate {
    name: ~str,
    module: Option<Item>,
}

impl Clean<Crate> for visit_ast::RustdocVisitor {
    fn clean(&self) -> Crate {
        use syntax::attr::{find_linkage_metas, last_meta_item_value_str_by_name};
        let maybe_meta = last_meta_item_value_str_by_name(find_linkage_metas(self.attrs), "name");

        Crate {
            name: match maybe_meta {
                Some(x) => x.to_owned(),
                None => fail!("rustdoc_ng requires a #[link(name=\"foo\")] crate attribute"),
            },
            module: Some(self.module.clean()),
        }
    }
}

/// Anything with a source location and set of attributes and, optionally, a
/// name. That is, anything that can be documented. This doesn't correspond
/// directly to the AST's concept of an item; it's a strict superset.
#[deriving(Clone, Encodable, Decodable)]
pub struct Item {
    /// Stringified span
    source: ~str,
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
        match self.inner { ModuleItem(*) => true, _ => false }
    }
    pub fn is_trait(&self) -> bool {
        match self.inner { TraitItem(*) => true, _ => false }
    }
    pub fn is_struct(&self) -> bool {
        match self.inner { StructItem(*) => true, _ => false }
    }
    pub fn is_enum(&self) -> bool {
        match self.inner { EnumItem(*) => true, _ => false }
    }
    pub fn is_fn(&self) -> bool {
        match self.inner { FunctionItem(*) => true, _ => false }
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
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Module {
    items: ~[Item],
}

impl Clean<Item> for doctree::Module {
    fn clean(&self) -> Item {
        let name = if self.name.is_some() {
            self.name.unwrap().clean()
        } else {
            ~""
        };
        Item {
            name: Some(name),
            attrs: self.attrs.clean(),
            source: self.where.clean(),
            visibility: self.vis.clean(),
            id: self.id,
            inner: ModuleItem(Module {
               items: [self.structs.clean(), self.enums.clean(),
                       self.fns.clean(), self.foreigns.clean().concat_vec(),
                       self.mods.clean(), self.typedefs.clean(),
                       self.statics.clean(), self.traits.clean(),
                       self.impls.clean(), self.view_items.clean()].concat_vec()
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
            ast::MetaWord(s) => Word(s.to_owned()),
            ast::MetaList(ref s, ref l) => List(s.to_owned(), l.clean()),
            ast::MetaNameValue(s, ref v) => NameValue(s.to_owned(), lit_to_str(v))
        }
    }
}

impl Clean<Attribute> for ast::Attribute {
    fn clean(&self) -> Attribute {
        self.desugar_doc().node.value.clean()
    }
}

// This is a rough approximation that gets us what we want.
impl<'self> attr::AttrMetaMethods for &'self Attribute {
    fn name(&self) -> @str {
        match **self {
            Word(ref n) | List(ref n, _) | NameValue(ref n, _) =>
                n.to_managed()
        }
    }

    fn value_str(&self) -> Option<@str> {
        match **self {
            NameValue(_, ref v) => Some(v.to_managed()),
            _ => None,
        }
    }
    fn meta_item_list<'a>(&'a self) -> Option<&'a [@ast::MetaItem]> { None }
    fn name_str_pair(&self) -> Option<(@str, @str)> { None }
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
            bounds: self.bounds.clean(),
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

impl Clean<Lifetime> for ast::Lifetime {
    fn clean(&self) -> Lifetime {
        Lifetime(self.ident.clean())
    }
}

// maybe use a Generic enum and use ~[Generic]?
#[deriving(Clone, Encodable, Decodable)]
pub struct Generics {
    lifetimes: ~[Lifetime],
    type_params: ~[TyParam]
}

impl Generics {
    fn new() -> Generics {
        Generics {
            lifetimes: ~[],
            type_params: ~[]
        }
    }
}

impl Clean<Generics> for ast::Generics {
    fn clean(&self) -> Generics {
        Generics {
            lifetimes: self.lifetimes.clean(),
            type_params: self.ty_params.clean(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Method {
    generics: Generics,
    self_: SelfTy,
    purity: ast::purity,
    decl: FnDecl,
}

impl Clean<Item> for ast::method {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean(),
            source: self.span.clean(),
            id: self.self_id.clone(),
            visibility: self.vis.clean(),
            inner: MethodItem(Method {
                generics: self.generics.clean(),
                self_: self.explicit_self.clean(),
                purity: self.purity.clone(),
                decl: self.decl.clean(),
            }),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct TyMethod {
    purity: ast::purity,
    decl: FnDecl,
    generics: Generics,
    self_: SelfTy,
}

impl Clean<Item> for ast::TypeMethod {
    fn clean(&self) -> Item {
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean(),
            source: self.span.clean(),
            id: self.id,
            visibility: None,
            inner: TyMethodItem(TyMethod {
                purity: self.purity.clone(),
                decl: self.decl.clean(),
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
    SelfManaged(Mutability),
    SelfOwned,
}

impl Clean<SelfTy> for ast::explicit_self {
    fn clean(&self) -> SelfTy {
        match self.node {
            ast::sty_static => SelfStatic,
            ast::sty_value => SelfValue,
            ast::sty_uniq => SelfOwned,
            ast::sty_region(lt, mt) => SelfBorrowed(lt.clean(), mt.clean()),
            ast::sty_box(mt) => SelfManaged(mt.clean()),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Function {
    decl: FnDecl,
    generics: Generics,
    purity: ast::purity,
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
    purity: ast::purity,
    bounds: ~[TyParamBound]
}

impl Clean<ClosureDecl> for ast::TyClosure {
    fn clean(&self) -> ClosureDecl {
        ClosureDecl {
            sigil: self.sigil,
            region: self.region.clean(),
            lifetimes: self.lifetimes.clean(),
            decl: self.decl.clean(),
            onceness: self.onceness,
            purity: self.purity,
            bounds: match self.bounds {
                Some(ref x) => x.clean(),
                None        => ~[]
            },
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct FnDecl {
    inputs: ~[Argument],
    output: Type,
    cf: RetStyle,
    attrs: ~[Attribute]
}

impl Clean<FnDecl> for ast::fn_decl {
    fn clean(&self) -> FnDecl {
        FnDecl {
            inputs: self.inputs.iter().map(|x| x.clean()).collect(),
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

impl Clean<Argument> for ast::arg {
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

impl Clean<RetStyle> for ast::ret_style {
    fn clean(&self) -> RetStyle {
        match *self {
            ast::return_val => Return,
            ast::noreturn => NoReturn
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

impl Clean<Type> for ast::trait_ref {
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
            &Required(*) => true,
            _ => false,
        }
    }
    pub fn is_def(&self) -> bool {
        match self {
            &Provided(*) => true,
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

impl Clean<TraitMethod> for ast::trait_method {
    fn clean(&self) -> TraitMethod {
        match self {
            &ast::required(ref t) => Required(t.clean()),
            &ast::provided(ref t) => Provided(t.clean()),
        }
    }
}

/// A representation of a Type suitable for hyperlinking purposes. Ideally one can get the original
/// type out of the AST/ty::ctxt given one of these, if more information is needed. Most importantly
/// it does not preserve mutability or boxes.
#[deriving(Clone, Encodable, Decodable)]
pub enum Type {
    /// structs/enums/traits (anything that'd be an ast::ty_path)
    ResolvedPath { path: Path, typarams: Option<~[TyParamBound]>, id: ast::NodeId },
    /// Reference to an item in an external crate (fully qualified path)
    External(~str, ~str),
    // I have no idea how to usefully use this.
    TyParamBinder(ast::NodeId),
    /// For parameterized types, so the consumer of the JSON don't go looking
    /// for types which don't exist anywhere.
    Generic(ast::NodeId),
    /// For references to self
    Self(ast::NodeId),
    /// Primitives are just the fixed-size numeric types (plus int/uint/float), and char.
    Primitive(ast::prim_ty),
    Closure(~ClosureDecl),
    /// extern "ABI" fn
    BareFunction(~BareFunctionDecl),
    Tuple(~[Type]),
    Vector(~Type),
    FixedVector(~Type, ~str),
    String,
    Bool,
    /// aka ty_nil
    Unit,
    /// aka ty_bot
    Bottom,
    Unique(~Type),
    Managed(Mutability, ~Type),
    RawPointer(Mutability, ~Type),
    BorrowedRef { lifetime: Option<Lifetime>, mutability: Mutability, type_: ~Type},
    // region, raw, other boxes, mutable
}

impl Clean<Type> for ast::Ty {
    fn clean(&self) -> Type {
        use syntax::ast::*;
        debug!("cleaning type `%?`", self);
        let codemap = local_data::get(super::ctxtkey, |x| *x.unwrap()).sess.codemap;
        debug!("span corresponds to `%s`", codemap.span_to_str(self.span));
        match self.node {
            ty_nil => Unit,
            ty_ptr(ref m) => RawPointer(m.mutbl.clean(), ~m.ty.clean()),
            ty_rptr(ref l, ref m) =>
                BorrowedRef {lifetime: l.clean(), mutability: m.mutbl.clean(),
                             type_: ~m.ty.clean()},
            ty_box(ref m) => Managed(m.mutbl.clean(), ~m.ty.clean()),
            ty_uniq(ref m) => Unique(~m.ty.clean()),
            ty_vec(ref m) => Vector(~m.ty.clean()),
            ty_fixed_length_vec(ref m, ref e) => FixedVector(~m.ty.clean(),
                                                             e.span.to_src()),
            ty_tup(ref tys) => Tuple(tys.iter().map(|x| x.clean()).collect()),
            ty_path(ref p, ref tpbs, id) =>
                resolve_type(p.clean(), tpbs.clean(), id),
            ty_closure(ref c) => Closure(~c.clean()),
            ty_bare_fn(ref barefn) => BareFunction(~barefn.clean()),
            ty_bot => Bottom,
            ref x => fail!("Unimplemented type %?", x),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct StructField {
    type_: Type,
}

impl Clean<Item> for ast::struct_field {
    fn clean(&self) -> Item {
        let (name, vis) = match self.node.kind {
            ast::named_field(id, vis) => (Some(id), Some(vis)),
            _ => (None, None)
        };
        Item {
            name: name.clean(),
            attrs: self.node.attrs.clean(),
            source: self.span.clean(),
            visibility: vis,
            id: self.node.id,
            inner: StructFieldItem(StructField {
                type_: self.node.ty.clean(),
            }),
        }
    }
}

pub type Visibility = ast::visibility;

impl Clean<Option<Visibility>> for ast::visibility {
    fn clean(&self) -> Option<Visibility> {
        Some(*self)
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Struct {
    struct_type: doctree::StructType,
    generics: Generics,
    fields: ~[Item],
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
            }),
        }
    }
}

/// This is a more limited form of the standard Struct, different in that it
/// it lacks the things most items have (name, id, parameterization). Found
/// only as a variant in an enum.
#[deriving(Clone, Encodable, Decodable)]
pub struct VariantStruct {
    struct_type: doctree::StructType,
    fields: ~[Item],
}

impl Clean<VariantStruct> for syntax::ast::struct_def {
    fn clean(&self) -> VariantStruct {
        VariantStruct {
            struct_type: doctree::struct_type_from_def(self),
            fields: self.fields.clean(),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct Enum {
    variants: ~[Item],
    generics: Generics,
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

impl Clean<VariantKind> for ast::variant_kind {
    fn clean(&self) -> VariantKind {
        match self {
            &ast::tuple_variant_kind(ref args) => {
                if args.len() == 0 {
                    CLikeVariant
                } else {
                    TupleVariant(args.iter().map(|x| x.ty.clean()).collect())
                }
            },
            &ast::struct_variant_kind(ref sd) => StructVariant(sd.clean()),
        }
    }
}

impl Clean<~str> for syntax::codemap::Span {
    fn clean(&self) -> ~str {
        let cm = local_data::get(super::ctxtkey, |x| x.unwrap().clone()).sess.codemap;
        cm.span_to_str(*self)
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
            segments: self.segments.clean()
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct PathSegment {
    name: ~str,
    lifetime: Option<Lifetime>,
    types: ~[Type],
}

impl Clean<PathSegment> for ast::PathSegment {
    fn clean(&self) -> PathSegment {
        PathSegment {
            name: self.identifier.clean(),
            lifetime: self.lifetime.clean(),
            types: self.types.clean()
        }
    }
}

fn path_to_str(p: &ast::Path) -> ~str {
    use syntax::parse::token::interner_get;

    let mut s = ~"";
    let mut first = true;
    for i in p.segments.iter().map(|x| interner_get(x.identifier.name)) {
        if !first || p.global {
            s.push_str("::");
        } else {
            first = false;
        }
        s.push_str(i);
    }
    s
}

impl Clean<~str> for ast::Ident {
    fn clean(&self) -> ~str {
        its(self).to_owned()
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
    purity: ast::purity,
    generics: Generics,
    decl: FnDecl,
    abi: ~str
}

impl Clean<BareFunctionDecl> for ast::TyBareFn {
    fn clean(&self) -> BareFunctionDecl {
        BareFunctionDecl {
            purity: self.purity,
            generics: Generics {
                lifetimes: self.lifetimes.clean(),
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
        debug!("claning static %s: %?", self.name.clean(), self);
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

#[deriving(ToStr, Clone, Encodable, Decodable)]
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

impl Clean<Item> for ast::view_item {
    fn clean(&self) -> Item {
        Item {
            name: None,
            attrs: self.attrs.clean(),
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
    ExternMod(~str, Option<~str>, ~[Attribute], ast::NodeId),
    Import(~[ViewPath])
}

impl Clean<ViewItemInner> for ast::view_item_ {
    fn clean(&self) -> ViewItemInner {
        match self {
            &ast::view_item_extern_mod(ref i, ref p, ref mi, ref id) =>
                ExternMod(i.clean(), p.map(|x| x.to_owned()),  mi.clean(), *id),
            &ast::view_item_use(ref vp) => Import(vp.clean())
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

impl Clean<ViewPath> for ast::view_path {
    fn clean(&self) -> ViewPath {
        match self.node {
            ast::view_path_simple(ref i, ref p, id) =>
                SimpleImport(i.clean(), resolve_use_source(p.clean(), id)),
            ast::view_path_glob(ref p, id) =>
                GlobImport(resolve_use_source(p.clean(), id)),
            ast::view_path_list(ref p, ref pl, id) =>
                ImportList(resolve_use_source(p.clean(), id), pl.clean()),
        }
    }
}

#[deriving(Clone, Encodable, Decodable)]
pub struct ViewListIdent {
    name: ~str,
    source: Option<ast::DefId>,
}

impl Clean<ViewListIdent> for ast::path_list_ident {
    fn clean(&self) -> ViewListIdent {
        ViewListIdent {
            name: self.node.name.clean(),
            source: resolve_def(self.node.id),
        }
    }
}

impl Clean<~[Item]> for ast::foreign_mod {
    fn clean(&self) -> ~[Item] {
        self.items.clean()
    }
}

impl Clean<Item> for ast::foreign_item {
    fn clean(&self) -> Item {
        let inner = match self.node {
            ast::foreign_item_fn(ref decl, ref generics) => {
                ForeignFunctionItem(Function {
                    decl: decl.clean(),
                    generics: generics.clean(),
                    purity: ast::extern_fn,
                })
            }
            ast::foreign_item_static(ref ty, mutbl) => {
                ForeignStaticItem(Static {
                    type_: ty.clean(),
                    mutability: if mutbl {Mutable} else {Immutable},
                    expr: ~"",
                })
            }
        };
        Item {
            name: Some(self.ident.clean()),
            attrs: self.attrs.clean(),
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
        debug!("converting span %s to snippet", self.clean());
        let cm = local_data::get(super::ctxtkey, |x| x.unwrap().clone()).sess.codemap.clone();
        let sn = match cm.span_to_snippet(*self) {
            Some(x) => x,
            None    => ~""
        };
        debug!("got snippet %s", sn);
        sn
    }
}

fn lit_to_str(lit: &ast::lit) -> ~str {
    match lit.node {
        ast::lit_str(st) => st.to_owned(),
        ast::lit_char(c) => ~"'" + std::char::from_u32(c).unwrap().to_str() + "'",
        ast::lit_int(i, _t) => i.to_str(),
        ast::lit_uint(u, _t) => u.to_str(),
        ast::lit_int_unsuffixed(i) => i.to_str(),
        ast::lit_float(f, _t) => f.to_str(),
        ast::lit_float_unsuffixed(f) => f.to_str(),
        ast::lit_bool(b) => b.to_str(),
        ast::lit_nil => ~"",
    }
}

fn name_from_pat(p: &ast::Pat) -> ~str {
    use syntax::ast::*;
    match p.node {
        PatWild => ~"_",
        PatIdent(_, ref p, _) => path_to_str(p),
        PatEnum(ref p, _) => path_to_str(p),
        PatStruct(*) => fail!("tried to get argument name from pat_struct, \
                                which is not allowed in function arguments"),
        PatTup(*) => ~"(tuple arg NYI)",
        PatBox(p) => name_from_pat(p),
        PatUniq(p) => name_from_pat(p),
        PatRegion(p) => name_from_pat(p),
        PatLit(*) => fail!("tried to get argument name from pat_lit, \
                            which is not allowed in function arguments"),
        PatRange(*) => fail!("tried to get argument name from pat_range, \
                              which is not allowed in function arguments"),
        PatVec(*) => fail!("tried to get argument name from pat_vec, \
                             which is not allowed in function arguments")
    }
}

fn remove_comment_tags(s: &str) -> ~str {
    if s.starts_with("/") {
        match s.slice(0,3) {
            &"///" => return s.slice(3, s.len()).trim().to_owned(),
            &"/**" | &"/*!" => return s.slice(3, s.len() - 2).trim().to_owned(),
            _ => return s.trim().to_owned()
        }
    } else {
        return s.to_owned();
    }
}

/// Given a Type, resolve it using the def_map
fn resolve_type(path: Path, tpbs: Option<~[TyParamBound]>,
                id: ast::NodeId) -> Type {
    use syntax::ast::*;

    let dm = local_data::get(super::ctxtkey, |x| *x.unwrap()).tycx.def_map;
    debug!("searching for %? in defmap", id);
    let d = match dm.find(&id) {
        Some(k) => k,
        None => {
            let ctxt = local_data::get(super::ctxtkey, |x| *x.unwrap());
            debug!("could not find %? in defmap (`%s`)", id,
                   syntax::ast_map::node_id_to_str(ctxt.tycx.items, id, ctxt.sess.intr()));
            fail!("Unexpected failure: unresolved id not in defmap (this is a bug!)")
        }
    };

    let def_id = match *d {
        DefFn(i, _) => i,
        DefSelf(i) | DefSelfTy(i) => return Self(i),
        DefTy(i) => i,
        DefTrait(i) => {
            debug!("saw DefTrait in def_to_id");
            i
        },
        DefPrimTy(p) => match p {
            ty_str => return String,
            ty_bool => return Bool,
            _ => return Primitive(p)
        },
        DefTyParam(i, _) => return Generic(i.node),
        DefStruct(i) => i,
        DefTyParamBinder(i) => {
            debug!("found a typaram_binder, what is it? %d", i);
            return TyParamBinder(i);
        },
        x => fail!("resolved type maps to a weird def %?", x),
    };

    if def_id.crate != ast::CRATE_NODE_ID {
        use rustc::metadata::decoder::*;

        let sess = local_data::get(super::ctxtkey, |x| *x.unwrap()).sess;
        let cratedata = ::rustc::metadata::cstore::get_crate_data(sess.cstore, def_id.crate);
        let doc = lookup_item(def_id.node, cratedata.data);
        let path = syntax::ast_map::path_to_str_with_sep(item_path(doc), "::", sess.intr());
        let ty = match def_like_to_def(item_to_def_like(doc, def_id, def_id.crate)) {
            DefFn(*) => ~"fn",
            DefTy(*) => ~"enum",
            DefTrait(*) => ~"trait",
            DefPrimTy(p) => match p {
                ty_str => ~"str",
                ty_bool => ~"bool",
                ty_int(t) => match t.to_str() {
                    ~"" => ~"i",
                    s => s
                },
                ty_uint(t) => t.to_str(),
                ty_float(t) => t.to_str(),
                ty_char => ~"char",
            },
            DefTyParam(*) => ~"generic",
            DefStruct(*) => ~"struct",
            DefTyParamBinder(*) => ~"typaram_binder",
            x => fail!("resolved external maps to a weird def %?", x),
        };
        let cname = cratedata.name.to_owned();
        External(cname + "::" + path, ty)
    } else {
        ResolvedPath {path: path.clone(), typarams: tpbs, id: def_id.node}
    }
}

fn resolve_use_source(path: Path, id: ast::NodeId) -> ImportSource {
    ImportSource {
        path: path,
        did: resolve_def(id),
    }
}

fn resolve_def(id: ast::NodeId) -> Option<ast::DefId> {
    let dm = local_data::get(super::ctxtkey, |x| *x.unwrap()).tycx.def_map;
    dm.find(&id).map_move(|&d| ast_util::def_id_of_def(d))
}
