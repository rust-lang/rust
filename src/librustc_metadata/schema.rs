// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use astencode;
use index;

use rustc::hir;
use rustc::hir::def;
use rustc::hir::def_id::{DefIndex, DefId};
use rustc::middle::cstore::{LinkagePreference, NativeLibraryKind};
use rustc::middle::lang_items;
use rustc::mir;
use rustc::ty::{self, Ty};
use rustc::session::config::PanicStrategy;

use rustc_serialize as serialize;
use syntax::{ast, attr};
use syntax_pos::{self, Span};

use std::marker::PhantomData;

pub const RUSTC_VERSION: &'static str = concat!("rustc ", env!("CFG_VERSION"));

/// Metadata encoding version.
/// NB: increment this if you change the format of metadata such that
/// the rustc version can't be found to compare with `RUSTC_VERSION`.
pub const METADATA_VERSION: u8 = 3;

/// Metadata header which includes `METADATA_VERSION`.
/// To get older versions of rustc to ignore this metadata,
/// there are 4 zero bytes at the start, which are treated
/// as a length of 0 by old compilers.
///
/// This header is followed by the position of the `CrateRoot`.
pub const METADATA_HEADER: &'static [u8; 12] = &[
    0, 0, 0, 0,
    b'r', b'u', b's', b't',
    0, 0, 0, METADATA_VERSION
];

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;

/// A value of type T referred to by its absolute position
/// in the metadata, and which can be decoded lazily.
#[must_use]
pub struct Lazy<T> {
    pub position: usize,
    _marker: PhantomData<T>
}

impl<T> Lazy<T> {
    pub fn with_position(position: usize) -> Lazy<T> {
        Lazy {
            position: position,
            _marker: PhantomData
        }
    }
}

impl<T> Copy for Lazy<T> {}
impl<T> Clone for Lazy<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> serialize::UseSpecializedEncodable for Lazy<T> {}
impl<T> serialize::UseSpecializedDecodable for Lazy<T> {}

/// A sequence of type T referred to by its absolute position
/// in the metadata and length, and which can be decoded lazily.
///
/// Unlike `Lazy<Vec<T>>`, the length is encoded next to the
/// position, not at the position, which means that the length
/// doesn't need to be known before encoding all the elements.
#[must_use]
pub struct LazySeq<T> {
    pub len: usize,
    pub position: usize,
    _marker: PhantomData<T>
}

impl<T> LazySeq<T> {
    pub fn empty() -> LazySeq<T> {
        LazySeq::with_position_and_length(0, 0)
    }

    pub fn with_position_and_length(position: usize, len: usize) -> LazySeq<T> {
        LazySeq {
            len: len,
            position: position,
            _marker: PhantomData
        }
    }
}

impl<T> Copy for LazySeq<T> {}
impl<T> Clone for LazySeq<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> serialize::UseSpecializedEncodable for LazySeq<T> {}
impl<T> serialize::UseSpecializedDecodable for LazySeq<T> {}

#[derive(RustcEncodable, RustcDecodable)]
pub struct CrateRoot {
    pub rustc_version: String,
    pub name: String,
    pub triple: String,
    pub hash: hir::svh::Svh,
    pub disambiguator: String,
    pub panic_strategy: PanicStrategy,
    pub plugin_registrar_fn: Option<DefIndex>,
    pub macro_derive_registrar: Option<DefIndex>,

    pub index: LazySeq<index::Index>,
    pub crate_deps: LazySeq<CrateDep>,
    pub dylib_dependency_formats: LazySeq<Option<LinkagePreference>>,
    pub native_libraries: LazySeq<(NativeLibraryKind, String)>,
    pub lang_items: LazySeq<(DefIndex, usize)>,
    pub lang_items_missing: LazySeq<lang_items::LangItem>,
    pub impls: LazySeq<TraitImpls>,
    pub reachable_ids: LazySeq<DefIndex>,
    pub macro_defs: LazySeq<MacroDef>,
    pub codemap: LazySeq<syntax_pos::FileMap>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct CrateDep {
    pub name: ast::Name,
    pub hash: hir::svh::Svh,
    pub explicitly_linked: bool
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct TraitImpls {
    pub trait_id: (u32, DefIndex),
    pub impls: LazySeq<DefIndex>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct MacroDef {
    pub name: ast::Name,
    pub attrs: Vec<ast::Attribute>,
    pub span: Span,
    pub body: String
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct Entry<'tcx> {
    pub kind: EntryKind<'tcx>,
    pub visibility: ty::Visibility,
    pub def_key: Lazy<hir::map::DefKey>,
    pub attributes: LazySeq<ast::Attribute>,
    pub children: LazySeq<DefIndex>,
    pub stability: Option<Lazy<attr::Stability>>,
    pub deprecation: Option<Lazy<attr::Deprecation>>,

    pub ty: Option<Lazy<Ty<'tcx>>>,
    pub inherent_impls: LazySeq<DefIndex>,
    pub variances: LazySeq<ty::Variance>,
    pub generics: Option<Lazy<ty::Generics<'tcx>>>,
    pub predicates: Option<Lazy<ty::GenericPredicates<'tcx>>>,

    pub ast: Option<Lazy<astencode::Ast<'tcx>>>,
    pub mir: Option<Lazy<mir::repr::Mir<'tcx>>>
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum EntryKind<'tcx> {
    Const,
    ImmStatic,
    MutStatic,
    ForeignImmStatic,
    ForeignMutStatic,
    ForeignMod,
    Type,
    Enum,
    Field,
    Variant(Lazy<VariantData>),
    Struct(Lazy<VariantData>),
    Union(Lazy<VariantData>),
    Fn(Lazy<FnData>),
    ForeignFn(Lazy<FnData>),
    Mod(Lazy<ModData>),
    Closure(Lazy<ClosureData<'tcx>>),
    Trait(Lazy<TraitData<'tcx>>),
    Impl(Lazy<ImplData<'tcx>>),
    DefaultImpl(Lazy<ImplData<'tcx>>),
    Method(Lazy<MethodData<'tcx>>),
    AssociatedType(AssociatedContainer),
    AssociatedConst(AssociatedContainer)
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ModData {
    pub reexports: LazySeq<def::Export>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct FnData {
    pub constness: hir::Constness,
    pub arg_names: LazySeq<ast::Name>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct VariantData {
    pub kind: ty::VariantKind,
    pub disr: u64,

    /// If this is a struct's only variant, this
    /// is the index of the "struct ctor" item.
    pub struct_ctor: Option<DefIndex>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct TraitData<'tcx> {
    pub unsafety: hir::Unsafety,
    pub paren_sugar: bool,
    pub has_default_impl: bool,
    pub trait_ref: Lazy<ty::TraitRef<'tcx>>,
    pub super_predicates: Lazy<ty::GenericPredicates<'tcx>>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ImplData<'tcx> {
    pub polarity: hir::ImplPolarity,
    pub parent_impl: Option<DefId>,
    pub coerce_unsized_kind: Option<ty::adjustment::CustomCoerceUnsized>,
    pub trait_ref: Option<Lazy<ty::TraitRef<'tcx>>>
}

/// Describes whether the container of an associated item
/// is a trait or an impl and whether, in a trait, it has
/// a default, or an in impl, whether it's marked "default".
#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum AssociatedContainer {
    TraitRequired,
    TraitWithDefault,
    ImplDefault,
    ImplFinal
}

impl AssociatedContainer {
    pub fn with_def_id(&self, def_id: DefId) -> ty::ImplOrTraitItemContainer {
        match *self {
            AssociatedContainer::TraitRequired |
            AssociatedContainer::TraitWithDefault => {
                ty::TraitContainer(def_id)
            }

            AssociatedContainer::ImplDefault |
            AssociatedContainer::ImplFinal => {
                ty::ImplContainer(def_id)
            }
        }
    }

    pub fn has_body(&self) -> bool {
        match *self {
            AssociatedContainer::TraitRequired => false,

            AssociatedContainer::TraitWithDefault |
            AssociatedContainer::ImplDefault |
            AssociatedContainer::ImplFinal => true
        }
    }

    pub fn defaultness(&self) -> hir::Defaultness {
        match *self {
            AssociatedContainer::TraitRequired |
            AssociatedContainer::TraitWithDefault |
            AssociatedContainer::ImplDefault => hir::Defaultness::Default,

            AssociatedContainer::ImplFinal => hir::Defaultness::Final
        }
    }
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct MethodData<'tcx> {
    pub fn_data: FnData,
    pub container: AssociatedContainer,
    pub explicit_self: Lazy<ty::ExplicitSelfCategory<'tcx>>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ClosureData<'tcx> {
    pub kind: ty::ClosureKind,
    pub ty: Lazy<ty::ClosureTy<'tcx>>
}
