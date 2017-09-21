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
use rustc::hir::def::{self, CtorKind};
use rustc::hir::def_id::{DefIndex, DefId, CrateNum};
use rustc::ich::StableHashingContext;
use rustc::middle::cstore::{DepKind, LinkagePreference, NativeLibrary};
use rustc::middle::lang_items;
use rustc::mir;
use rustc::ty::{self, Ty, ReprOptions};
use rustc_back::PanicStrategy;

use rustc_serialize as serialize;
use syntax::{ast, attr};
use syntax::symbol::Symbol;
use syntax_pos::{self, Span};

use std::marker::PhantomData;
use std::mem;

use rustc_data_structures::stable_hasher::{StableHasher, HashStable,
                                           StableHasherResult};

pub fn rustc_version() -> String {
    format!("rustc {}",
            option_env!("CFG_VERSION").unwrap_or("unknown version"))
}

/// Metadata encoding version.
/// NB: increment this if you change the format of metadata such that
/// the rustc version can't be found to compare with `rustc_version()`.
pub const METADATA_VERSION: u8 = 4;

/// Metadata header which includes `METADATA_VERSION`.
/// To get older versions of rustc to ignore this metadata,
/// there are 4 zero bytes at the start, which are treated
/// as a length of 0 by old compilers.
///
/// This header is followed by the position of the `CrateRoot`,
/// which is encoded as a 32-bit big-endian unsigned integer,
/// and further followed by the rustc version string.
pub const METADATA_HEADER: &'static [u8; 12] =
    &[0, 0, 0, 0, b'r', b'u', b's', b't', 0, 0, 0, METADATA_VERSION];

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;

/// A value of type T referred to by its absolute position
/// in the metadata, and which can be decoded lazily.
///
/// Metadata is effective a tree, encoded in post-order,
/// and with the root's position written next to the header.
/// That means every single `Lazy` points to some previous
/// location in the metadata and is part of a larger node.
///
/// The first `Lazy` in a node is encoded as the backwards
/// distance from the position where the containing node
/// starts and where the `Lazy` points to, while the rest
/// use the forward distance from the previous `Lazy`.
/// Distances start at 1, as 0-byte nodes are invalid.
/// Also invalid are nodes being referred in a different
/// order than they were encoded in.
#[must_use]
pub struct Lazy<T> {
    pub position: usize,
    _marker: PhantomData<T>,
}

impl<T> Lazy<T> {
    pub fn with_position(position: usize) -> Lazy<T> {
        Lazy {
            position,
            _marker: PhantomData,
        }
    }

    /// Returns the minimum encoded size of a value of type `T`.
    // FIXME(eddyb) Give better estimates for certain types.
    pub fn min_size() -> usize {
        1
    }
}

impl<T> Copy for Lazy<T> {}
impl<T> Clone for Lazy<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> serialize::UseSpecializedEncodable for Lazy<T> {}
impl<T> serialize::UseSpecializedDecodable for Lazy<T> {}

impl<CTX, T> HashStable<CTX> for Lazy<T> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut CTX,
                                          _: &mut StableHasher<W>) {
        // There's nothing to do. Whatever got encoded within this Lazy<>
        // wrapper has already been hashed.
    }
}

/// A sequence of type T referred to by its absolute position
/// in the metadata and length, and which can be decoded lazily.
/// The sequence is a single node for the purposes of `Lazy`.
///
/// Unlike `Lazy<Vec<T>>`, the length is encoded next to the
/// position, not at the position, which means that the length
/// doesn't need to be known before encoding all the elements.
///
/// If the length is 0, no position is encoded, but otherwise,
/// the encoding is that of `Lazy`, with the distinction that
/// the minimal distance the length of the sequence, i.e.
/// it's assumed there's no 0-byte element in the sequence.
#[must_use]
pub struct LazySeq<T> {
    pub len: usize,
    pub position: usize,
    _marker: PhantomData<T>,
}

impl<T> LazySeq<T> {
    pub fn empty() -> LazySeq<T> {
        LazySeq::with_position_and_length(0, 0)
    }

    pub fn with_position_and_length(position: usize, len: usize) -> LazySeq<T> {
        LazySeq {
            len,
            position,
            _marker: PhantomData,
        }
    }

    /// Returns the minimum encoded size of `length` values of type `T`.
    pub fn min_size(length: usize) -> usize {
        length
    }
}

impl<T> Copy for LazySeq<T> {}
impl<T> Clone for LazySeq<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> serialize::UseSpecializedEncodable for LazySeq<T> {}
impl<T> serialize::UseSpecializedDecodable for LazySeq<T> {}

impl<CTX, T> HashStable<CTX> for LazySeq<T> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut CTX,
                                          _: &mut StableHasher<W>) {
        // There's nothing to do. Whatever got encoded within this Lazy<>
        // wrapper has already been hashed.
    }
}

/// Encoding / decoding state for `Lazy` and `LazySeq`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum LazyState {
    /// Outside of a metadata node.
    NoNode,

    /// Inside a metadata node, and before any `Lazy` or `LazySeq`.
    /// The position is that of the node itself.
    NodeStart(usize),

    /// Inside a metadata node, with a previous `Lazy` or `LazySeq`.
    /// The position is a conservative estimate of where that
    /// previous `Lazy` / `LazySeq` would end (see their comments).
    Previous(usize),
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct CrateRoot {
    pub name: Symbol,
    pub triple: String,
    pub hash: hir::svh::Svh,
    pub disambiguator: Symbol,
    pub panic_strategy: PanicStrategy,
    pub has_global_allocator: bool,
    pub has_default_lib_allocator: bool,
    pub plugin_registrar_fn: Option<DefIndex>,
    pub macro_derive_registrar: Option<DefIndex>,

    pub crate_deps: LazySeq<CrateDep>,
    pub dylib_dependency_formats: LazySeq<Option<LinkagePreference>>,
    pub lang_items: LazySeq<(DefIndex, usize)>,
    pub lang_items_missing: LazySeq<lang_items::LangItem>,
    pub native_libraries: LazySeq<NativeLibrary>,
    pub codemap: LazySeq<syntax_pos::FileMap>,
    pub def_path_table: Lazy<hir::map::definitions::DefPathTable>,
    pub impls: LazySeq<TraitImpls>,
    pub exported_symbols: LazySeq<DefIndex>,
    pub index: LazySeq<index::Index>,
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct CrateDep {
    pub name: ast::Name,
    pub hash: hir::svh::Svh,
    pub kind: DepKind,
}

impl_stable_hash_for!(struct CrateDep {
    name,
    hash,
    kind
});

#[derive(RustcEncodable, RustcDecodable)]
pub struct TraitImpls {
    pub trait_id: (u32, DefIndex),
    pub impls: LazySeq<DefIndex>,
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for TraitImpls {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let TraitImpls {
            trait_id: (krate, def_index),
            ref impls,
        } = *self;

        DefId {
            krate: CrateNum::from_u32(krate),
            index: def_index
        }.hash_stable(hcx, hasher);
        impls.hash_stable(hcx, hasher);
    }
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct Entry<'tcx> {
    pub kind: EntryKind<'tcx>,
    pub visibility: Lazy<ty::Visibility>,
    pub span: Lazy<Span>,
    pub attributes: LazySeq<ast::Attribute>,
    pub children: LazySeq<DefIndex>,
    pub stability: Option<Lazy<attr::Stability>>,
    pub deprecation: Option<Lazy<attr::Deprecation>>,

    pub ty: Option<Lazy<Ty<'tcx>>>,
    pub inherent_impls: LazySeq<DefIndex>,
    pub variances: LazySeq<ty::Variance>,
    pub generics: Option<Lazy<ty::Generics>>,
    pub predicates: Option<Lazy<ty::GenericPredicates<'tcx>>>,

    pub ast: Option<Lazy<astencode::Ast<'tcx>>>,
    pub mir: Option<Lazy<mir::Mir<'tcx>>>,
}

impl_stable_hash_for!(struct Entry<'tcx> {
    kind,
    visibility,
    span,
    attributes,
    children,
    stability,
    deprecation,
    ty,
    inherent_impls,
    variances,
    generics,
    predicates,
    ast,
    mir
});

#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum EntryKind<'tcx> {
    Const(u8),
    ImmStatic,
    MutStatic,
    ForeignImmStatic,
    ForeignMutStatic,
    ForeignMod,
    GlobalAsm,
    Type,
    Enum(ReprOptions),
    Field,
    Variant(Lazy<VariantData<'tcx>>),
    Struct(Lazy<VariantData<'tcx>>, ReprOptions),
    Union(Lazy<VariantData<'tcx>>, ReprOptions),
    Fn(Lazy<FnData<'tcx>>),
    ForeignFn(Lazy<FnData<'tcx>>),
    Mod(Lazy<ModData>),
    MacroDef(Lazy<MacroDef>),
    Closure(Lazy<ClosureData<'tcx>>),
    Generator(Lazy<GeneratorData<'tcx>>),
    Trait(Lazy<TraitData<'tcx>>),
    Impl(Lazy<ImplData<'tcx>>),
    DefaultImpl(Lazy<ImplData<'tcx>>),
    Method(Lazy<MethodData<'tcx>>),
    AssociatedType(AssociatedContainer),
    AssociatedConst(AssociatedContainer, u8),
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for EntryKind<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            EntryKind::ImmStatic        |
            EntryKind::MutStatic        |
            EntryKind::ForeignImmStatic |
            EntryKind::ForeignMutStatic |
            EntryKind::ForeignMod       |
            EntryKind::GlobalAsm        |
            EntryKind::Field |
            EntryKind::Type => {
                // Nothing else to hash here.
            }
            EntryKind::Const(qualif) => {
                qualif.hash_stable(hcx, hasher);
            }
            EntryKind::Enum(ref repr_options) => {
                repr_options.hash_stable(hcx, hasher);
            }
            EntryKind::Variant(ref variant_data) => {
                variant_data.hash_stable(hcx, hasher);
            }
            EntryKind::Struct(ref variant_data, ref repr_options) |
            EntryKind::Union(ref variant_data, ref repr_options)  => {
                variant_data.hash_stable(hcx, hasher);
                repr_options.hash_stable(hcx, hasher);
            }
            EntryKind::Fn(ref fn_data) |
            EntryKind::ForeignFn(ref fn_data) => {
                fn_data.hash_stable(hcx, hasher);
            }
            EntryKind::Mod(ref mod_data) => {
                mod_data.hash_stable(hcx, hasher);
            }
            EntryKind::MacroDef(ref macro_def) => {
                macro_def.hash_stable(hcx, hasher);
            }
            EntryKind::Generator(data) => {
                data.hash_stable(hcx, hasher);
            }
            EntryKind::Closure(closure_data) => {
                closure_data.hash_stable(hcx, hasher);
            }
            EntryKind::Trait(ref trait_data) => {
                trait_data.hash_stable(hcx, hasher);
            }
            EntryKind::DefaultImpl(ref impl_data) |
            EntryKind::Impl(ref impl_data) => {
                impl_data.hash_stable(hcx, hasher);
            }
            EntryKind::Method(ref method_data) => {
                method_data.hash_stable(hcx, hasher);
            }
            EntryKind::AssociatedType(associated_container) => {
                associated_container.hash_stable(hcx, hasher);
            }
            EntryKind::AssociatedConst(associated_container, qualif) => {
                associated_container.hash_stable(hcx, hasher);
                qualif.hash_stable(hcx, hasher);
            }
        }
    }
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ModData {
    pub reexports: LazySeq<def::Export>,
}

impl_stable_hash_for!(struct ModData { reexports });

#[derive(RustcEncodable, RustcDecodable)]
pub struct MacroDef {
    pub body: String,
    pub legacy: bool,
}

impl_stable_hash_for!(struct MacroDef { body, legacy });

#[derive(RustcEncodable, RustcDecodable)]
pub struct FnData<'tcx> {
    pub constness: hir::Constness,
    pub arg_names: LazySeq<ast::Name>,
    pub sig: Lazy<ty::PolyFnSig<'tcx>>,
}

impl_stable_hash_for!(struct FnData<'tcx> { constness, arg_names, sig });

#[derive(RustcEncodable, RustcDecodable)]
pub struct VariantData<'tcx> {
    pub ctor_kind: CtorKind,
    pub discr: ty::VariantDiscr,

    /// If this is a struct's only variant, this
    /// is the index of the "struct ctor" item.
    pub struct_ctor: Option<DefIndex>,

    /// If this is a tuple struct or variant
    /// ctor, this is its "function" signature.
    pub ctor_sig: Option<Lazy<ty::PolyFnSig<'tcx>>>,
}

impl_stable_hash_for!(struct VariantData<'tcx> {
    ctor_kind,
    discr,
    struct_ctor,
    ctor_sig
});

#[derive(RustcEncodable, RustcDecodable)]
pub struct TraitData<'tcx> {
    pub unsafety: hir::Unsafety,
    pub paren_sugar: bool,
    pub has_default_impl: bool,
    pub super_predicates: Lazy<ty::GenericPredicates<'tcx>>,
}

impl_stable_hash_for!(struct TraitData<'tcx> {
    unsafety,
    paren_sugar,
    has_default_impl,
    super_predicates
});

#[derive(RustcEncodable, RustcDecodable)]
pub struct ImplData<'tcx> {
    pub polarity: hir::ImplPolarity,
    pub defaultness: hir::Defaultness,
    pub parent_impl: Option<DefId>,

    /// This is `Some` only for impls of `CoerceUnsized`.
    pub coerce_unsized_info: Option<ty::adjustment::CoerceUnsizedInfo>,
    pub trait_ref: Option<Lazy<ty::TraitRef<'tcx>>>,
}

impl_stable_hash_for!(struct ImplData<'tcx> {
    polarity,
    defaultness,
    parent_impl,
    coerce_unsized_info,
    trait_ref
});


/// Describes whether the container of an associated item
/// is a trait or an impl and whether, in a trait, it has
/// a default, or an in impl, whether it's marked "default".
#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum AssociatedContainer {
    TraitRequired,
    TraitWithDefault,
    ImplDefault,
    ImplFinal,
}

impl_stable_hash_for!(enum ::schema::AssociatedContainer {
    TraitRequired,
    TraitWithDefault,
    ImplDefault,
    ImplFinal
});

impl AssociatedContainer {
    pub fn with_def_id(&self, def_id: DefId) -> ty::AssociatedItemContainer {
        match *self {
            AssociatedContainer::TraitRequired |
            AssociatedContainer::TraitWithDefault => ty::TraitContainer(def_id),

            AssociatedContainer::ImplDefault |
            AssociatedContainer::ImplFinal => ty::ImplContainer(def_id),
        }
    }

    pub fn defaultness(&self) -> hir::Defaultness {
        match *self {
            AssociatedContainer::TraitRequired => hir::Defaultness::Default {
                has_value: false,
            },

            AssociatedContainer::TraitWithDefault |
            AssociatedContainer::ImplDefault => hir::Defaultness::Default {
                has_value: true,
            },

            AssociatedContainer::ImplFinal => hir::Defaultness::Final,
        }
    }
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct MethodData<'tcx> {
    pub fn_data: FnData<'tcx>,
    pub container: AssociatedContainer,
    pub has_self: bool,
}
impl_stable_hash_for!(struct MethodData<'tcx> { fn_data, container, has_self });

#[derive(RustcEncodable, RustcDecodable)]
pub struct ClosureData<'tcx> {
    pub kind: ty::ClosureKind,
    pub sig: Lazy<ty::PolyFnSig<'tcx>>,
}
impl_stable_hash_for!(struct ClosureData<'tcx> { kind, sig });

#[derive(RustcEncodable, RustcDecodable)]
pub struct GeneratorData<'tcx> {
    pub sig: ty::PolyGenSig<'tcx>,
    pub layout: mir::GeneratorLayout<'tcx>,
}
impl_stable_hash_for!(struct GeneratorData<'tcx> { sig, layout });
