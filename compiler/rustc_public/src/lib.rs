//! # `rustc_public` — A Public Interface to `rustc`
//!
//! This crate provides a public API for querying and analyzing Rust programs through the
//! compiler's internal representations. It is designed for third-party tools such as
//! verification engines, linters, and code generators that need access to type information,
//! MIR bodies, monomorphized instances, and ABI details.
//!
//! The goal is to publish this crate on [crates.io](https://crates.io) with semver
//! guarantees. For more details on the proposed plan, see
//! <https://github.com/rust-lang/compiler-team/issues/949>.
//!
//! ## Usage
//!
//! For now, the entry point is the [`run!`] macro, which sets up the compiler session and provides
//! access to the API within a callback. All queries must be performed inside this callback
//! since the data structures are tied to the compiler's thread-local state.
//!
//! ```rust,ignore (requires compiler session)
//! use rustc_public::*;
//! use std::ops::ControlFlow;
//!
//! let result = run!(args, || -> ControlFlow<()> {
//!     // Find all crates with the same name (potential duplicates).
//!     for krate in external_crates() {
//!         let dupes = find_crates(&krate.name);
//!         if dupes.len() > 1 {
//!             println!("Warning: multiple versions of `{}`", krate.name);
//!         }
//!     }
//!     ControlFlow::Continue(())
//! });
//! ```
//!
//! ## Crate Discovery
//!
//! Use [`local_crate()`] to access the crate being compiled, [`external_crates()`] to list
//! dependencies, or [`find_crates()`] to search by name. Use [`entry_fn()`] to find the
//! program entry point, and [`all_local_items()`] to retrieve all local definitions.
//!
//! ## Status
//!
//! This API is not yet published and is still subject to breaking changes.
//! For more information, see <https://github.com/rust-lang/rustc_public>.

#![allow(rustc::usage_of_ty_tykind)]
#![doc(test(attr(allow(unused_variables), deny(warnings), allow(internal_features))))]
#![feature(sized_hierarchy)]

use std::fmt::Debug;
use std::marker::PhantomData;
use std::{fmt, io};

pub(crate) use rustc_public_bridge::IndexedVal;
use rustc_public_bridge::Tables;
use rustc_public_bridge::context::CompilerCtxt;
use serde::Serialize;

/// Unstable internal APIs for bridging with `rustc` internals.
///
/// This module has no stability guarantees and is not covered by semver.
/// It is only available when the `rustc_internal` feature is enabled.
#[cfg(feature = "rustc_internal")]
pub mod rustc_internal;

use crate::compiler_interface::with;
pub use crate::crate_def::{CrateDef, CrateDefType, DefId};
pub use crate::error::*;
use crate::mir::mono::StaticDef;
use crate::mir::{Body, Mutability};
use crate::ty::{
    AdtDef, AssocItem, FnDef, ForeignModuleDef, ImplDef, ProvenanceMap, Span, TraitDef, Ty,
    serialize_index_impl,
};
use crate::unstable::Stable;

pub mod abi;
mod alloc;
pub(crate) mod unstable;
#[macro_use]
pub mod crate_def;
pub mod compiler_interface;
#[macro_use]
pub mod error;
pub mod mir;
pub mod target;
#[cfg(test)]
mod tests;
pub mod ty;
pub mod visitor;

// FIXME: Consider replacing with an opaque or interned type.
/// A symbol name (e.g., function name, crate name), currently represented as a `String`.
pub type Symbol = String;

/// A unique identifier for a crate within the current compilation session.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CrateNum(pub(crate) usize, ThreadLocalIndex);
serialize_index_impl!(CrateNum);

impl Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefId").field("id", &self.0).field("name", &self.name()).finish()
    }
}

/// A collection of items defined in a crate.
pub type CrateItems = Vec<CrateItem>;

/// A collection of trait declarations.
pub type TraitDecls = Vec<TraitDef>;

/// A collection of trait implementation blocks.
pub type ImplTraitDecls = Vec<ImplDef>;

/// A collection of associated items (methods, constants, or types within a trait or impl).
pub type AssocItems = Vec<AssocItem>;

/// Metadata about a crate in the current compilation.
///
/// Use [`local_crate()`] to obtain the crate being compiled, or [`external_crates()`]
/// and [`find_crates()`] to discover dependencies.
#[derive(Clone, PartialEq, Eq, Debug, Serialize)]
pub struct Crate {
    /// The crate's unique identifier in this compilation session.
    pub id: CrateNum,
    /// The crate name as declared in `Cargo.toml` or `--crate-name`.
    pub name: Symbol,
    /// Whether this is the crate currently being compiled.
    pub is_local: bool,
}

impl Crate {
    /// Return all foreign modules (e.g., `extern "C" { ... }` blocks) in this crate.
    pub fn foreign_modules(&self) -> Vec<ForeignModuleDef> {
        with(|cx| cx.foreign_modules(self.id))
    }

    /// Return all trait declarations in this crate.
    ///
    /// For the local crate, this includes private traits. For external crates,
    /// it returns all traits available in metadata.
    pub fn trait_decls(&self) -> TraitDecls {
        with(|cx| cx.trait_decls(self.id))
    }

    /// Return all trait implementation blocks in this crate.
    pub fn trait_impls(&self) -> ImplTraitDecls {
        with(|cx| cx.trait_impls(self.id))
    }

    /// Return all function definitions in this crate.
    ///
    /// For the local crate, this includes private functions. For external crates,
    /// it returns all functions available in metadata.
    pub fn fn_defs(&self) -> Vec<FnDef> {
        with(|cx| cx.crate_functions(self.id))
    }

    /// Return all static items in this crate.
    ///
    /// For the local crate, this includes private statics. For external crates,
    /// it returns all statics available in metadata.
    pub fn statics(&self) -> Vec<StaticDef> {
        with(|cx| cx.crate_statics(self.id))
    }

    /// Return all ADT (struct, enum, union) definitions in this crate.
    ///
    /// For the local crate, this includes private types. For external crates,
    /// it returns all ADTs available in metadata.
    pub fn adts(&self) -> Vec<AdtDef> {
        with(|cx| cx.crate_adts(self.id))
    }
}

/// The kind of a [`CrateItem`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Serialize)]
pub enum ItemKind {
    /// A function (`fn`) or method.
    Fn,
    /// A static variable (`static`).
    Static,
    /// A compile-time constant (`const`).
    Const,
    /// A data constructor for a struct or enum variant.
    Ctor(CtorKind),
}

/// The kind of a data constructor.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Serialize)]
pub enum CtorKind {
    /// A unit or constant constructor (e.g., `None`, `struct Foo;`).
    Const,
    /// A function-like constructor (e.g., `Some(x)`, `struct Foo(u32)`).
    Fn,
}

/// A file path string used for source locations and diagnostics.
pub type Filename = String;

crate_def_with_ty! {
    /// A definition in the local crate (function, static, const, or constructor).
    ///
    /// Obtain instances via [`all_local_items()`] or [`Crate::fn_defs()`].
    /// Use [`CrateItem::body()`] to access the MIR, or convert to an
    /// [`Instance`](mir::mono::Instance) for monomorphized analysis.
    #[derive(Serialize)]
    pub CrateItem;
}

impl CrateItem {
    /// Return the MIR body of this item, panicking if unavailable.
    ///
    /// Prefer [`body()`](Self::body) for a non-panicking alternative.
    pub fn expect_body(&self) -> mir::Body {
        with(|cx| cx.mir_body(self.0))
    }

    /// Return the MIR body of this item, or `None` if unavailable.
    ///
    /// A body may be unavailable for foreign items or compiler built-ins.
    pub fn body(&self) -> Option<mir::Body> {
        with(|cx| cx.has_body(self.0).then(|| cx.mir_body(self.0)))
    }

    /// Check whether this item has a MIR body available.
    pub fn has_body(&self) -> bool {
        with(|cx| cx.has_body(self.0))
    }

    /// Return the source span of this item's definition.
    pub fn span(&self) -> Span {
        self.0.span()
    }

    /// Return what kind of item this is (function, static, const, or constructor).
    pub fn kind(&self) -> ItemKind {
        with(|cx| cx.item_kind(*self))
    }

    /// Check whether this item is generic and requires monomorphization.
    ///
    /// Returns `true` if this item, or any enclosing definition (e.g., the impl
    /// block it belongs to), has type or const generic parameters.
    pub fn requires_monomorphization(&self) -> bool {
        with(|cx| cx.requires_monomorphization(self.0))
    }

    /// Return the type of this item.
    ///
    /// For functions, this returns the function type (e.g., `fn(u32) -> bool`).
    pub fn ty(&self) -> Ty {
        with(|cx| cx.def_ty(self.0))
    }

    /// Check whether this item was declared inside an `extern` block.
    ///
    /// Foreign items (e.g., `extern "C" { fn foo(); }`) are locally defined
    /// declarations of externally-linked symbols.
    pub fn is_foreign_item(&self) -> bool {
        with(|cx| cx.is_foreign_item(self.0))
    }

    /// Write the MIR textual representation of this item's body to `w`.
    pub fn emit_mir<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        self.body()
            .ok_or_else(|| io::Error::other(format!("No body found for `{}`", self.name())))?
            .dump(w, &self.trimmed_name())
    }
}

/// Return the program entry point (usually `main`) if defined in the local crate.
///
/// Returns `None` for library crates. For `no_std` crates this may resolve to
/// a `#[start]` function.
pub fn entry_fn() -> Option<CrateItem> {
    with(|cx| cx.entry_fn())
}

/// Return the local crate (the crate currently being compiled).
pub fn local_crate() -> Crate {
    with(|cx| cx.local_crate())
}

/// Find all crates matching the given name.
///
/// Multiple crates with the same name can exist when different versions of the
/// same dependency are linked.
pub fn find_crates(name: &str) -> Vec<Crate> {
    with(|cx| cx.find_crates(name))
}

/// Return all external (non-local) crates in the compilation.
pub fn external_crates() -> Vec<Crate> {
    with(|cx| cx.external_crates())
}

/// Return all items in the local crate that have a MIR body.
///
/// This includes functions, closures, statics with initializers, and constants.
pub fn all_local_items() -> CrateItems {
    with(|cx| cx.all_local_items())
}

/// Return all trait declarations from the local crate and all its dependencies.
///
/// This includes private traits. Use [`Crate::trait_decls()`] to query a specific crate.
pub fn all_trait_decls() -> TraitDecls {
    with(|cx| cx.all_trait_decls())
}

/// Return all trait implementations from the local crate and all its dependencies.
///
/// Use [`Crate::trait_impls()`] to query a specific crate.
pub fn all_trait_impls() -> ImplTraitDecls {
    with(|cx| cx.all_trait_impls())
}

/// An opaque wrapper around internal compiler data.
///
/// This type is used for compiler details that are exposed for debug printing
/// but whose internal structure is not part of the public API.
#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Opaque(String);

impl std::fmt::Display for Opaque {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Debug for Opaque {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Create an [`Opaque`] value from any debuggable type.
pub fn opaque<T: Debug>(value: &T) -> Opaque {
    Opaque(format!("{value:?}"))
}

macro_rules! bridge_impl {
    ( $( $name:ident, $ty:ty ),* $(,)? ) => {
        $(
            impl rustc_public_bridge::bridge::$name<compiler_interface::BridgeTys> for $ty {
                fn new(def: crate::DefId) -> Self {
                    Self(def)
                }
            }
        )*
    };
}

#[rustfmt::skip]
bridge_impl!(
    CrateItem,           crate::CrateItem,
    AdtDef,              crate::ty::AdtDef,
    ForeignModuleDef,    crate::ty::ForeignModuleDef,
    ForeignDef,          crate::ty::ForeignDef,
    FnDef,               crate::ty::FnDef,
    ClosureDef,          crate::ty::ClosureDef,
    CoroutineDef,        crate::ty::CoroutineDef,
    CoroutineClosureDef, crate::ty::CoroutineClosureDef,
    AliasDef,            crate::ty::AliasDef,
    ParamDef,            crate::ty::ParamDef,
    BrNamedDef,          crate::ty::BrNamedDef,
    TraitDef,            crate::ty::TraitDef,
    GenericDef,          crate::ty::GenericDef,
    ConstDef,            crate::ty::ConstDef,
    ImplDef,             crate::ty::ImplDef,
    RegionDef,           crate::ty::RegionDef,
    CoroutineWitnessDef, crate::ty::CoroutineWitnessDef,
    AssocDef,            crate::ty::AssocDef,
    OpaqueDef,           crate::ty::OpaqueDef,
    StaticDef,           crate::mir::mono::StaticDef
);

impl rustc_public_bridge::bridge::Prov<compiler_interface::BridgeTys> for crate::ty::Prov {
    fn new(aid: crate::mir::alloc::AllocId) -> Self {
        Self(aid)
    }
}

impl rustc_public_bridge::bridge::Allocation<compiler_interface::BridgeTys>
    for crate::ty::Allocation
{
    fn new<'tcx>(
        bytes: Vec<Option<u8>>,
        ptrs: Vec<(usize, rustc_middle::mir::interpret::AllocId)>,
        align: u64,
        mutability: rustc_middle::mir::Mutability,
        tables: &mut Tables<'tcx, compiler_interface::BridgeTys>,
        cx: &CompilerCtxt<'tcx, compiler_interface::BridgeTys>,
    ) -> Self {
        Self {
            bytes,
            provenance: ProvenanceMap {
                ptrs: ptrs.iter().map(|(i, aid)| (*i, tables.prov(*aid))).collect(),
            },
            align,
            mutability: mutability.stable(tables, cx),
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
/// Marker type for indexes into thread local structures.
///
/// Makes things `!Send`/`!Sync`, so users don't move `rustc_public` types to
/// thread with no (or worse, different) `rustc_public` pointer.
///
/// Note. This doesn't make it impossible to confuse TLS. You could return a
/// `DefId` from one `run!` invocation, and then use it inside a different
/// `run!` invocation with different tables.
pub(crate) struct ThreadLocalIndex {
    _phantom: PhantomData<*const ()>,
}
#[expect(non_upper_case_globals)]
/// Emulating unit struct `struct ThreadLocalIndex`;
pub(crate) const ThreadLocalIndex: ThreadLocalIndex = ThreadLocalIndex { _phantom: PhantomData };

impl fmt::Debug for ThreadLocalIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ThreadLocalIndex").finish()
    }
}
