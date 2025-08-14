//! The WIP public interface to rustc internals.
//!
//! For more information see <https://github.com/rust-lang/project-stable-mir>
//!
//! # Note
//!
//! This API is still completely unstable and subject to change.

#![allow(rustc::usage_of_ty_tykind)]
#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![feature(sized_hierarchy)]
//!
//! This crate shall contain all type definitions and APIs that we expect third-party tools to invoke to
//! interact with the compiler.
//!
//! The goal is to eventually be published on
//! [crates.io](https://crates.io).

use std::fmt::Debug;
use std::{fmt, io};

pub(crate) use rustc_public_bridge::IndexedVal;
use rustc_public_bridge::Tables;
use rustc_public_bridge::context::CompilerCtxt;
/// Export the rustc_internal APIs. Note that this module has no stability
/// guarantees and it is not taken into account for semver.
#[cfg(feature = "rustc_internal")]
pub mod rustc_internal;
use serde::Serialize;

use crate::compiler_interface::with;
pub use crate::crate_def::{CrateDef, CrateDefItems, CrateDefType, DefId};
pub use crate::error::*;
use crate::mir::mono::StaticDef;
use crate::mir::{Body, Mutability};
use crate::ty::{AssocItem, FnDef, ForeignModuleDef, ImplDef, ProvenanceMap, Span, TraitDef, Ty};
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
pub mod ty;
pub mod visitor;

/// Use String for now but we should replace it.
pub type Symbol = String;

/// The number that identifies a crate.
pub type CrateNum = usize;

impl Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefId").field("id", &self.0).field("name", &self.name()).finish()
    }
}

impl IndexedVal for DefId {
    fn to_val(index: usize) -> Self {
        DefId(index)
    }

    fn to_index(&self) -> usize {
        self.0
    }
}

/// A list of crate items.
pub type CrateItems = Vec<CrateItem>;

/// A list of trait decls.
pub type TraitDecls = Vec<TraitDef>;

/// A list of impl trait decls.
pub type ImplTraitDecls = Vec<ImplDef>;

/// A list of associated items.
pub type AssocItems = Vec<AssocItem>;

/// Holds information about a crate.
#[derive(Clone, PartialEq, Eq, Debug, Serialize)]
pub struct Crate {
    pub id: CrateNum,
    pub name: Symbol,
    pub is_local: bool,
}

impl Crate {
    /// The list of foreign modules in this crate.
    pub fn foreign_modules(&self) -> Vec<ForeignModuleDef> {
        with(|cx| cx.foreign_modules(self.id))
    }

    /// The list of traits declared in this crate.
    pub fn trait_decls(&self) -> TraitDecls {
        with(|cx| cx.trait_decls(self.id))
    }

    /// The list of trait implementations in this crate.
    pub fn trait_impls(&self) -> ImplTraitDecls {
        with(|cx| cx.trait_impls(self.id))
    }

    /// Return a list of function definitions from this crate independent on their visibility.
    pub fn fn_defs(&self) -> Vec<FnDef> {
        with(|cx| cx.crate_functions(self.id))
    }

    /// Return a list of static items defined in this crate independent on their visibility.
    pub fn statics(&self) -> Vec<StaticDef> {
        with(|cx| cx.crate_statics(self.id))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Serialize)]
pub enum ItemKind {
    Fn,
    Static,
    Const,
    Ctor(CtorKind),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Serialize)]
pub enum CtorKind {
    Const,
    Fn,
}

pub type Filename = String;

crate_def_with_ty! {
    /// Holds information about an item in a crate.
    #[derive(Serialize)]
    pub CrateItem;
}

impl CrateItem {
    /// This will return the body of an item or panic if it's not available.
    pub fn expect_body(&self) -> mir::Body {
        with(|cx| cx.mir_body(self.0))
    }

    /// Return the body of an item if available.
    pub fn body(&self) -> Option<mir::Body> {
        with(|cx| cx.has_body(self.0).then(|| cx.mir_body(self.0)))
    }

    /// Check if a body is available for this item.
    pub fn has_body(&self) -> bool {
        with(|cx| cx.has_body(self.0))
    }

    pub fn span(&self) -> Span {
        with(|cx| cx.span_of_an_item(self.0))
    }

    pub fn kind(&self) -> ItemKind {
        with(|cx| cx.item_kind(*self))
    }

    pub fn requires_monomorphization(&self) -> bool {
        with(|cx| cx.requires_monomorphization(self.0))
    }

    pub fn ty(&self) -> Ty {
        with(|cx| cx.def_ty(self.0))
    }

    pub fn is_foreign_item(&self) -> bool {
        with(|cx| cx.is_foreign_item(self.0))
    }

    /// Emit MIR for this item body.
    pub fn emit_mir<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        self.body()
            .ok_or_else(|| io::Error::other(format!("No body found for `{}`", self.name())))?
            .dump(w, &self.name())
    }
}

/// Return the function where execution starts if the current
/// crate defines that. This is usually `main`, but could be
/// `start` if the crate is a no-std crate.
pub fn entry_fn() -> Option<CrateItem> {
    with(|cx| cx.entry_fn())
}

/// Access to the local crate.
pub fn local_crate() -> Crate {
    with(|cx| cx.local_crate())
}

/// Try to find a crate or crates if multiple crates exist from given name.
pub fn find_crates(name: &str) -> Vec<Crate> {
    with(|cx| cx.find_crates(name))
}

/// Try to find a crate with the given name.
pub fn external_crates() -> Vec<Crate> {
    with(|cx| cx.external_crates())
}

/// Retrieve all items in the local crate that have a MIR associated with them.
pub fn all_local_items() -> CrateItems {
    with(|cx| cx.all_local_items())
}

pub fn all_trait_decls() -> TraitDecls {
    with(|cx| cx.all_trait_decls())
}

pub fn all_trait_impls() -> ImplTraitDecls {
    with(|cx| cx.all_trait_impls())
}

/// A type that provides internal information but that can still be used for debug purpose.
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

pub fn opaque<T: Debug>(value: &T) -> Opaque {
    Opaque(format!("{value:?}"))
}

macro_rules! bridge_impl {
    ($name: ident, $ty: ty) => {
        impl rustc_public_bridge::bridge::$name<compiler_interface::BridgeTys> for $ty {
            fn new(def: crate::DefId) -> Self {
                Self(def)
            }
        }
    };
}

bridge_impl!(CrateItem, crate::CrateItem);
bridge_impl!(AdtDef, crate::ty::AdtDef);
bridge_impl!(ForeignModuleDef, crate::ty::ForeignModuleDef);
bridge_impl!(ForeignDef, crate::ty::ForeignDef);
bridge_impl!(FnDef, crate::ty::FnDef);
bridge_impl!(ClosureDef, crate::ty::ClosureDef);
bridge_impl!(CoroutineDef, crate::ty::CoroutineDef);
bridge_impl!(CoroutineClosureDef, crate::ty::CoroutineClosureDef);
bridge_impl!(AliasDef, crate::ty::AliasDef);
bridge_impl!(ParamDef, crate::ty::ParamDef);
bridge_impl!(BrNamedDef, crate::ty::BrNamedDef);
bridge_impl!(TraitDef, crate::ty::TraitDef);
bridge_impl!(GenericDef, crate::ty::GenericDef);
bridge_impl!(ConstDef, crate::ty::ConstDef);
bridge_impl!(ImplDef, crate::ty::ImplDef);
bridge_impl!(RegionDef, crate::ty::RegionDef);
bridge_impl!(CoroutineWitnessDef, crate::ty::CoroutineWitnessDef);
bridge_impl!(AssocDef, crate::ty::AssocDef);
bridge_impl!(OpaqueDef, crate::ty::OpaqueDef);
bridge_impl!(StaticDef, crate::mir::mono::StaticDef);

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
