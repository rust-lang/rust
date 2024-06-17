//! The WIP stable interface to rustc internals.
//!
//! For more information see <https://github.com/rust-lang/project-stable-mir>
//!
//! # Note
//!
//! This API is still completely unstable and subject to change.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
//!
//! This crate shall contain all type definitions and APIs that we expect third-party tools to invoke to
//! interact with the compiler.
//!
//! The goal is to eventually be published on
//! [crates.io](https://crates.io).

use std::fmt;
use std::fmt::Debug;
use std::io;

use crate::compiler_interface::with;
pub use crate::crate_def::{CrateDef, CrateDefType, DefId};
pub use crate::error::*;
use crate::mir::Body;
use crate::mir::Mutability;
use crate::ty::{ForeignModuleDef, ImplDef, IndexedVal, Span, TraitDef, Ty};
use serde::Serialize;

pub mod abi;
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
        f.debug_struct("DefId")
            .field("id", &self.0)
            .field("name", &with(|cx| cx.def_name(*self, false)))
            .finish()
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
    /// This will return the body of an item.
    ///
    /// This will panic if no body is available.
    pub fn body(&self) -> mir::Body {
        with(|cx| cx.mir_body(self.0))
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

    pub fn emit_mir<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        self.body().dump(w, &self.name())
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
