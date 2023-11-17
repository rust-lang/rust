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

use crate::mir::mono::InstanceDef;
use crate::mir::Body;
use std::fmt;
use std::fmt::Debug;
use std::{cell::Cell, io};

use self::ty::{
    GenericPredicates, Generics, ImplDef, ImplTrait, IndexedVal, LineInfo, Span, TraitDecl,
    TraitDef, Ty, TyKind,
};

#[macro_use]
extern crate scoped_tls;

pub mod error;
pub mod mir;
pub mod ty;
pub mod visitor;

use crate::mir::pretty::function_name;
use crate::mir::Mutability;
use crate::ty::{AdtDef, AdtKind, ClosureDef, ClosureKind};
pub use error::*;
use mir::mono::Instance;
use ty::{Const, FnDef, GenericArgs};

/// Use String for now but we should replace it.
pub type Symbol = String;

/// The number that identifies a crate.
pub type CrateNum = usize;

/// A unique identification number for each item accessible for the current compilation unit.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(usize);

impl Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefId")
            .field("id", &self.0)
            .field("name", &with(|cx| cx.name_of_def_id(*self)))
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

/// A unique identification number for each provenance
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AllocId(usize);

impl IndexedVal for AllocId {
    fn to_val(index: usize) -> Self {
        AllocId(index)
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
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Crate {
    pub id: CrateNum,
    pub name: Symbol,
    pub is_local: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ItemKind {
    Fn,
    Static,
    Const,
}

pub type Filename = Opaque;

/// Holds information about an item in the crate.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CrateItem(pub DefId);

impl CrateItem {
    pub fn body(&self) -> mir::Body {
        with(|cx| cx.mir_body(self.0))
    }

    pub fn span(&self) -> Span {
        with(|cx| cx.span_of_an_item(self.0))
    }

    pub fn name(&self) -> String {
        with(|cx| cx.name_of_def_id(self.0))
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

    pub fn dump<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(w, "{}", function_name(*self))?;
        self.body().dump(w)
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

pub fn trait_decl(trait_def: &TraitDef) -> TraitDecl {
    with(|cx| cx.trait_decl(trait_def))
}

pub fn all_trait_impls() -> ImplTraitDecls {
    with(|cx| cx.all_trait_impls())
}

pub fn trait_impl(trait_impl: &ImplDef) -> ImplTrait {
    with(|cx| cx.trait_impl(trait_impl))
}

pub trait Context {
    fn entry_fn(&self) -> Option<CrateItem>;
    /// Retrieve all items of the local crate that have a MIR associated with them.
    fn all_local_items(&self) -> CrateItems;
    fn mir_body(&self, item: DefId) -> mir::Body;
    fn all_trait_decls(&self) -> TraitDecls;
    fn trait_decl(&self, trait_def: &TraitDef) -> TraitDecl;
    fn all_trait_impls(&self) -> ImplTraitDecls;
    fn trait_impl(&self, trait_impl: &ImplDef) -> ImplTrait;
    fn generics_of(&self, def_id: DefId) -> Generics;
    fn predicates_of(&self, def_id: DefId) -> GenericPredicates;
    fn explicit_predicates_of(&self, def_id: DefId) -> GenericPredicates;
    /// Get information about the local crate.
    fn local_crate(&self) -> Crate;
    /// Retrieve a list of all external crates.
    fn external_crates(&self) -> Vec<Crate>;

    /// Find a crate with the given name.
    fn find_crates(&self, name: &str) -> Vec<Crate>;

    /// Returns the name of given `DefId`
    fn name_of_def_id(&self, def_id: DefId) -> String;

    /// Returns printable, human readable form of `Span`
    fn span_to_string(&self, span: Span) -> String;

    /// Return filename from given `Span`, for diagnostic purposes
    fn get_filename(&self, span: &Span) -> Filename;

    /// Return lines corresponding to this `Span`
    fn get_lines(&self, span: &Span) -> LineInfo;

    /// Returns the `kind` of given `DefId`
    fn item_kind(&self, item: CrateItem) -> ItemKind;

    /// Returns the kind of a given algebraic data type
    fn adt_kind(&self, def: AdtDef) -> AdtKind;

    /// Returns the type of given crate item.
    fn def_ty(&self, item: DefId) -> Ty;

    /// Returns literal value of a const as a string.
    fn const_literal(&self, cnst: &Const) -> String;

    /// `Span` of an item
    fn span_of_an_item(&self, def_id: DefId) -> Span;

    /// Obtain the representation of a type.
    fn ty_kind(&self, ty: Ty) -> TyKind;

    /// Get the body of an Instance.
    /// FIXME: Monomorphize the body.
    fn instance_body(&self, instance: InstanceDef) -> Option<Body>;

    /// Get the instance type with generic substitutions applied and lifetimes erased.
    fn instance_ty(&self, instance: InstanceDef) -> Ty;

    /// Get the instance.
    fn instance_def_id(&self, instance: InstanceDef) -> DefId;

    /// Get the instance mangled name.
    fn instance_mangled_name(&self, instance: InstanceDef) -> String;

    /// Convert a non-generic crate item into an instance.
    /// This function will panic if the item is generic.
    fn mono_instance(&self, item: CrateItem) -> Instance;

    /// Item requires monomorphization.
    fn requires_monomorphization(&self, def_id: DefId) -> bool;

    /// Resolve an instance from the given function definition and generic arguments.
    fn resolve_instance(&self, def: FnDef, args: &GenericArgs) -> Option<Instance>;

    /// Resolve an instance for drop_in_place for the given type.
    fn resolve_drop_in_place(&self, ty: Ty) -> Instance;

    /// Resolve instance for a function pointer.
    fn resolve_for_fn_ptr(&self, def: FnDef, args: &GenericArgs) -> Option<Instance>;

    /// Resolve instance for a closure with the requested type.
    fn resolve_closure(
        &self,
        def: ClosureDef,
        args: &GenericArgs,
        kind: ClosureKind,
    ) -> Option<Instance>;
}

// A thread local variable that stores a pointer to the tables mapping between TyCtxt
// datastructures and stable MIR datastructures
scoped_thread_local! (static TLV: Cell<*const ()>);

pub fn run<F, T>(context: &dyn Context, f: F) -> Result<T, Error>
where
    F: FnOnce() -> T,
{
    if TLV.is_set() {
        Err(Error::from("StableMIR already running"))
    } else {
        let ptr: *const () = &context as *const &_ as _;
        TLV.set(&Cell::new(ptr), || Ok(f()))
    }
}

/// Loads the current context and calls a function with it.
/// Do not nest these, as that will ICE.
pub fn with<R>(f: impl FnOnce(&dyn Context) -> R) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        f(unsafe { *(ptr as *const &dyn Context) })
    })
}

/// A type that provides internal information but that can still be used for debug purpose.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Opaque(String);

impl std::fmt::Display for Opaque {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Debug for Opaque {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

pub fn opaque<T: Debug>(value: &T) -> Opaque {
    Opaque(format!("{value:?}"))
}
