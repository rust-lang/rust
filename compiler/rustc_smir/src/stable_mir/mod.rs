//! Module that implements the public interface to the Stable MIR.
//!
//! This module shall contain all type definitions and APIs that we expect third-party tools to invoke to
//! interact with the compiler.
//!
//! The goal is to eventually move this module to its own crate which shall be published on
//! [crates.io](https://crates.io).
//!
//! ## Note:
//!
//! There shouldn't be any direct references to internal compiler constructs in this module.
//! If you need an internal construct, consider using `rustc_internal` or `rustc_smir`.

use std::cell::Cell;
use std::fmt;
use std::fmt::Debug;

use self::ty::{
    GenericPredicates, Generics, ImplDef, ImplTrait, Span, TraitDecl, TraitDef, Ty, TyKind,
};
use crate::rustc_smir::Tables;

pub mod fold;
pub mod mir;
pub mod ty;
pub mod visitor;

/// Use String for now but we should replace it.
pub type Symbol = String;

/// The number that identifies a crate.
pub type CrateNum = usize;

/// A unique identification number for each item accessible for the current compilation unit.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DefId(pub(crate) usize);

impl Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefId:")
            .field("id", &self.0)
            .field("name", &with(|cx| cx.name_of_def_id(*self)))
            .finish()
    }
}

/// A unique identification number for each provenance
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AllocId(pub(crate) usize);

/// A list of crate items.
pub type CrateItems = Vec<CrateItem>;

/// A list of trait decls.
pub type TraitDecls = Vec<TraitDef>;

/// A list of impl trait decls.
pub type ImplTraitDecls = Vec<ImplDef>;

/// An error type used to represent an error that has already been reported by the compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompilerError<T> {
    /// Internal compiler error (I.e.: Compiler crashed).
    ICE,
    /// Compilation failed.
    CompilationFailed,
    /// Compilation was interrupted.
    Interrupted(T),
    /// Compilation skipped. This happens when users invoke rustc to retrieve information such as
    /// --version.
    Skipped,
}

/// Holds information about a crate.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Crate {
    pub(crate) id: CrateNum,
    pub name: Symbol,
    pub is_local: bool,
}

/// Holds information about an item in the crate.
/// For now, it only stores the item DefId. Use functions inside `rustc_internal` module to
/// use this item.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CrateItem(pub(crate) DefId);

impl CrateItem {
    pub fn body(&self) -> mir::Body {
        with(|cx| cx.mir_body(self.0))
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

/// Try to find a crate with the given name.
pub fn find_crate(name: &str) -> Option<Crate> {
    with(|cx| cx.find_crate(name))
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
    fn entry_fn(&mut self) -> Option<CrateItem>;
    /// Retrieve all items of the local crate that have a MIR associated with them.
    fn all_local_items(&mut self) -> CrateItems;
    fn mir_body(&mut self, item: DefId) -> mir::Body;
    fn all_trait_decls(&mut self) -> TraitDecls;
    fn trait_decl(&mut self, trait_def: &TraitDef) -> TraitDecl;
    fn all_trait_impls(&mut self) -> ImplTraitDecls;
    fn trait_impl(&mut self, trait_impl: &ImplDef) -> ImplTrait;
    fn generics_of(&mut self, def_id: DefId) -> Generics;
    fn predicates_of(&mut self, def_id: DefId) -> GenericPredicates;
    /// Get information about the local crate.
    fn local_crate(&self) -> Crate;
    /// Retrieve a list of all external crates.
    fn external_crates(&self) -> Vec<Crate>;

    /// Find a crate with the given name.
    fn find_crate(&self, name: &str) -> Option<Crate>;

    /// Prints the name of given `DefId`
    fn name_of_def_id(&self, def_id: DefId) -> String;

    /// Obtain the representation of a type.
    fn ty_kind(&mut self, ty: Ty) -> TyKind;

    /// Create a new `Ty` from scratch without information from rustc.
    fn mk_ty(&mut self, kind: TyKind) -> Ty;

    /// HACK: Until we have fully stable consumers, we need an escape hatch
    /// to get `DefId`s out of `CrateItem`s.
    fn rustc_tables(&mut self, f: &mut dyn FnMut(&mut Tables<'_>));
}

// A thread local variable that stores a pointer to the tables mapping between TyCtxt
// datastructures and stable MIR datastructures
scoped_thread_local! (static TLV: Cell<*mut ()>);

pub fn run(mut context: impl Context, f: impl FnOnce()) {
    assert!(!TLV.is_set());
    fn g<'a>(mut context: &mut (dyn Context + 'a), f: impl FnOnce()) {
        let ptr: *mut () = &mut context as *mut &mut _ as _;
        TLV.set(&Cell::new(ptr), || {
            f();
        });
    }
    g(&mut context, f);
}

/// Loads the current context and calls a function with it.
/// Do not nest these, as that will ICE.
pub(crate) fn with<R>(f: impl FnOnce(&mut dyn Context) -> R) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        f(unsafe { *(ptr as *mut &mut dyn Context) })
    })
}
