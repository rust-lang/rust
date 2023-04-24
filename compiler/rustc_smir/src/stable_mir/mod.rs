//! Module that implements the public interface to the Stable MIR.
//!
//! This module shall contain all type definitions and APIs that we expect 3P tools to invoke to
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

use crate::rustc_smir::Tables;

use self::ty::{Ty, TyKind};

pub mod mir;
pub mod ty;

/// Use String for now but we should replace it.
pub type Symbol = String;

/// The number that identifies a crate.
pub type CrateNum = usize;

/// A unique identification number for each item accessible for the current compilation unit.
pub type DefId = usize;

/// A list of crate items.
pub type CrateItems = Vec<CrateItem>;

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
        with(|cx| cx.mir_body(self))
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

pub trait Context {
    fn entry_fn(&mut self) -> Option<CrateItem>;
    /// Retrieve all items of the local crate that have a MIR associated with them.
    fn all_local_items(&mut self) -> CrateItems;
    fn mir_body(&mut self, item: &CrateItem) -> mir::Body;
    /// Get information about the local crate.
    fn local_crate(&self) -> Crate;
    /// Retrieve a list of all external crates.
    fn external_crates(&self) -> Vec<Crate>;

    /// Find a crate with the given name.
    fn find_crate(&self, name: &str) -> Option<Crate>;

    /// Obtain the representation of a type.
    fn ty_kind(&mut self, ty: Ty) -> TyKind;

    /// HACK: Until we have fully stable consumers, we need an escape hatch
    /// to get `DefId`s out of `CrateItem`s.
    fn rustc_tables(&mut self, f: &mut dyn FnMut(&mut Tables<'_>));
}

thread_local! {
    /// A thread local variable that stores a pointer to the tables mapping between TyCtxt
    /// datastructures and stable MIR datastructures.
    static TLV: Cell<*mut ()> = const { Cell::new(std::ptr::null_mut()) };
}

pub fn run(mut context: impl Context, f: impl FnOnce()) {
    assert!(TLV.get().is_null());
    fn g<'a>(mut context: &mut (dyn Context + 'a), f: impl FnOnce()) {
        TLV.set(&mut context as *mut &mut _ as _);
        f();
        TLV.replace(std::ptr::null_mut());
    }
    g(&mut context, f);
}

/// Loads the current context and calls a function with it.
/// Do not nest these, as that will ICE.
pub(crate) fn with<R>(f: impl FnOnce(&mut dyn Context) -> R) -> R {
    let ptr = TLV.replace(std::ptr::null_mut()) as *mut &mut dyn Context;
    assert!(!ptr.is_null());
    let ret = f(unsafe { *ptr });
    TLV.set(ptr as _);
    ret
}
