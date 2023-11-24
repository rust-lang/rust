//! Define the interface with the Rust compiler.
//!
//! StableMIR users should not use any of the items in this module directly.
//! These APIs have no stability guarantee.

use std::cell::Cell;

use crate::mir::alloc::{AllocId, GlobalAlloc};
use crate::mir::mono::{Instance, InstanceDef, StaticDef};
use crate::mir::Body;
use crate::ty::{
    AdtDef, AdtKind, Allocation, ClosureDef, ClosureKind, Const, FnDef, GenericArgs,
    GenericPredicates, Generics, ImplDef, ImplTrait, LineInfo, RigidTy, Span, TraitDecl, TraitDef,
    Ty, TyKind,
};
use crate::{
    mir, Crate, CrateItem, CrateItems, DefId, Error, Filename, ImplTraitDecls, ItemKind, Symbol,
    TraitDecls,
};

/// This trait defines the interface between stable_mir and the Rust compiler.
/// Do not use this directly.
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
    fn def_name(&self, def_id: DefId, trimmed: bool) -> Symbol;

    /// Returns printable, human readable form of `Span`
    fn span_to_string(&self, span: Span) -> String;

    /// Return filename from given `Span`, for diagnostic purposes
    fn get_filename(&self, span: &Span) -> Filename;

    /// Return lines corresponding to this `Span`
    fn get_lines(&self, span: &Span) -> LineInfo;

    /// Returns the `kind` of given `DefId`
    fn item_kind(&self, item: CrateItem) -> ItemKind;

    /// Returns whether this is a foreign item.
    fn is_foreign_item(&self, item: CrateItem) -> bool;

    /// Returns the kind of a given algebraic data type
    fn adt_kind(&self, def: AdtDef) -> AdtKind;

    /// Returns if the ADT is a box.
    fn adt_is_box(&self, def: AdtDef) -> bool;

    /// Evaluate constant as a target usize.
    fn eval_target_usize(&self, cnst: &Const) -> Result<u64, Error>;

    /// Create a target usize constant for the given value.
    fn usize_to_const(&self, val: u64) -> Result<Const, Error>;

    /// Create a new type from the given kind.
    fn new_rigid_ty(&self, kind: RigidTy) -> Ty;

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
    fn instance_mangled_name(&self, instance: InstanceDef) -> Symbol;

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

    /// Evaluate a static's initializer.
    fn eval_static_initializer(&self, def: StaticDef) -> Result<Allocation, Error>;

    /// Retrieve global allocation for the given allocation ID.
    fn global_alloc(&self, id: AllocId) -> GlobalAlloc;

    /// Retrieve the id for the virtual table.
    fn vtable_allocation(&self, global_alloc: &GlobalAlloc) -> Option<AllocId>;
    fn krate(&self, def_id: DefId) -> Crate;
    fn instance_name(&self, def: InstanceDef, trimmed: bool) -> Symbol;
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

/// Execute the given function with access the compiler [Context].
///
/// I.e., This function will load the current context and calls a function with it.
/// Do not nest these, as that will ICE.
pub(crate) fn with<R>(f: impl FnOnce(&dyn Context) -> R) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        f(unsafe { *(ptr as *const &dyn Context) })
    })
}
