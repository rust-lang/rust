//! Define the interface with the Rust compiler.
//!
//! StableMIR users should not use any of the items in this module directly.
//! These APIs have no stability guarantee.

use std::cell::Cell;

use stable_mir::abi::{FnAbi, Layout, LayoutShape};
use stable_mir::crate_def::Attribute;
use stable_mir::mir::alloc::{AllocId, GlobalAlloc};
use stable_mir::mir::mono::{Instance, InstanceDef, StaticDef};
use stable_mir::mir::{BinOp, Body, Place, UnOp};
use stable_mir::target::MachineInfo;
use stable_mir::ty::{
    AdtDef, AdtKind, Allocation, ClosureDef, ClosureKind, FieldDef, FnDef, ForeignDef,
    ForeignItemKind, ForeignModule, ForeignModuleDef, GenericArgs, GenericPredicates, Generics,
    ImplDef, ImplTrait, IntrinsicDef, LineInfo, MirConst, PolyFnSig, RigidTy, Span, TraitDecl,
    TraitDef, Ty, TyConst, TyConstId, TyKind, UintTy, VariantDef,
};
use stable_mir::{
    AssocItems, Crate, CrateItem, CrateItems, CrateNum, DefId, Error, Filename, ImplTraitDecls,
    ItemKind, Symbol, TraitDecls, mir,
};

use crate::stable_mir;

/// This trait defines the interface between stable_mir and the Rust compiler.
/// Do not use this directly.
pub trait Context {
    fn entry_fn(&self) -> Option<CrateItem>;
    /// Retrieve all items of the local crate that have a MIR associated with them.
    fn all_local_items(&self) -> CrateItems;
    /// Retrieve the body of a function.
    /// This function will panic if the body is not available.
    fn mir_body(&self, item: DefId) -> mir::Body;
    /// Check whether the body of a function is available.
    fn has_body(&self, item: DefId) -> bool;
    fn foreign_modules(&self, crate_num: CrateNum) -> Vec<ForeignModuleDef>;

    /// Retrieve all functions defined in this crate.
    fn crate_functions(&self, crate_num: CrateNum) -> Vec<FnDef>;

    /// Retrieve all static items defined in this crate.
    fn crate_statics(&self, crate_num: CrateNum) -> Vec<StaticDef>;
    fn foreign_module(&self, mod_def: ForeignModuleDef) -> ForeignModule;
    fn foreign_items(&self, mod_def: ForeignModuleDef) -> Vec<ForeignDef>;
    fn all_trait_decls(&self) -> TraitDecls;
    fn trait_decls(&self, crate_num: CrateNum) -> TraitDecls;
    fn trait_decl(&self, trait_def: &TraitDef) -> TraitDecl;
    fn all_trait_impls(&self) -> ImplTraitDecls;
    fn trait_impls(&self, crate_num: CrateNum) -> ImplTraitDecls;
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

    /// Return registered tool attributes with the given attribute name.
    ///
    /// FIXME(jdonszelmann): may panic on non-tool attributes. After more attribute work, non-tool
    /// attributes will simply return an empty list.
    ///
    /// Single segmented name like `#[clippy]` is specified as `&["clippy".to_string()]`.
    /// Multi-segmented name like `#[rustfmt::skip]` is specified as `&["rustfmt".to_string(), "skip".to_string()]`.
    fn tool_attrs(&self, def_id: DefId, attr: &[Symbol]) -> Vec<Attribute>;

    /// Get all tool attributes of a definition.
    fn all_tool_attrs(&self, def_id: DefId) -> Vec<Attribute>;

    /// Returns printable, human readable form of `Span`
    fn span_to_string(&self, span: Span) -> String;

    /// Return filename from given `Span`, for diagnostic purposes
    fn get_filename(&self, span: &Span) -> Filename;

    /// Return lines corresponding to this `Span`
    fn get_lines(&self, span: &Span) -> LineInfo;

    /// Returns the `kind` of given `DefId`
    fn item_kind(&self, item: CrateItem) -> ItemKind;

    /// Returns whether this is a foreign item.
    fn is_foreign_item(&self, item: DefId) -> bool;

    /// Returns the kind of a given foreign item.
    fn foreign_item_kind(&self, def: ForeignDef) -> ForeignItemKind;

    /// Returns the kind of a given algebraic data type
    fn adt_kind(&self, def: AdtDef) -> AdtKind;

    /// Returns if the ADT is a box.
    fn adt_is_box(&self, def: AdtDef) -> bool;

    /// Returns whether this ADT is simd.
    fn adt_is_simd(&self, def: AdtDef) -> bool;

    /// Returns whether this definition is a C string.
    fn adt_is_cstr(&self, def: AdtDef) -> bool;

    /// Retrieve the function signature for the given generic arguments.
    fn fn_sig(&self, def: FnDef, args: &GenericArgs) -> PolyFnSig;

    /// Retrieve the intrinsic definition if the item corresponds one.
    fn intrinsic(&self, item: DefId) -> Option<IntrinsicDef>;

    /// Retrieve the plain function name of an intrinsic.
    fn intrinsic_name(&self, def: IntrinsicDef) -> Symbol;

    /// Retrieve the closure signature for the given generic arguments.
    fn closure_sig(&self, args: &GenericArgs) -> PolyFnSig;

    /// The number of variants in this ADT.
    fn adt_variants_len(&self, def: AdtDef) -> usize;

    /// The name of a variant.
    fn variant_name(&self, def: VariantDef) -> Symbol;
    fn variant_fields(&self, def: VariantDef) -> Vec<FieldDef>;

    /// Evaluate constant as a target usize.
    fn eval_target_usize(&self, cnst: &MirConst) -> Result<u64, Error>;
    fn eval_target_usize_ty(&self, cnst: &TyConst) -> Result<u64, Error>;

    /// Create a new zero-sized constant.
    fn try_new_const_zst(&self, ty: Ty) -> Result<MirConst, Error>;

    /// Create a new constant that represents the given string value.
    fn new_const_str(&self, value: &str) -> MirConst;

    /// Create a new constant that represents the given boolean value.
    fn new_const_bool(&self, value: bool) -> MirConst;

    /// Create a new constant that represents the given value.
    fn try_new_const_uint(&self, value: u128, uint_ty: UintTy) -> Result<MirConst, Error>;
    fn try_new_ty_const_uint(&self, value: u128, uint_ty: UintTy) -> Result<TyConst, Error>;

    /// Create a new type from the given kind.
    fn new_rigid_ty(&self, kind: RigidTy) -> Ty;

    /// Create a new box type, `Box<T>`, for the given inner type `T`.
    fn new_box_ty(&self, ty: Ty) -> Ty;

    /// Returns the type of given crate item.
    fn def_ty(&self, item: DefId) -> Ty;

    /// Returns the type of given definition instantiated with the given arguments.
    fn def_ty_with_args(&self, item: DefId, args: &GenericArgs) -> Ty;

    /// Returns literal value of a const as a string.
    fn mir_const_pretty(&self, cnst: &MirConst) -> String;

    /// `Span` of an item
    fn span_of_an_item(&self, def_id: DefId) -> Span;

    fn ty_const_pretty(&self, ct: TyConstId) -> String;

    /// Obtain the representation of a type.
    fn ty_pretty(&self, ty: Ty) -> String;

    /// Obtain the representation of a type.
    fn ty_kind(&self, ty: Ty) -> TyKind;

    // Get the discriminant Ty for this Ty if there's one.
    fn rigid_ty_discriminant_ty(&self, ty: &RigidTy) -> Ty;

    /// Get the body of an Instance which is already monomorphized.
    fn instance_body(&self, instance: InstanceDef) -> Option<Body>;

    /// Get the instance type with generic instantiations applied and lifetimes erased.
    fn instance_ty(&self, instance: InstanceDef) -> Ty;

    /// Get the instantiation types.
    fn instance_args(&self, def: InstanceDef) -> GenericArgs;

    /// Get the instance.
    fn instance_def_id(&self, instance: InstanceDef) -> DefId;

    /// Get the instance mangled name.
    fn instance_mangled_name(&self, instance: InstanceDef) -> Symbol;

    /// Check if this is an empty DropGlue shim.
    fn is_empty_drop_shim(&self, def: InstanceDef) -> bool;

    /// Check if this is an empty AsyncDropGlueCtor shim.
    fn is_empty_async_drop_ctor_shim(&self, def: InstanceDef) -> bool;

    /// Convert a non-generic crate item into an instance.
    /// This function will panic if the item is generic.
    fn mono_instance(&self, def_id: DefId) -> Instance;

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

    /// Try to evaluate an instance into a constant.
    fn eval_instance(&self, def: InstanceDef, const_ty: Ty) -> Result<Allocation, Error>;

    /// Retrieve global allocation for the given allocation ID.
    fn global_alloc(&self, id: AllocId) -> GlobalAlloc;

    /// Retrieve the id for the virtual table.
    fn vtable_allocation(&self, global_alloc: &GlobalAlloc) -> Option<AllocId>;
    fn krate(&self, def_id: DefId) -> Crate;
    fn instance_name(&self, def: InstanceDef, trimmed: bool) -> Symbol;

    /// Return information about the target machine.
    fn target_info(&self) -> MachineInfo;

    /// Get an instance ABI.
    fn instance_abi(&self, def: InstanceDef) -> Result<FnAbi, Error>;

    /// Get the ABI of a function pointer.
    fn fn_ptr_abi(&self, fn_ptr: PolyFnSig) -> Result<FnAbi, Error>;

    /// Get the layout of a type.
    fn ty_layout(&self, ty: Ty) -> Result<Layout, Error>;

    /// Get the layout shape.
    fn layout_shape(&self, id: Layout) -> LayoutShape;

    /// Get a debug string representation of a place.
    fn place_pretty(&self, place: &Place) -> String;

    /// Get the resulting type of binary operation.
    fn binop_ty(&self, bin_op: BinOp, rhs: Ty, lhs: Ty) -> Ty;

    /// Get the resulting type of unary operation.
    fn unop_ty(&self, un_op: UnOp, arg: Ty) -> Ty;

    /// Get all associated items of a definition.
    fn associated_items(&self, def_id: DefId) -> AssocItems;
}

// A thread local variable that stores a pointer to the tables mapping between TyCtxt
// datastructures and stable MIR datastructures
scoped_tls::scoped_thread_local!(static TLV: Cell<*const ()>);

pub fn run<F, T>(context: &dyn Context, f: F) -> Result<T, Error>
where
    F: FnOnce() -> T,
{
    if TLV.is_set() {
        Err(Error::from("StableMIR already running"))
    } else {
        let ptr: *const () = (&raw const context) as _;
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
