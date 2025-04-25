//! Define the interface with the Rust compiler.
//!
//! StableMIR users should not use any of the items in this module directly.
//! These APIs have no stability guarantee.

use std::cell::Cell;

use rustc_smir::context::SmirCtxt;
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

use crate::{rustc_smir, stable_mir};

/// Stable public API for querying compiler information.
///
/// All queries are delegated to an internal [`SmirCtxt`] that provides
/// similar APIs but based on internal rustc constructs.
///
/// Do not use this directly. This is currently used in the macro expansion.
pub(crate) struct SmirInterface<'tcx> {
    pub(crate) cx: SmirCtxt<'tcx>,
}

impl<'tcx> SmirInterface<'tcx> {
    pub(crate) fn entry_fn(&self) -> Option<CrateItem> {
        self.cx.entry_fn()
    }

    /// Retrieve all items of the local crate that have a MIR associated with them.
    pub(crate) fn all_local_items(&self) -> CrateItems {
        self.cx.all_local_items()
    }

    /// Retrieve the body of a function.
    /// This function will panic if the body is not available.
    pub(crate) fn mir_body(&self, item: DefId) -> mir::Body {
        self.cx.mir_body(item)
    }

    /// Check whether the body of a function is available.
    pub(crate) fn has_body(&self, item: DefId) -> bool {
        self.cx.has_body(item)
    }

    pub(crate) fn foreign_modules(&self, crate_num: CrateNum) -> Vec<ForeignModuleDef> {
        self.cx.foreign_modules(crate_num)
    }

    /// Retrieve all functions defined in this crate.
    pub(crate) fn crate_functions(&self, crate_num: CrateNum) -> Vec<FnDef> {
        self.cx.crate_functions(crate_num)
    }

    /// Retrieve all static items defined in this crate.
    pub(crate) fn crate_statics(&self, crate_num: CrateNum) -> Vec<StaticDef> {
        self.cx.crate_statics(crate_num)
    }

    pub(crate) fn foreign_module(&self, mod_def: ForeignModuleDef) -> ForeignModule {
        self.cx.foreign_module(mod_def)
    }

    pub(crate) fn foreign_items(&self, mod_def: ForeignModuleDef) -> Vec<ForeignDef> {
        self.cx.foreign_items(mod_def)
    }

    pub(crate) fn all_trait_decls(&self) -> TraitDecls {
        self.cx.all_trait_decls()
    }

    pub(crate) fn trait_decls(&self, crate_num: CrateNum) -> TraitDecls {
        self.cx.trait_decls(crate_num)
    }

    pub(crate) fn trait_decl(&self, trait_def: &TraitDef) -> TraitDecl {
        self.cx.trait_decl(trait_def)
    }

    pub(crate) fn all_trait_impls(&self) -> ImplTraitDecls {
        self.cx.all_trait_impls()
    }

    pub(crate) fn trait_impls(&self, crate_num: CrateNum) -> ImplTraitDecls {
        self.cx.trait_impls(crate_num)
    }

    pub(crate) fn trait_impl(&self, trait_impl: &ImplDef) -> ImplTrait {
        self.cx.trait_impl(trait_impl)
    }

    pub(crate) fn generics_of(&self, def_id: DefId) -> Generics {
        self.cx.generics_of(def_id)
    }

    pub(crate) fn predicates_of(&self, def_id: DefId) -> GenericPredicates {
        self.cx.predicates_of(def_id)
    }

    pub(crate) fn explicit_predicates_of(&self, def_id: DefId) -> GenericPredicates {
        self.cx.explicit_predicates_of(def_id)
    }

    /// Get information about the local crate.
    pub(crate) fn local_crate(&self) -> Crate {
        self.cx.local_crate()
    }

    /// Retrieve a list of all external crates.
    pub(crate) fn external_crates(&self) -> Vec<Crate> {
        self.cx.external_crates()
    }

    /// Find a crate with the given name.
    pub(crate) fn find_crates(&self, name: &str) -> Vec<Crate> {
        self.cx.find_crates(name)
    }

    /// Returns the name of given `DefId`.
    pub(crate) fn def_name(&self, def_id: DefId, trimmed: bool) -> Symbol {
        self.cx.def_name(def_id, trimmed)
    }

    /// Return registered tool attributes with the given attribute name.
    ///
    /// FIXME(jdonszelmann): may panic on non-tool attributes. After more attribute work, non-tool
    /// attributes will simply return an empty list.
    ///
    /// Single segmented name like `#[clippy]` is specified as `&["clippy".to_string()]`.
    /// Multi-segmented name like `#[rustfmt::skip]` is specified as `&["rustfmt".to_string(), "skip".to_string()]`.
    pub(crate) fn tool_attrs(&self, def_id: DefId, attr: &[Symbol]) -> Vec<Attribute> {
        self.cx.tool_attrs(def_id, attr)
    }

    /// Get all tool attributes of a definition.
    pub(crate) fn all_tool_attrs(&self, def_id: DefId) -> Vec<Attribute> {
        self.cx.all_tool_attrs(def_id)
    }

    /// Returns printable, human readable form of `Span`.
    pub(crate) fn span_to_string(&self, span: Span) -> String {
        self.cx.span_to_string(span)
    }

    /// Return filename from given `Span`, for diagnostic purposes.
    pub(crate) fn get_filename(&self, span: &Span) -> Filename {
        self.cx.get_filename(span)
    }

    /// Return lines corresponding to this `Span`.
    pub(crate) fn get_lines(&self, span: &Span) -> LineInfo {
        self.cx.get_lines(span)
    }

    /// Returns the `kind` of given `DefId`.
    pub(crate) fn item_kind(&self, item: CrateItem) -> ItemKind {
        self.cx.item_kind(item)
    }

    /// Returns whether this is a foreign item.
    pub(crate) fn is_foreign_item(&self, item: DefId) -> bool {
        self.cx.is_foreign_item(item)
    }

    /// Returns the kind of a given foreign item.
    pub(crate) fn foreign_item_kind(&self, def: ForeignDef) -> ForeignItemKind {
        self.cx.foreign_item_kind(def)
    }

    /// Returns the kind of a given algebraic data type.
    pub(crate) fn adt_kind(&self, def: AdtDef) -> AdtKind {
        self.cx.adt_kind(def)
    }

    /// Returns if the ADT is a box.
    pub(crate) fn adt_is_box(&self, def: AdtDef) -> bool {
        self.cx.adt_is_box(def)
    }

    /// Returns whether this ADT is simd.
    pub(crate) fn adt_is_simd(&self, def: AdtDef) -> bool {
        self.cx.adt_is_simd(def)
    }

    /// Returns whether this definition is a C string.
    pub(crate) fn adt_is_cstr(&self, def: AdtDef) -> bool {
        self.cx.adt_is_cstr(def)
    }

    /// Retrieve the function signature for the given generic arguments.
    pub(crate) fn fn_sig(&self, def: FnDef, args: &GenericArgs) -> PolyFnSig {
        self.cx.fn_sig(def, args)
    }

    /// Retrieve the intrinsic definition if the item corresponds one.
    pub(crate) fn intrinsic(&self, item: DefId) -> Option<IntrinsicDef> {
        self.cx.intrinsic(item)
    }

    /// Retrieve the plain function name of an intrinsic.
    pub(crate) fn intrinsic_name(&self, def: IntrinsicDef) -> Symbol {
        self.cx.intrinsic_name(def)
    }

    /// Retrieve the closure signature for the given generic arguments.
    pub(crate) fn closure_sig(&self, args: &GenericArgs) -> PolyFnSig {
        self.cx.closure_sig(args)
    }

    /// The number of variants in this ADT.
    pub(crate) fn adt_variants_len(&self, def: AdtDef) -> usize {
        self.cx.adt_variants_len(def)
    }

    /// The name of a variant.
    pub(crate) fn variant_name(&self, def: VariantDef) -> Symbol {
        self.cx.variant_name(def)
    }

    pub(crate) fn variant_fields(&self, def: VariantDef) -> Vec<FieldDef> {
        self.cx.variant_fields(def)
    }

    /// Evaluate constant as a target usize.
    pub(crate) fn eval_target_usize(&self, cnst: &MirConst) -> Result<u64, Error> {
        self.cx.eval_target_usize(cnst)
    }

    pub(crate) fn eval_target_usize_ty(&self, cnst: &TyConst) -> Result<u64, Error> {
        self.cx.eval_target_usize_ty(cnst)
    }

    /// Create a new zero-sized constant.
    pub(crate) fn try_new_const_zst(&self, ty: Ty) -> Result<MirConst, Error> {
        self.cx.try_new_const_zst(ty)
    }

    /// Create a new constant that represents the given string value.
    pub(crate) fn new_const_str(&self, value: &str) -> MirConst {
        self.cx.new_const_str(value)
    }

    /// Create a new constant that represents the given boolean value.
    pub(crate) fn new_const_bool(&self, value: bool) -> MirConst {
        self.cx.new_const_bool(value)
    }

    /// Create a new constant that represents the given value.
    pub(crate) fn try_new_const_uint(
        &self,
        value: u128,
        uint_ty: UintTy,
    ) -> Result<MirConst, Error> {
        self.cx.try_new_const_uint(value, uint_ty)
    }

    pub(crate) fn try_new_ty_const_uint(
        &self,
        value: u128,
        uint_ty: UintTy,
    ) -> Result<TyConst, Error> {
        self.cx.try_new_ty_const_uint(value, uint_ty)
    }

    /// Create a new type from the given kind.
    pub(crate) fn new_rigid_ty(&self, kind: RigidTy) -> Ty {
        self.cx.new_rigid_ty(kind)
    }

    /// Create a new box type, `Box<T>`, for the given inner type `T`.
    pub(crate) fn new_box_ty(&self, ty: Ty) -> Ty {
        self.cx.new_box_ty(ty)
    }

    /// Returns the type of given crate item.
    pub(crate) fn def_ty(&self, item: DefId) -> Ty {
        self.cx.def_ty(item)
    }

    /// Returns the type of given definition instantiated with the given arguments.
    pub(crate) fn def_ty_with_args(&self, item: DefId, args: &GenericArgs) -> Ty {
        self.cx.def_ty_with_args(item, args)
    }

    /// Returns literal value of a const as a string.
    pub(crate) fn mir_const_pretty(&self, cnst: &MirConst) -> String {
        self.cx.mir_const_pretty(cnst)
    }

    /// `Span` of an item.
    pub(crate) fn span_of_an_item(&self, def_id: DefId) -> Span {
        self.cx.span_of_an_item(def_id)
    }

    pub(crate) fn ty_const_pretty(&self, ct: TyConstId) -> String {
        self.cx.ty_const_pretty(ct)
    }

    /// Obtain the representation of a type.
    pub(crate) fn ty_pretty(&self, ty: Ty) -> String {
        self.cx.ty_pretty(ty)
    }

    /// Obtain the representation of a type.
    pub(crate) fn ty_kind(&self, ty: Ty) -> TyKind {
        self.cx.ty_kind(ty)
    }

    /// Get the discriminant Ty for this Ty if there's one.
    pub(crate) fn rigid_ty_discriminant_ty(&self, ty: &RigidTy) -> Ty {
        self.cx.rigid_ty_discriminant_ty(ty)
    }

    /// Get the body of an Instance which is already monomorphized.
    pub(crate) fn instance_body(&self, instance: InstanceDef) -> Option<Body> {
        self.cx.instance_body(instance)
    }

    /// Get the instance type with generic instantiations applied and lifetimes erased.
    pub(crate) fn instance_ty(&self, instance: InstanceDef) -> Ty {
        self.cx.instance_ty(instance)
    }

    /// Get the instantiation types.
    pub(crate) fn instance_args(&self, def: InstanceDef) -> GenericArgs {
        self.cx.instance_args(def)
    }

    /// Get the instance.
    pub(crate) fn instance_def_id(&self, instance: InstanceDef) -> DefId {
        self.cx.instance_def_id(instance)
    }

    /// Get the instance mangled name.
    pub(crate) fn instance_mangled_name(&self, instance: InstanceDef) -> Symbol {
        self.cx.instance_mangled_name(instance)
    }

    /// Check if this is an empty DropGlue shim.
    pub(crate) fn is_empty_drop_shim(&self, def: InstanceDef) -> bool {
        self.cx.is_empty_drop_shim(def)
    }

    /// Check if this is an empty AsyncDropGlueCtor shim.
    pub(crate) fn is_empty_async_drop_ctor_shim(&self, def: InstanceDef) -> bool {
        self.cx.is_empty_async_drop_ctor_shim(def)
    }

    /// Convert a non-generic crate item into an instance.
    /// This function will panic if the item is generic.
    pub(crate) fn mono_instance(&self, def_id: DefId) -> Instance {
        self.cx.mono_instance(def_id)
    }

    /// Item requires monomorphization.
    pub(crate) fn requires_monomorphization(&self, def_id: DefId) -> bool {
        self.cx.requires_monomorphization(def_id)
    }

    /// Resolve an instance from the given function definition and generic arguments.
    pub(crate) fn resolve_instance(&self, def: FnDef, args: &GenericArgs) -> Option<Instance> {
        self.cx.resolve_instance(def, args)
    }

    /// Resolve an instance for drop_in_place for the given type.
    pub(crate) fn resolve_drop_in_place(&self, ty: Ty) -> Instance {
        self.cx.resolve_drop_in_place(ty)
    }

    /// Resolve instance for a function pointer.
    pub(crate) fn resolve_for_fn_ptr(&self, def: FnDef, args: &GenericArgs) -> Option<Instance> {
        self.cx.resolve_for_fn_ptr(def, args)
    }

    /// Resolve instance for a closure with the requested type.
    pub(crate) fn resolve_closure(
        &self,
        def: ClosureDef,
        args: &GenericArgs,
        kind: ClosureKind,
    ) -> Option<Instance> {
        self.cx.resolve_closure(def, args, kind)
    }

    /// Evaluate a static's initializer.
    pub(crate) fn eval_static_initializer(&self, def: StaticDef) -> Result<Allocation, Error> {
        self.cx.eval_static_initializer(def)
    }

    /// Try to evaluate an instance into a constant.
    pub(crate) fn eval_instance(
        &self,
        def: InstanceDef,
        const_ty: Ty,
    ) -> Result<Allocation, Error> {
        self.cx.eval_instance(def, const_ty)
    }

    /// Retrieve global allocation for the given allocation ID.
    pub(crate) fn global_alloc(&self, id: AllocId) -> GlobalAlloc {
        self.cx.global_alloc(id)
    }

    /// Retrieve the id for the virtual table.
    pub(crate) fn vtable_allocation(&self, global_alloc: &GlobalAlloc) -> Option<AllocId> {
        self.cx.vtable_allocation(global_alloc)
    }

    pub(crate) fn krate(&self, def_id: DefId) -> Crate {
        self.cx.krate(def_id)
    }

    pub(crate) fn instance_name(&self, def: InstanceDef, trimmed: bool) -> Symbol {
        self.cx.instance_name(def, trimmed)
    }

    /// Return information about the target machine.
    pub(crate) fn target_info(&self) -> MachineInfo {
        self.cx.target_info()
    }

    /// Get an instance ABI.
    pub(crate) fn instance_abi(&self, def: InstanceDef) -> Result<FnAbi, Error> {
        self.cx.instance_abi(def)
    }

    /// Get the ABI of a function pointer.
    pub(crate) fn fn_ptr_abi(&self, fn_ptr: PolyFnSig) -> Result<FnAbi, Error> {
        self.cx.fn_ptr_abi(fn_ptr)
    }

    /// Get the layout of a type.
    pub(crate) fn ty_layout(&self, ty: Ty) -> Result<Layout, Error> {
        self.cx.ty_layout(ty)
    }

    /// Get the layout shape.
    pub(crate) fn layout_shape(&self, id: Layout) -> LayoutShape {
        self.cx.layout_shape(id)
    }

    /// Get a debug string representation of a place.
    pub(crate) fn place_pretty(&self, place: &Place) -> String {
        self.cx.place_pretty(place)
    }

    /// Get the resulting type of binary operation.
    pub(crate) fn binop_ty(&self, bin_op: BinOp, rhs: Ty, lhs: Ty) -> Ty {
        self.cx.binop_ty(bin_op, rhs, lhs)
    }

    /// Get the resulting type of unary operation.
    pub(crate) fn unop_ty(&self, un_op: UnOp, arg: Ty) -> Ty {
        self.cx.unop_ty(un_op, arg)
    }

    /// Get all associated items of a definition.
    pub(crate) fn associated_items(&self, def_id: DefId) -> AssocItems {
        self.cx.associated_items(def_id)
    }
}

// A thread local variable that stores a pointer to [`SmirInterface`].
scoped_tls::scoped_thread_local!(static TLV: Cell<*const ()>);

pub(crate) fn run<'tcx, T, F>(interface: &SmirInterface<'tcx>, f: F) -> Result<T, Error>
where
    F: FnOnce() -> T,
{
    if TLV.is_set() {
        Err(Error::from("StableMIR already running"))
    } else {
        let ptr: *const () = (interface as *const SmirInterface<'tcx>) as *const ();
        TLV.set(&Cell::new(ptr), || Ok(f()))
    }
}

/// Execute the given function with access the [`SmirInterface`].
///
/// I.e., This function will load the current interface and calls a function with it.
/// Do not nest these, as that will ICE.
pub(crate) fn with<R>(f: impl FnOnce(&SmirInterface<'_>) -> R) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        f(unsafe { &*(ptr as *const SmirInterface<'_>) })
    })
}
