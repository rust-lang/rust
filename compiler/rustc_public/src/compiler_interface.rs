//! Define the interface with the Rust compiler.
//!
//! rustc_public users should not use any of the items in this module directly.
//! These APIs have no stability guarantee.

use std::cell::Cell;

use rustc_hir::def::DefKind;
use rustc_public_bridge::context::CompilerCtxt;
use rustc_public_bridge::{Bridge, Container};
use tracing::debug;

use crate::abi::{FnAbi, Layout, LayoutShape, ReprOptions};
use crate::crate_def::Attribute;
use crate::mir::alloc::{AllocId, GlobalAlloc};
use crate::mir::mono::{Instance, InstanceDef, StaticDef};
use crate::mir::{BinOp, Body, Place, UnOp};
use crate::target::{MachineInfo, MachineSize};
use crate::ty::{
    AdtDef, AdtKind, Allocation, ClosureDef, ClosureKind, CoroutineDef, Discr, FieldDef, FnDef,
    ForeignDef, ForeignItemKind, ForeignModule, ForeignModuleDef, GenericArgs, GenericPredicates,
    Generics, ImplDef, ImplTrait, IntrinsicDef, LineInfo, MirConst, PolyFnSig, RigidTy, Span,
    TraitDecl, TraitDef, Ty, TyConst, TyConstId, TyKind, UintTy, VariantDef, VariantIdx,
};
use crate::unstable::{RustcInternal, Stable, new_item_kind};
use crate::{
    AssocItems, Crate, CrateDef, CrateItem, CrateItems, CrateNum, DefId, Error, Filename,
    ImplTraitDecls, ItemKind, Symbol, TraitDecls, alloc, mir,
};

pub struct BridgeTys;

impl Bridge for BridgeTys {
    type DefId = crate::DefId;
    type AllocId = crate::mir::alloc::AllocId;
    type Span = crate::ty::Span;
    type Ty = crate::ty::Ty;
    type InstanceDef = crate::mir::mono::InstanceDef;
    type TyConstId = crate::ty::TyConstId;
    type MirConstId = crate::ty::MirConstId;
    type Layout = crate::abi::Layout;

    type Error = crate::Error;
    type CrateItem = crate::CrateItem;
    type AdtDef = crate::ty::AdtDef;
    type ForeignModuleDef = crate::ty::ForeignModuleDef;
    type ForeignDef = crate::ty::ForeignDef;
    type FnDef = crate::ty::FnDef;
    type ClosureDef = crate::ty::ClosureDef;
    type CoroutineDef = crate::ty::CoroutineDef;
    type CoroutineClosureDef = crate::ty::CoroutineClosureDef;
    type AliasDef = crate::ty::AliasDef;
    type ParamDef = crate::ty::ParamDef;
    type BrNamedDef = crate::ty::BrNamedDef;
    type TraitDef = crate::ty::TraitDef;
    type GenericDef = crate::ty::GenericDef;
    type ConstDef = crate::ty::ConstDef;
    type ImplDef = crate::ty::ImplDef;
    type RegionDef = crate::ty::RegionDef;
    type CoroutineWitnessDef = crate::ty::CoroutineWitnessDef;
    type AssocDef = crate::ty::AssocDef;
    type OpaqueDef = crate::ty::OpaqueDef;
    type Prov = crate::ty::Prov;
    type StaticDef = crate::mir::mono::StaticDef;

    type Allocation = crate::ty::Allocation;
}

/// Public API for querying compiler information.
///
/// All queries are delegated to [`rustc_public_bridge::context::CompilerCtxt`] that provides
/// similar APIs but based on internal rustc constructs.
///
/// Do not use this directly. This is currently used in the macro expansion.
pub(crate) trait CompilerInterface {
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

    /// Returns the representation options for this ADT.
    fn adt_repr(&self, def: AdtDef) -> ReprOptions;

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

    /// Discriminant for a given variant index of AdtDef.
    fn adt_discr_for_variant(&self, adt: AdtDef, variant: VariantIdx) -> Discr;

    /// Discriminant for a given variand index and args of a coroutine.
    fn coroutine_discr_for_variant(
        &self,
        coroutine: CoroutineDef,
        args: &GenericArgs,
        variant: VariantIdx,
    ) -> Discr;

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

    /// Obtain the kind of a type.
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

impl<'tcx> CompilerInterface for Container<'tcx, BridgeTys> {
    fn entry_fn(&self) -> Option<CrateItem> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = cx.entry_fn();
        Some(tables.crate_item(did?))
    }

    /// Retrieve all items of the local crate that have a MIR associated with them.
    fn all_local_items(&self) -> CrateItems {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.all_local_items().iter().map(|did| tables.crate_item(*did)).collect()
    }

    /// Retrieve the body of a function.
    /// This function will panic if the body is not available.
    fn mir_body(&self, item: DefId) -> mir::Body {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[item];
        cx.mir_body(did).stable(&mut *tables, cx)
    }

    /// Check whether the body of a function is available.
    fn has_body(&self, item: DefId) -> bool {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def = item.internal(&mut *tables, cx.tcx);
        cx.has_body(def)
    }

    fn foreign_modules(&self, crate_num: CrateNum) -> Vec<ForeignModuleDef> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.foreign_modules(crate_num.internal(&mut *tables, cx.tcx))
            .iter()
            .map(|did| tables.foreign_module_def(*did))
            .collect()
    }

    /// Retrieve all functions defined in this crate.
    fn crate_functions(&self, crate_num: CrateNum) -> Vec<FnDef> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let krate = crate_num.internal(&mut *tables, cx.tcx);
        cx.crate_functions(krate).iter().map(|did| tables.fn_def(*did)).collect()
    }

    /// Retrieve all static items defined in this crate.
    fn crate_statics(&self, crate_num: CrateNum) -> Vec<StaticDef> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let krate = crate_num.internal(&mut *tables, cx.tcx);
        cx.crate_statics(krate).iter().map(|did| tables.static_def(*did)).collect()
    }

    fn foreign_module(&self, mod_def: ForeignModuleDef) -> ForeignModule {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[mod_def.def_id()];
        cx.foreign_module(did).stable(&mut *tables, cx)
    }

    fn foreign_items(&self, mod_def: ForeignModuleDef) -> Vec<ForeignDef> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[mod_def.def_id()];
        cx.foreign_items(did).iter().map(|did| tables.foreign_def(*did)).collect()
    }

    fn all_trait_decls(&self) -> TraitDecls {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.all_trait_decls().map(|did| tables.trait_def(did)).collect()
    }

    fn trait_decls(&self, crate_num: CrateNum) -> TraitDecls {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let krate = crate_num.internal(&mut *tables, cx.tcx);
        cx.trait_decls(krate).iter().map(|did| tables.trait_def(*did)).collect()
    }

    fn trait_decl(&self, trait_def: &TraitDef) -> TraitDecl {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[trait_def.0];
        cx.trait_decl(did).stable(&mut *tables, cx)
    }

    fn all_trait_impls(&self) -> ImplTraitDecls {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.all_trait_impls().iter().map(|did| tables.impl_def(*did)).collect()
    }

    fn trait_impls(&self, crate_num: CrateNum) -> ImplTraitDecls {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let krate = crate_num.internal(&mut *tables, cx.tcx);
        cx.trait_impls(krate).iter().map(|did| tables.impl_def(*did)).collect()
    }

    fn trait_impl(&self, trait_impl: &ImplDef) -> ImplTrait {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[trait_impl.0];
        cx.trait_impl(did).stable(&mut *tables, cx)
    }

    fn generics_of(&self, def_id: DefId) -> Generics {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        cx.generics_of(did).stable(&mut *tables, cx)
    }

    fn predicates_of(&self, def_id: DefId) -> GenericPredicates {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        let (parent, kinds) = cx.predicates_of(did);
        crate::ty::GenericPredicates {
            parent: parent.map(|did| tables.trait_def(did)),
            predicates: kinds
                .iter()
                .map(|(kind, span)| (kind.stable(&mut *tables, cx), span.stable(&mut *tables, cx)))
                .collect(),
        }
    }

    fn explicit_predicates_of(&self, def_id: DefId) -> GenericPredicates {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        let (parent, kinds) = cx.explicit_predicates_of(did);
        crate::ty::GenericPredicates {
            parent: parent.map(|did| tables.trait_def(did)),
            predicates: kinds
                .iter()
                .map(|(kind, span)| (kind.stable(&mut *tables, cx), span.stable(&mut *tables, cx)))
                .collect(),
        }
    }

    /// Get information about the local crate.
    fn local_crate(&self) -> Crate {
        let cx = &*self.cx.borrow();
        smir_crate(cx, cx.local_crate_num())
    }

    /// Retrieve a list of all external crates.
    fn external_crates(&self) -> Vec<Crate> {
        let cx = &*self.cx.borrow();
        cx.external_crates().iter().map(|crate_num| smir_crate(cx, *crate_num)).collect()
    }

    /// Find a crate with the given name.
    fn find_crates(&self, name: &str) -> Vec<Crate> {
        let cx = &*self.cx.borrow();
        cx.find_crates(name).iter().map(|crate_num| smir_crate(cx, *crate_num)).collect()
    }

    /// Returns the name of given `DefId`.
    fn def_name(&self, def_id: DefId, trimmed: bool) -> Symbol {
        let tables = self.tables.borrow();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        cx.def_name(did, trimmed)
    }

    /// Return registered tool attributes with the given attribute name.
    ///
    /// FIXME(jdonszelmann): may panic on non-tool attributes. After more attribute work, non-tool
    /// attributes will simply return an empty list.
    ///
    /// Single segmented name like `#[clippy]` is specified as `&["clippy".to_string()]`.
    /// Multi-segmented name like `#[rustfmt::skip]` is specified as `&["rustfmt".to_string(), "skip".to_string()]`.
    fn tool_attrs(&self, def_id: DefId, attr: &[Symbol]) -> Vec<Attribute> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        cx.tool_attrs(did, attr)
            .into_iter()
            .map(|(attr_str, span)| Attribute::new(attr_str, span.stable(&mut *tables, cx)))
            .collect()
    }

    /// Get all tool attributes of a definition.
    fn all_tool_attrs(&self, def_id: DefId) -> Vec<Attribute> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        cx.all_tool_attrs(did)
            .into_iter()
            .map(|(attr_str, span)| Attribute::new(attr_str, span.stable(&mut *tables, cx)))
            .collect()
    }

    /// Returns printable, human readable form of `Span`.
    fn span_to_string(&self, span: Span) -> String {
        let tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let sp = tables.spans[span];
        cx.span_to_string(sp)
    }

    /// Return filename from given `Span`, for diagnostic purposes.
    fn get_filename(&self, span: &Span) -> Filename {
        let tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let sp = tables.spans[*span];
        cx.get_filename(sp)
    }

    /// Return lines corresponding to this `Span`.
    fn get_lines(&self, span: &Span) -> LineInfo {
        let tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let sp = tables.spans[*span];
        let lines = cx.get_lines(sp);
        LineInfo::from(lines)
    }

    /// Returns the `kind` of given `DefId`.
    fn item_kind(&self, item: CrateItem) -> ItemKind {
        let tables = self.tables.borrow();
        let cx = &*self.cx.borrow();
        let did = tables[item.0];
        new_item_kind(cx.def_kind(did))
    }

    /// Returns whether this is a foreign item.
    fn is_foreign_item(&self, item: DefId) -> bool {
        let tables = self.tables.borrow();
        let cx = &*self.cx.borrow();
        let did = tables[item];
        cx.is_foreign_item(did)
    }

    /// Returns the kind of a given foreign item.
    fn foreign_item_kind(&self, def: ForeignDef) -> ForeignItemKind {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def_id = tables[def.def_id()];
        let def_kind = cx.foreign_item_kind(def_id);
        match def_kind {
            DefKind::Fn => ForeignItemKind::Fn(tables.fn_def(def_id)),
            DefKind::Static { .. } => ForeignItemKind::Static(tables.static_def(def_id)),
            DefKind::ForeignTy => {
                use rustc_public_bridge::context::TyHelpers;
                ForeignItemKind::Type(tables.intern_ty(cx.new_foreign(def_id)))
            }
            def_kind => unreachable!("Unexpected kind for a foreign item: {:?}", def_kind),
        }
    }

    /// Returns the kind of a given algebraic data type.
    fn adt_kind(&self, def: AdtDef) -> AdtKind {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.adt_kind(def.internal(&mut *tables, cx.tcx)).stable(&mut *tables, cx)
    }

    /// Returns if the ADT is a box.
    fn adt_is_box(&self, def: AdtDef) -> bool {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.adt_is_box(def.internal(&mut *tables, cx.tcx))
    }

    /// Returns whether this ADT is simd.
    fn adt_is_simd(&self, def: AdtDef) -> bool {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.adt_is_simd(def.internal(&mut *tables, cx.tcx))
    }

    /// Returns whether this definition is a C string.
    fn adt_is_cstr(&self, def: AdtDef) -> bool {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.adt_is_cstr(def.0.internal(&mut *tables, cx.tcx))
    }

    /// Returns the representation options for this ADT
    fn adt_repr(&self, def: AdtDef) -> ReprOptions {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.adt_repr(def.internal(&mut *tables, cx.tcx)).stable(&mut *tables, cx)
    }

    /// Retrieve the function signature for the given generic arguments.
    fn fn_sig(&self, def: FnDef, args: &GenericArgs) -> PolyFnSig {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def_id = def.0.internal(&mut *tables, cx.tcx);
        let args_ref = args.internal(&mut *tables, cx.tcx);
        cx.fn_sig(def_id, args_ref).stable(&mut *tables, cx)
    }

    /// Retrieve the intrinsic definition if the item corresponds one.
    fn intrinsic(&self, item: DefId) -> Option<IntrinsicDef> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def_id = item.internal(&mut *tables, cx.tcx);
        cx.intrinsic(def_id).map(|_| IntrinsicDef(item))
    }

    /// Retrieve the plain function name of an intrinsic.
    fn intrinsic_name(&self, def: IntrinsicDef) -> Symbol {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def_id = def.0.internal(&mut *tables, cx.tcx);
        cx.intrinsic_name(def_id)
    }

    /// Retrieve the closure signature for the given generic arguments.
    fn closure_sig(&self, args: &GenericArgs) -> PolyFnSig {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let args_ref = args.internal(&mut *tables, cx.tcx);
        cx.closure_sig(args_ref).stable(&mut *tables, cx)
    }

    /// The number of variants in this ADT.
    fn adt_variants_len(&self, def: AdtDef) -> usize {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.adt_variants_len(def.internal(&mut *tables, cx.tcx))
    }

    /// Discriminant for a given variant index of AdtDef.
    fn adt_discr_for_variant(&self, adt: AdtDef, variant: VariantIdx) -> Discr {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.adt_discr_for_variant(
            adt.internal(&mut *tables, cx.tcx),
            variant.internal(&mut *tables, cx.tcx),
        )
        .stable(&mut *tables, cx)
    }

    /// Discriminant for a given variand index and args of a coroutine.
    fn coroutine_discr_for_variant(
        &self,
        coroutine: CoroutineDef,
        args: &GenericArgs,
        variant: VariantIdx,
    ) -> Discr {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let tcx = cx.tcx;
        let def = coroutine.def_id().internal(&mut *tables, tcx);
        let args_ref = args.internal(&mut *tables, tcx);
        cx.coroutine_discr_for_variant(def, args_ref, variant.internal(&mut *tables, tcx))
            .stable(&mut *tables, cx)
    }

    /// The name of a variant.
    fn variant_name(&self, def: VariantDef) -> Symbol {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.variant_name(def.internal(&mut *tables, cx.tcx))
    }

    fn variant_fields(&self, def: VariantDef) -> Vec<FieldDef> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        def.internal(&mut *tables, cx.tcx)
            .fields
            .iter()
            .map(|f| f.stable(&mut *tables, cx))
            .collect()
    }

    /// Evaluate constant as a target usize.
    fn eval_target_usize(&self, mir_const: &MirConst) -> Result<u64, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let cnst = mir_const.internal(&mut *tables, cx.tcx);
        cx.eval_target_usize(cnst)
    }

    fn eval_target_usize_ty(&self, ty_const: &TyConst) -> Result<u64, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let cnst = ty_const.internal(&mut *tables, cx.tcx);
        cx.eval_target_usize_ty(cnst)
    }

    /// Create a new zero-sized constant.
    fn try_new_const_zst(&self, ty: Ty) -> Result<MirConst, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let ty_internal = ty.internal(&mut *tables, cx.tcx);
        cx.try_new_const_zst(ty_internal).map(|cnst| cnst.stable(&mut *tables, cx))
    }

    /// Create a new constant that represents the given string value.
    fn new_const_str(&self, value: &str) -> MirConst {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.new_const_str(value).stable(&mut *tables, cx)
    }

    /// Create a new constant that represents the given boolean value.
    fn new_const_bool(&self, value: bool) -> MirConst {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.new_const_bool(value).stable(&mut *tables, cx)
    }

    /// Create a new constant that represents the given value.
    fn try_new_const_uint(&self, value: u128, uint_ty: UintTy) -> Result<MirConst, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let ty = cx.ty_new_uint(uint_ty.internal(&mut *tables, cx.tcx));
        cx.try_new_const_uint(value, ty).map(|cnst| cnst.stable(&mut *tables, cx))
    }

    fn try_new_ty_const_uint(&self, value: u128, uint_ty: UintTy) -> Result<TyConst, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let ty = cx.ty_new_uint(uint_ty.internal(&mut *tables, cx.tcx));
        cx.try_new_ty_const_uint(value, ty).map(|cnst| cnst.stable(&mut *tables, cx))
    }

    /// Create a new type from the given kind.
    fn new_rigid_ty(&self, kind: RigidTy) -> Ty {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let internal_kind = kind.internal(&mut *tables, cx.tcx);
        cx.new_rigid_ty(internal_kind).stable(&mut *tables, cx)
    }

    /// Create a new box type, `Box<T>`, for the given inner type `T`.
    fn new_box_ty(&self, ty: Ty) -> Ty {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let inner = ty.internal(&mut *tables, cx.tcx);
        cx.new_box_ty(inner).stable(&mut *tables, cx)
    }

    /// Returns the type of given crate item.
    fn def_ty(&self, item: DefId) -> Ty {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let inner = item.internal(&mut *tables, cx.tcx);
        cx.def_ty(inner).stable(&mut *tables, cx)
    }

    /// Returns the type of given definition instantiated with the given arguments.
    fn def_ty_with_args(&self, item: DefId, args: &GenericArgs) -> Ty {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let inner = item.internal(&mut *tables, cx.tcx);
        let args_ref = args.internal(&mut *tables, cx.tcx);
        cx.def_ty_with_args(inner, args_ref).stable(&mut *tables, cx)
    }

    /// Returns literal value of a const as a string.
    fn mir_const_pretty(&self, cnst: &MirConst) -> String {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cnst.internal(&mut *tables, cx.tcx).to_string()
    }

    /// `Span` of an item.
    fn span_of_an_item(&self, def_id: DefId) -> Span {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        cx.span_of_an_item(did).stable(&mut *tables, cx)
    }

    fn ty_const_pretty(&self, ct: TyConstId) -> String {
        let tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.ty_const_pretty(tables.ty_consts[ct])
    }

    /// Obtain the representation of a type.
    fn ty_pretty(&self, ty: Ty) -> String {
        let tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.ty_pretty(tables.types[ty])
    }

    /// Obtain the kind of a type.
    fn ty_kind(&self, ty: Ty) -> TyKind {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        cx.ty_kind(tables.types[ty]).stable(&mut *tables, cx)
    }

    /// Get the discriminant Ty for this Ty if there's one.
    fn rigid_ty_discriminant_ty(&self, ty: &RigidTy) -> Ty {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let internal_kind = ty.internal(&mut *tables, cx.tcx);
        cx.rigid_ty_discriminant_ty(internal_kind).stable(&mut *tables, cx)
    }

    /// Get the body of an Instance which is already monomorphized.
    fn instance_body(&self, instance: InstanceDef) -> Option<Body> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let instance = tables.instances[instance];
        cx.instance_body(instance).map(|body| body.stable(&mut *tables, cx))
    }

    /// Get the instance type with generic instantiations applied and lifetimes erased.
    fn instance_ty(&self, instance: InstanceDef) -> Ty {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let instance = tables.instances[instance];
        cx.instance_ty(instance).stable(&mut *tables, cx)
    }

    /// Get the instantiation types.
    fn instance_args(&self, def: InstanceDef) -> GenericArgs {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let instance = tables.instances[def];
        cx.instance_args(instance).stable(&mut *tables, cx)
    }

    /// Get the instance.
    fn instance_def_id(&self, instance: InstanceDef) -> DefId {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let instance = tables.instances[instance];
        cx.instance_def_id(instance, &mut *tables)
    }

    /// Get the instance mangled name.
    fn instance_mangled_name(&self, instance: InstanceDef) -> Symbol {
        let tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let instance = tables.instances[instance];
        cx.instance_mangled_name(instance)
    }

    /// Check if this is an empty DropGlue shim.
    fn is_empty_drop_shim(&self, def: InstanceDef) -> bool {
        let tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let instance = tables.instances[def];
        cx.is_empty_drop_shim(instance)
    }

    /// Convert a non-generic crate item into an instance.
    /// This function will panic if the item is generic.
    fn mono_instance(&self, def_id: DefId) -> Instance {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        cx.mono_instance(did).stable(&mut *tables, cx)
    }

    /// Item requires monomorphization.
    fn requires_monomorphization(&self, def_id: DefId) -> bool {
        let tables = self.tables.borrow();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        cx.requires_monomorphization(did)
    }

    /// Resolve an instance from the given function definition and generic arguments.
    fn resolve_instance(&self, def: FnDef, args: &GenericArgs) -> Option<Instance> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def_id = def.0.internal(&mut *tables, cx.tcx);
        let args_ref = args.internal(&mut *tables, cx.tcx);
        cx.resolve_instance(def_id, args_ref).map(|inst| inst.stable(&mut *tables, cx))
    }

    /// Resolve an instance for drop_in_place for the given type.
    fn resolve_drop_in_place(&self, ty: Ty) -> Instance {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let internal_ty = ty.internal(&mut *tables, cx.tcx);

        cx.resolve_drop_in_place(internal_ty).stable(&mut *tables, cx)
    }

    /// Resolve instance for a function pointer.
    fn resolve_for_fn_ptr(&self, def: FnDef, args: &GenericArgs) -> Option<Instance> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def_id = def.0.internal(&mut *tables, cx.tcx);
        let args_ref = args.internal(&mut *tables, cx.tcx);
        cx.resolve_for_fn_ptr(def_id, args_ref).stable(&mut *tables, cx)
    }

    /// Resolve instance for a closure with the requested type.
    fn resolve_closure(
        &self,
        def: ClosureDef,
        args: &GenericArgs,
        kind: ClosureKind,
    ) -> Option<Instance> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def_id = def.0.internal(&mut *tables, cx.tcx);
        let args_ref = args.internal(&mut *tables, cx.tcx);
        let closure_kind = kind.internal(&mut *tables, cx.tcx);
        cx.resolve_closure(def_id, args_ref, closure_kind).map(|inst| inst.stable(&mut *tables, cx))
    }

    /// Evaluate a static's initializer.
    fn eval_static_initializer(&self, def: StaticDef) -> Result<Allocation, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let def_id = def.0.internal(&mut *tables, cx.tcx);

        cx.eval_static_initializer(def_id).stable(&mut *tables, cx)
    }

    /// Try to evaluate an instance into a constant.
    fn eval_instance(&self, def: InstanceDef, const_ty: Ty) -> Result<Allocation, Error> {
        let mut tables = self.tables.borrow_mut();
        let instance = tables.instances[def];
        let cx = &*self.cx.borrow();
        let const_ty = const_ty.internal(&mut *tables, cx.tcx);
        cx.eval_instance(instance)
            .map(|const_val| alloc::try_new_allocation(const_ty, const_val, &mut *tables, cx))
            .map_err(|e| e.stable(&mut *tables, cx))?
    }

    /// Retrieve global allocation for the given allocation ID.
    fn global_alloc(&self, id: AllocId) -> GlobalAlloc {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let alloc_id = id.internal(&mut *tables, cx.tcx);
        cx.global_alloc(alloc_id).stable(&mut *tables, cx)
    }

    /// Retrieve the id for the virtual table.
    fn vtable_allocation(&self, global_alloc: &GlobalAlloc) -> Option<AllocId> {
        let mut tables = self.tables.borrow_mut();
        let GlobalAlloc::VTable(ty, trait_ref) = global_alloc else {
            return None;
        };
        let cx = &*self.cx.borrow();
        let ty = ty.internal(&mut *tables, cx.tcx);
        let trait_ref = trait_ref.internal(&mut *tables, cx.tcx);
        let alloc_id = cx.vtable_allocation(ty, trait_ref);
        Some(alloc_id.stable(&mut *tables, cx))
    }

    fn krate(&self, def_id: DefId) -> Crate {
        let tables = self.tables.borrow();
        let cx = &*self.cx.borrow();
        smir_crate(cx, tables[def_id].krate)
    }

    fn instance_name(&self, def: InstanceDef, trimmed: bool) -> Symbol {
        let tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let instance = tables.instances[def];
        cx.instance_name(instance, trimmed)
    }

    /// Return information about the target machine.
    fn target_info(&self) -> MachineInfo {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        MachineInfo {
            endian: cx.target_endian().stable(&mut *tables, cx),
            pointer_width: MachineSize::from_bits(cx.target_pointer_size()),
        }
    }

    /// Get an instance ABI.
    fn instance_abi(&self, def: InstanceDef) -> Result<FnAbi, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let instance = tables.instances[def];
        cx.instance_abi(instance).map(|fn_abi| fn_abi.stable(&mut *tables, cx))
    }

    /// Get the ABI of a function pointer.
    fn fn_ptr_abi(&self, fn_ptr: PolyFnSig) -> Result<FnAbi, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let sig = fn_ptr.internal(&mut *tables, cx.tcx);
        cx.fn_ptr_abi(sig).map(|fn_abi| fn_abi.stable(&mut *tables, cx))
    }

    /// Get the layout of a type.
    fn ty_layout(&self, ty: Ty) -> Result<Layout, Error> {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let internal_ty = ty.internal(&mut *tables, cx.tcx);
        cx.ty_layout(internal_ty).map(|layout| layout.stable(&mut *tables, cx))
    }

    /// Get the layout shape.
    fn layout_shape(&self, id: Layout) -> LayoutShape {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        id.internal(&mut *tables, cx.tcx).0.stable(&mut *tables, cx)
    }

    /// Get a debug string representation of a place.
    fn place_pretty(&self, place: &Place) -> String {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();

        format!("{:?}", place.internal(&mut *tables, cx.tcx))
    }

    /// Get the resulting type of binary operation.
    fn binop_ty(&self, bin_op: BinOp, rhs: Ty, lhs: Ty) -> Ty {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let rhs_internal = rhs.internal(&mut *tables, cx.tcx);
        let lhs_internal = lhs.internal(&mut *tables, cx.tcx);
        let bin_op_internal = bin_op.internal(&mut *tables, cx.tcx);
        cx.binop_ty(bin_op_internal, rhs_internal, lhs_internal).stable(&mut *tables, cx)
    }

    /// Get the resulting type of unary operation.
    fn unop_ty(&self, un_op: UnOp, arg: Ty) -> Ty {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let un_op = un_op.internal(&mut *tables, cx.tcx);
        let arg = arg.internal(&mut *tables, cx.tcx);
        cx.unop_ty(un_op, arg).stable(&mut *tables, cx)
    }

    /// Get all associated items of a definition.
    fn associated_items(&self, def_id: DefId) -> AssocItems {
        let mut tables = self.tables.borrow_mut();
        let cx = &*self.cx.borrow();
        let did = tables[def_id];
        cx.associated_items(did).iter().map(|assoc| assoc.stable(&mut *tables, cx)).collect()
    }
}

// A thread local variable that stores a pointer to [`CompilerInterface`].
scoped_tls::scoped_thread_local!(static TLV: Cell<*const ()>);

pub(crate) fn run<F, T>(interface: &dyn CompilerInterface, f: F) -> Result<T, Error>
where
    F: FnOnce() -> T,
{
    if TLV.is_set() {
        Err(Error::from("rustc_public already running"))
    } else {
        let ptr: *const () = (&raw const interface) as _;
        TLV.set(&Cell::new(ptr), || Ok(f()))
    }
}

/// Execute the given function with access the [`CompilerInterface`].
///
/// I.e., This function will load the current interface and calls a function with it.
/// Do not nest these, as that will ICE.
pub(crate) fn with<R>(f: impl FnOnce(&dyn CompilerInterface) -> R) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        f(unsafe { *(ptr as *const &dyn CompilerInterface) })
    })
}

fn smir_crate<'tcx>(
    cx: &CompilerCtxt<'tcx, BridgeTys>,
    crate_num: rustc_span::def_id::CrateNum,
) -> Crate {
    let name = cx.crate_name(crate_num);
    let is_local = cx.crate_is_local(crate_num);
    let id = cx.crate_num_id(crate_num);
    debug!(?name, ?crate_num, "smir_crate");
    Crate { id, name, is_local }
}
