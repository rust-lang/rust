//! Define the interface with the Rust compiler.
//!
//! rustc_public users should not use any of the items in this module directly.
//! These APIs have no stability guarantee.

use std::cell::{Cell, RefCell};

use rustc_hir::def::DefKind;
use rustc_public_bridge::context::CompilerCtxt;
use rustc_public_bridge::{Bridge, Tables};
use tracing::debug;

use crate::abi::{FnAbi, Layout, LayoutShape, ReprOptions};
use crate::crate_def::Attribute;
use crate::mir::alloc::{AllocId, GlobalAlloc};
use crate::mir::mono::{Instance, InstanceDef, StaticDef};
use crate::mir::{BinOp, Body, Place, UnOp};
use crate::target::{MachineInfo, MachineSize};
use crate::ty::{
    AdtDef, AdtKind, Allocation, AssocItem, Asyncness, ClosureDef, ClosureKind, Constness,
    CoroutineDef, Discr, FieldDef, FnDef, ForeignDef, ForeignItemKind, ForeignModule,
    ForeignModuleDef, GenericArgs, GenericPredicates, Generics, ImplDef, ImplTrait, IntrinsicDef,
    LineInfo, MirConst, PolyFnSig, RigidTy, Span, TraitDecl, TraitDef, TraitRef, Ty, TyConst,
    TyConstId, TyKind, UintTy, VariantDef, VariantIdx, VtblEntry,
};
use crate::unstable::{RustcInternal, Stable, new_item_kind};
use crate::{
    AssocItems, Crate, CrateDef, CrateItem, CrateItems, CrateNum, DefId, Error, Filename,
    ImplTraitDecls, ItemKind, Symbol, ThreadLocalIndex, TraitDecls, alloc, mir,
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
/// All queries are delegated to [`rustc_public_bridge::context::CompilerCtxt`]
/// that provides similar APIs but based on internal rustc constructs.
///
/// Do not use this directly. This is currently used in the macro expansion.
pub(crate) struct CompilerInterface<'tcx> {
    pub tables: RefCell<Tables<'tcx, BridgeTys>>,
    pub cx: RefCell<CompilerCtxt<'tcx, BridgeTys>>,
}

impl<'tcx> CompilerInterface<'tcx> {
    fn with_cx<R>(
        &self,
        f: impl FnOnce(&mut Tables<'tcx, BridgeTys>, &CompilerCtxt<'tcx, BridgeTys>) -> R,
    ) -> R {
        let mut tables = self.tables.borrow_mut();
        let cx = self.cx.borrow();
        f(&mut *tables, &*cx)
    }

    pub(crate) fn entry_fn(&self) -> Option<CrateItem> {
        self.with_cx(|tables, cx| {
            let did = cx.entry_fn();
            Some(tables.crate_item(did?))
        })
    }

    /// Retrieve all items of the local crate that have a MIR associated with them.
    pub(crate) fn all_local_items(&self) -> CrateItems {
        self.with_cx(|tables, cx| {
            cx.all_local_items().iter().map(|did| tables.crate_item(*did)).collect()
        })
    }

    /// Retrieve the body of a function.
    /// This function will panic if the body is not available.
    pub(crate) fn mir_body(&self, item: DefId) -> mir::Body {
        self.with_cx(|tables, cx| {
            let did = tables[item];
            cx.mir_body(did).stable(tables, cx)
        })
    }

    /// Check whether the body of a function is available.
    pub(crate) fn has_body(&self, item: DefId) -> bool {
        self.with_cx(|tables, cx| {
            let def = item.internal(tables, cx.tcx);
            cx.has_body(def)
        })
    }

    pub(crate) fn foreign_modules(&self, crate_num: CrateNum) -> Vec<ForeignModuleDef> {
        self.with_cx(|tables, cx| {
            cx.foreign_modules(crate_num.internal(tables, cx.tcx))
                .iter()
                .map(|did| tables.foreign_module_def(*did))
                .collect()
        })
    }

    /// Retrieve all functions defined in this crate.
    pub(crate) fn crate_functions(&self, crate_num: CrateNum) -> Vec<FnDef> {
        self.with_cx(|tables, cx| {
            let krate = crate_num.internal(tables, cx.tcx);
            cx.crate_functions(krate).iter().map(|did| tables.fn_def(*did)).collect()
        })
    }

    pub(crate) fn crate_adts(&self, crate_num: CrateNum) -> Vec<AdtDef> {
        self.with_cx(|tables, cx| {
            let krate = crate_num.internal(tables, cx.tcx);
            cx.crate_adts(krate).iter().map(|did| tables.adt_def(*did)).collect()
        })
    }

    /// Retrieve all static items defined in this crate.
    pub(crate) fn crate_statics(&self, crate_num: CrateNum) -> Vec<StaticDef> {
        self.with_cx(|tables, cx| {
            let krate = crate_num.internal(tables, cx.tcx);
            cx.crate_statics(krate).iter().map(|did| tables.static_def(*did)).collect()
        })
    }

    pub(crate) fn foreign_module(&self, mod_def: ForeignModuleDef) -> ForeignModule {
        self.with_cx(|tables, cx| {
            let did = tables[mod_def.def_id()];
            cx.foreign_module(did).stable(tables, cx)
        })
    }

    pub(crate) fn foreign_items(&self, mod_def: ForeignModuleDef) -> Vec<ForeignDef> {
        self.with_cx(|tables, cx| {
            let did = tables[mod_def.def_id()];
            cx.foreign_items(did).iter().map(|did| tables.foreign_def(*did)).collect()
        })
    }

    pub(crate) fn all_trait_decls(&self) -> TraitDecls {
        self.with_cx(|tables, cx| cx.all_trait_decls().map(|did| tables.trait_def(did)).collect())
    }

    pub(crate) fn trait_decls(&self, crate_num: CrateNum) -> TraitDecls {
        self.with_cx(|tables, cx| {
            let krate = crate_num.internal(tables, cx.tcx);
            cx.trait_decls(krate).iter().map(|did| tables.trait_def(*did)).collect()
        })
    }

    pub(crate) fn trait_decl(&self, trait_def: &TraitDef) -> TraitDecl {
        self.with_cx(|tables, cx| {
            let did = tables[trait_def.0];
            cx.trait_decl(did).stable(tables, cx)
        })
    }

    pub(crate) fn all_trait_impls(&self) -> ImplTraitDecls {
        self.with_cx(|tables, cx| {
            cx.all_trait_impls().iter().map(|did| tables.impl_def(*did)).collect()
        })
    }

    pub(crate) fn trait_impls(&self, crate_num: CrateNum) -> ImplTraitDecls {
        self.with_cx(|tables, cx| {
            let krate = crate_num.internal(tables, cx.tcx);
            cx.trait_impls(krate).iter().map(|did| tables.impl_def(*did)).collect()
        })
    }

    pub(crate) fn trait_impl(&self, trait_impl: &ImplDef) -> ImplTrait {
        self.with_cx(|tables, cx| {
            let did = tables[trait_impl.0];
            cx.trait_impl(did).stable(tables, cx)
        })
    }

    pub(crate) fn generics_of(&self, def_id: DefId) -> Generics {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.generics_of(did).stable(tables, cx)
        })
    }

    pub(crate) fn inherent_impls(&self, adt: AdtDef) -> Vec<ImplDef> {
        self.with_cx(|tables, cx| {
            let def_id = tables[adt.0];
            cx.tcx.inherent_impls(def_id).iter().map(|&did| tables.impl_def(did)).collect()
        })
    }

    pub(crate) fn predicates_of(&self, def_id: DefId) -> GenericPredicates {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            let (parent, kinds) = cx.predicates_of(did);
            crate::ty::GenericPredicates {
                parent: parent.map(|did| tables.trait_def(did)),
                predicates: kinds
                    .iter()
                    .map(|(kind, span)| (kind.stable(tables, cx), span.stable(tables, cx)))
                    .collect(),
            }
        })
    }

    pub(crate) fn explicit_predicates_of(&self, def_id: DefId) -> GenericPredicates {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            let (parent, kinds) = cx.explicit_predicates_of(did);
            crate::ty::GenericPredicates {
                parent: parent.map(|did| tables.trait_def(did)),
                predicates: kinds
                    .iter()
                    .map(|(kind, span)| (kind.stable(tables, cx), span.stable(tables, cx)))
                    .collect(),
            }
        })
    }

    /// Get information about the local crate.
    pub(crate) fn local_crate(&self) -> Crate {
        self.with_cx(|_, cx| smir_crate(cx, cx.local_crate_num()))
    }

    /// Retrieve a list of all external crates.
    pub(crate) fn external_crates(&self) -> Vec<Crate> {
        self.with_cx(|_, cx| {
            cx.external_crates().iter().map(|crate_num| smir_crate(cx, *crate_num)).collect()
        })
    }

    /// Find a crate with the given name.
    pub(crate) fn find_crates(&self, name: &str) -> Vec<Crate> {
        self.with_cx(|_, cx| {
            cx.find_crates(name).iter().map(|crate_num| smir_crate(cx, *crate_num)).collect()
        })
    }

    /// Returns the name of given `DefId`.
    pub(crate) fn def_name(&self, def_id: DefId, trimmed: bool) -> Symbol {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.def_name(did, trimmed)
        })
    }

    /// Returns the parent of the given `DefId`.
    pub(crate) fn def_parent(&self, def_id: DefId) -> Option<DefId> {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.def_parent(did).map(|did| tables.create_def_id(did))
        })
    }

    /// Return registered tool attributes with the given attribute name.
    ///
    /// FIXME(jdonszelmann): may panic on non-tool attributes. After more attribute work, non-tool
    /// attributes will simply return an empty list.
    ///
    /// Single segmented name like `#[clippy]` is specified as `&["clippy".to_string()]`.
    /// Multi-segmented name like `#[rustfmt::skip]` is specified as `&["rustfmt".to_string(), "skip".to_string()]`.
    pub(crate) fn tool_attrs(&self, def_id: DefId, attr: &[Symbol]) -> Vec<Attribute> {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.tool_attrs(did, attr)
                .into_iter()
                .map(|(attr_str, span)| Attribute::new(attr_str, span.stable(tables, cx)))
                .collect()
        })
    }

    /// Get all tool attributes of a definition.
    pub(crate) fn all_tool_attrs(&self, def_id: DefId) -> Vec<Attribute> {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.all_tool_attrs(did)
                .into_iter()
                .map(|(attr_str, span)| Attribute::new(attr_str, span.stable(tables, cx)))
                .collect()
        })
    }

    /// Returns printable, human readable form of `Span`.
    pub(crate) fn span_to_string(&self, span: Span) -> String {
        self.with_cx(|tables, cx| {
            let sp = tables.spans[span];
            cx.span_to_string(sp)
        })
    }

    /// Return filename from given `Span`, for diagnostic purposes.
    pub(crate) fn get_filename(&self, span: &Span) -> Filename {
        self.with_cx(|tables, cx| {
            let sp = tables.spans[*span];
            cx.get_filename(sp)
        })
    }

    /// Return lines corresponding to this `Span`.
    pub(crate) fn get_lines(&self, span: &Span) -> LineInfo {
        self.with_cx(|tables, cx| {
            let sp = tables.spans[*span];
            let lines = cx.get_lines(sp);
            LineInfo::from(lines)
        })
    }

    /// Returns the `kind` of given `DefId`.
    pub(crate) fn item_kind(&self, item: CrateItem) -> ItemKind {
        self.with_cx(|tables, cx| {
            let did = tables[item.0];
            new_item_kind(cx.def_kind(did))
        })
    }

    /// Returns whether this is a foreign item.
    pub(crate) fn is_foreign_item(&self, item: DefId) -> bool {
        self.with_cx(|tables, cx| {
            let did = tables[item];
            cx.is_foreign_item(did)
        })
    }

    /// Returns the kind of a given foreign item.
    pub(crate) fn foreign_item_kind(&self, def: ForeignDef) -> ForeignItemKind {
        self.with_cx(|tables, cx| {
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
        })
    }

    /// Returns the kind of a given algebraic data type.
    pub(crate) fn adt_kind(&self, def: AdtDef) -> AdtKind {
        self.with_cx(|tables, cx| cx.adt_kind(def.internal(tables, cx.tcx)).stable(tables, cx))
    }

    /// Returns if the ADT is a box.
    pub(crate) fn adt_is_box(&self, def: AdtDef) -> bool {
        self.with_cx(|tables, cx| cx.adt_is_box(def.internal(tables, cx.tcx)))
    }

    /// Returns whether this ADT is simd.
    pub(crate) fn adt_is_simd(&self, def: AdtDef) -> bool {
        self.with_cx(|tables, cx| cx.adt_is_simd(def.internal(tables, cx.tcx)))
    }

    /// Returns whether this definition is a C string.
    pub(crate) fn adt_is_cstr(&self, def: AdtDef) -> bool {
        self.with_cx(|tables, cx| cx.adt_is_cstr(def.0.internal(tables, cx.tcx)))
    }

    /// Returns the representation options for this ADT
    pub(crate) fn adt_repr(&self, def: AdtDef) -> ReprOptions {
        self.with_cx(|tables, cx| cx.adt_repr(def.internal(tables, cx.tcx)).stable(tables, cx))
    }

    /// Retrieve the function signature for the given generic arguments.
    pub(crate) fn fn_sig(&self, def: FnDef, args: &GenericArgs) -> PolyFnSig {
        self.with_cx(|tables, cx| {
            let def_id = def.0.internal(tables, cx.tcx);
            let args_ref = args.internal(tables, cx.tcx);
            cx.fn_sig(def_id, args_ref).stable(tables, cx)
        })
    }

    /// Retrieve the constness for the given function definition.
    pub(crate) fn constness(&self, def: FnDef) -> Constness {
        self.with_cx(|tables, cx| {
            let def_id = def.0.internal(tables, cx.tcx);
            cx.constness(def_id).stable(tables, cx)
        })
    }

    /// Retrieve the asyncness for the given function definition.
    pub(crate) fn asyncness(&self, def: FnDef) -> Asyncness {
        self.with_cx(|tables, cx| {
            let def_id = def.0.internal(tables, cx.tcx);
            cx.asyncness(def_id).stable(tables, cx)
        })
    }

    /// Retrieve the intrinsic definition if the item corresponds one.
    pub(crate) fn intrinsic(&self, item: DefId) -> Option<IntrinsicDef> {
        self.with_cx(|tables, cx| {
            let def_id = item.internal(tables, cx.tcx);
            cx.intrinsic(def_id).map(|_| IntrinsicDef(item))
        })
    }

    /// Retrieve the plain function name of an intrinsic.
    pub(crate) fn intrinsic_name(&self, def: IntrinsicDef) -> Symbol {
        self.with_cx(|tables, cx| {
            let def_id = def.0.internal(tables, cx.tcx);
            cx.intrinsic_name(def_id)
        })
    }

    /// Retrieve the closure signature for the given generic arguments.
    pub(crate) fn closure_sig(&self, args: &GenericArgs) -> PolyFnSig {
        self.with_cx(|tables, cx| {
            let args_ref = args.internal(tables, cx.tcx);
            cx.closure_sig(args_ref).stable(tables, cx)
        })
    }

    /// The number of variants in this ADT.
    pub(crate) fn adt_variants_len(&self, def: AdtDef) -> usize {
        self.with_cx(|tables, cx| cx.adt_variants_len(def.internal(tables, cx.tcx)))
    }

    /// Discriminant for a given variant index of AdtDef.
    pub(crate) fn adt_discr_for_variant(&self, adt: AdtDef, variant: VariantIdx) -> Discr {
        self.with_cx(|tables, cx| {
            cx.adt_discr_for_variant(adt.internal(tables, cx.tcx), variant.internal(tables, cx.tcx))
                .stable(tables, cx)
        })
    }

    /// Discriminant for a given variand index and args of a coroutine.
    pub(crate) fn coroutine_discr_for_variant(
        &self,
        coroutine: CoroutineDef,
        args: &GenericArgs,
        variant: VariantIdx,
    ) -> Discr {
        self.with_cx(|tables, cx| {
            let tcx = cx.tcx;
            let def = coroutine.def_id().internal(tables, tcx);
            let args_ref = args.internal(tables, tcx);
            cx.coroutine_discr_for_variant(def, args_ref, variant.internal(tables, tcx))
                .stable(tables, cx)
        })
    }

    /// The name of a variant.
    pub(crate) fn variant_name(&self, def: VariantDef) -> Symbol {
        self.with_cx(|tables, cx| cx.variant_name(def.internal(tables, cx.tcx)))
    }

    pub(crate) fn variant_fields(&self, def: VariantDef) -> Vec<FieldDef> {
        self.with_cx(|tables, cx| {
            def.internal(tables, cx.tcx).fields.iter().map(|f| f.stable(tables, cx)).collect()
        })
    }

    /// Evaluate constant as a target usize.
    pub(crate) fn eval_target_usize(&self, mir_const: &MirConst) -> Result<u64, Error> {
        self.with_cx(|tables, cx| {
            let cnst = mir_const.internal(tables, cx.tcx);
            cx.eval_target_usize(cnst)
        })
    }

    pub(crate) fn eval_target_usize_ty(&self, ty_const: &TyConst) -> Result<u64, Error> {
        self.with_cx(|tables, cx| {
            let cnst = ty_const.internal(tables, cx.tcx);
            cx.eval_target_usize_ty(cnst)
        })
    }

    /// Create a new zero-sized constant.
    pub(crate) fn try_new_const_zst(&self, ty: Ty) -> Result<MirConst, Error> {
        self.with_cx(|tables, cx| {
            let ty_internal = ty.internal(tables, cx.tcx);
            cx.try_new_const_zst(ty_internal).map(|cnst| cnst.stable(tables, cx))
        })
    }

    /// Create a new constant that represents the given string value.
    pub(crate) fn new_const_str(&self, value: &str) -> MirConst {
        self.with_cx(|tables, cx| cx.new_const_str(value).stable(tables, cx))
    }

    /// Create a new constant that represents the given boolean value.
    pub(crate) fn new_const_bool(&self, value: bool) -> MirConst {
        self.with_cx(|tables, cx| cx.new_const_bool(value).stable(tables, cx))
    }

    /// Create a new constant that represents the given value.
    pub(crate) fn try_new_const_uint(
        &self,
        value: u128,
        uint_ty: UintTy,
    ) -> Result<MirConst, Error> {
        self.with_cx(|tables, cx| {
            let ty = cx.ty_new_uint(uint_ty.internal(tables, cx.tcx));
            cx.try_new_const_uint(value, ty).map(|cnst| cnst.stable(tables, cx))
        })
    }

    pub(crate) fn try_new_ty_const_uint(
        &self,
        value: u128,
        uint_ty: UintTy,
    ) -> Result<TyConst, Error> {
        self.with_cx(|tables, cx| {
            let ty = cx.ty_new_uint(uint_ty.internal(tables, cx.tcx));
            cx.try_new_ty_const_uint(value, ty).map(|cnst| cnst.stable(tables, cx))
        })
    }

    /// Create a new type from the given kind.
    pub(crate) fn new_rigid_ty(&self, kind: RigidTy) -> Ty {
        self.with_cx(|tables, cx| {
            let internal_kind = kind.internal(tables, cx.tcx);
            cx.new_rigid_ty(internal_kind).stable(tables, cx)
        })
    }

    /// Create a new box type, `Box<T>`, for the given inner type `T`.
    pub(crate) fn new_box_ty(&self, ty: Ty) -> Ty {
        self.with_cx(|tables, cx| {
            let inner = ty.internal(tables, cx.tcx);
            cx.new_box_ty(inner).stable(tables, cx)
        })
    }

    /// Returns the type of given crate item.
    pub(crate) fn def_ty(&self, item: DefId) -> Ty {
        self.with_cx(|tables, cx| {
            let inner = item.internal(tables, cx.tcx);
            cx.def_ty(inner).stable(tables, cx)
        })
    }

    /// Returns the type of given definition instantiated with the given arguments.
    pub(crate) fn def_ty_with_args(&self, item: DefId, args: &GenericArgs) -> Ty {
        self.with_cx(|tables, cx| {
            let inner = item.internal(tables, cx.tcx);
            let args_ref = args.internal(tables, cx.tcx);
            cx.def_ty_with_args(inner, args_ref).stable(tables, cx)
        })
    }

    /// Returns literal value of a const as a string.
    pub(crate) fn mir_const_pretty(&self, cnst: &MirConst) -> String {
        self.with_cx(|tables, cx| cnst.internal(tables, cx.tcx).to_string())
    }

    /// `Span` of a `DefId`.
    pub(crate) fn span_of_a_def(&self, def_id: DefId) -> Span {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.span_of_a_def(did).stable(tables, cx)
        })
    }

    pub(crate) fn ty_const_pretty(&self, ct: TyConstId) -> String {
        self.with_cx(|tables, cx| cx.ty_const_pretty(tables.ty_consts[ct]))
    }

    /// Obtain the representation of a type.
    pub(crate) fn ty_pretty(&self, ty: Ty) -> String {
        self.with_cx(|tables, cx| cx.ty_pretty(tables.types[ty]))
    }

    /// Obtain the kind of a type.
    pub(crate) fn ty_kind(&self, ty: Ty) -> TyKind {
        self.with_cx(|tables, cx| cx.ty_kind(tables.types[ty]).stable(tables, cx))
    }

    /// Get the discriminant Ty for this Ty if there's one.
    pub(crate) fn rigid_ty_discriminant_ty(&self, ty: &RigidTy) -> Ty {
        self.with_cx(|tables, cx| {
            let internal_kind = ty.internal(tables, cx.tcx);
            cx.rigid_ty_discriminant_ty(internal_kind).stable(tables, cx)
        })
    }

    /// Get the body of an Instance which is already monomorphized.
    pub(crate) fn instance_body(&self, instance: InstanceDef) -> Option<Body> {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[instance];
            cx.instance_body(instance).map(|body| body.stable(tables, cx))
        })
    }

    /// Get the instance type with generic instantiations applied and lifetimes erased.
    pub(crate) fn instance_ty(&self, instance: InstanceDef) -> Ty {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[instance];
            cx.instance_ty(instance).stable(tables, cx)
        })
    }

    /// Get the instantiation types.
    pub(crate) fn instance_args(&self, def: InstanceDef) -> GenericArgs {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[def];
            cx.instance_args(instance).stable(tables, cx)
        })
    }

    /// Get the instance.
    pub(crate) fn instance_def_id(&self, instance: InstanceDef) -> DefId {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[instance];
            cx.instance_def_id(instance, tables)
        })
    }

    /// Get the instance mangled name.
    pub(crate) fn instance_mangled_name(&self, instance: InstanceDef) -> Symbol {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[instance];
            cx.instance_mangled_name(instance)
        })
    }

    /// Check if this is an empty DropGlue shim.
    pub(crate) fn is_empty_drop_shim(&self, def: InstanceDef) -> bool {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[def];
            cx.is_empty_drop_shim(instance)
        })
    }

    /// Convert a non-generic crate item into an instance.
    /// This function will panic if the item is generic.
    pub(crate) fn mono_instance(&self, def_id: DefId) -> Instance {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.mono_instance(did).stable(tables, cx)
        })
    }

    /// Item requires monomorphization.
    pub(crate) fn requires_monomorphization(&self, def_id: DefId) -> bool {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.requires_monomorphization(did)
        })
    }

    /// Resolve an instance from the given function definition and generic arguments.
    pub(crate) fn resolve_instance(&self, def: FnDef, args: &GenericArgs) -> Option<Instance> {
        self.with_cx(|tables, cx| {
            let def_id = def.0.internal(tables, cx.tcx);
            let args_ref = args.internal(tables, cx.tcx);
            cx.resolve_instance(def_id, args_ref).map(|inst| inst.stable(tables, cx))
        })
    }

    /// Resolve an instance for drop_in_place for the given type.
    pub(crate) fn resolve_drop_in_place(&self, ty: Ty) -> Instance {
        self.with_cx(|tables, cx| {
            let internal_ty = ty.internal(tables, cx.tcx);

            cx.resolve_drop_in_place(internal_ty).stable(tables, cx)
        })
    }

    /// Resolve instance for a function pointer.
    pub(crate) fn resolve_for_fn_ptr(&self, def: FnDef, args: &GenericArgs) -> Option<Instance> {
        self.with_cx(|tables, cx| {
            let def_id = def.0.internal(tables, cx.tcx);
            let args_ref = args.internal(tables, cx.tcx);
            cx.resolve_for_fn_ptr(def_id, args_ref).stable(tables, cx)
        })
    }

    /// Resolve instance for a closure with the requested type.
    pub(crate) fn resolve_closure(
        &self,
        def: ClosureDef,
        args: &GenericArgs,
        kind: ClosureKind,
    ) -> Option<Instance> {
        self.with_cx(|tables, cx| {
            let def_id = def.0.internal(tables, cx.tcx);
            let args_ref = args.internal(tables, cx.tcx);
            let closure_kind = kind.internal(tables, cx.tcx);
            cx.resolve_closure(def_id, args_ref, closure_kind).map(|inst| inst.stable(tables, cx))
        })
    }

    /// Evaluate a static's initializer.
    pub(crate) fn eval_static_initializer(&self, def: StaticDef) -> Result<Allocation, Error> {
        self.with_cx(|tables, cx| {
            let def_id = def.0.internal(tables, cx.tcx);

            cx.eval_static_initializer(def_id).stable(tables, cx)
        })
    }

    /// Try to evaluate an instance into a constant.
    pub(crate) fn eval_instance(
        &self,
        def: InstanceDef,
        const_ty: Ty,
    ) -> Result<Allocation, Error> {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[def];
            let const_ty = const_ty.internal(tables, cx.tcx);
            cx.eval_instance(instance)
                .map(|const_val| alloc::try_new_allocation(const_ty, const_val, tables, cx))
                .map_err(|e| e.stable(tables, cx))?
        })
    }

    /// Retrieve global allocation for the given allocation ID.
    pub(crate) fn global_alloc(&self, id: AllocId) -> GlobalAlloc {
        self.with_cx(|tables, cx| {
            let alloc_id = id.internal(tables, cx.tcx);
            cx.global_alloc(alloc_id).stable(tables, cx)
        })
    }

    /// Retrieve the id for the virtual table.
    pub(crate) fn vtable_allocation(&self, global_alloc: &GlobalAlloc) -> Option<AllocId> {
        self.with_cx(|tables, cx| {
            let GlobalAlloc::VTable(ty, trait_ref) = global_alloc else {
                return None;
            };
            let ty = ty.internal(tables, cx.tcx);
            let trait_ref = trait_ref.internal(tables, cx.tcx);
            let alloc_id = cx.vtable_allocation(ty, trait_ref);
            Some(alloc_id.stable(tables, cx))
        })
    }

    pub(crate) fn krate(&self, def_id: DefId) -> Crate {
        self.with_cx(|tables, cx| smir_crate(cx, tables[def_id].krate))
    }

    pub(crate) fn instance_name(&self, def: InstanceDef, trimmed: bool) -> Symbol {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[def];
            cx.instance_name(instance, trimmed)
        })
    }

    /// Return information about the target machine.
    pub(crate) fn target_info(&self) -> MachineInfo {
        self.with_cx(|tables, cx| MachineInfo {
            endian: cx.target_endian().stable(tables, cx),
            pointer_width: MachineSize::from_bits(cx.target_pointer_size()),
        })
    }

    /// Get an instance ABI.
    pub(crate) fn instance_abi(&self, def: InstanceDef) -> Result<FnAbi, Error> {
        self.with_cx(|tables, cx| {
            let instance = tables.instances[def];
            cx.instance_abi(instance).map(|fn_abi| fn_abi.stable(tables, cx))
        })
    }

    /// Get the ABI of a function pointer.
    pub(crate) fn fn_ptr_abi(&self, fn_ptr: PolyFnSig) -> Result<FnAbi, Error> {
        self.with_cx(|tables, cx| {
            let sig = fn_ptr.internal(tables, cx.tcx);
            cx.fn_ptr_abi(sig).map(|fn_abi| fn_abi.stable(tables, cx))
        })
    }

    /// Get the layout of a type.
    pub(crate) fn ty_layout(&self, ty: Ty) -> Result<Layout, Error> {
        self.with_cx(|tables, cx| {
            let internal_ty = ty.internal(tables, cx.tcx);
            cx.ty_layout(internal_ty).map(|layout| layout.stable(tables, cx))
        })
    }

    /// Get the layout shape.
    pub(crate) fn layout_shape(&self, id: Layout) -> LayoutShape {
        self.with_cx(|tables, cx| id.internal(tables, cx.tcx).0.stable(tables, cx))
    }

    /// Get a debug string representation of a place.
    pub(crate) fn place_pretty(&self, place: &Place) -> String {
        self.with_cx(|tables, cx| format!("{:?}", place.internal(tables, cx.tcx)))
    }

    /// Get the resulting type of binary operation.
    pub(crate) fn binop_ty(&self, bin_op: BinOp, rhs: Ty, lhs: Ty) -> Ty {
        self.with_cx(|tables, cx| {
            let rhs_internal = rhs.internal(tables, cx.tcx);
            let lhs_internal = lhs.internal(tables, cx.tcx);
            let bin_op_internal = bin_op.internal(tables, cx.tcx);
            cx.binop_ty(bin_op_internal, rhs_internal, lhs_internal).stable(tables, cx)
        })
    }

    /// Get the resulting type of unary operation.
    pub(crate) fn unop_ty(&self, un_op: UnOp, arg: Ty) -> Ty {
        self.with_cx(|tables, cx| {
            let un_op = un_op.internal(tables, cx.tcx);
            let arg = arg.internal(tables, cx.tcx);
            cx.unop_ty(un_op, arg).stable(tables, cx)
        })
    }

    /// Get the associated item of a definition if it is one.
    pub(crate) fn associated_item(&self, def_id: DefId) -> Option<AssocItem> {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.associated_item(did).map(|assoc| assoc.stable(tables, cx))
        })
    }

    /// Get all associated items of a definition.
    pub(crate) fn associated_items(&self, def_id: DefId) -> AssocItems {
        self.with_cx(|tables, cx| {
            let did = tables[def_id];
            cx.associated_items(did).iter().map(|assoc| assoc.stable(tables, cx)).collect()
        })
    }

    /// Get all vtable entries of a trait.
    pub(crate) fn vtable_entries(&self, trait_ref: &TraitRef) -> Vec<VtblEntry> {
        self.with_cx(|tables, cx| {
            cx.vtable_entries(trait_ref.internal(tables, cx.tcx))
                .iter()
                .map(|v| v.stable(tables, cx))
                .collect()
        })
    }

    /// Returns the vtable entry at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    pub(crate) fn vtable_entry(&self, trait_ref: &TraitRef, idx: usize) -> Option<VtblEntry> {
        self.with_cx(|tables, cx| {
            cx.vtable_entry(trait_ref.internal(tables, cx.tcx), idx).stable(tables, cx)
        })
    }
}

// A thread local variable that stores a pointer to [`CompilerInterface`].
scoped_tls::scoped_thread_local!(static TLV: Cell<*const ()>);

// remove this cfg when we have a stable driver.
#[cfg(feature = "rustc_internal")]
pub(crate) fn run<'tcx, F, T>(interface: &CompilerInterface<'tcx>, f: F) -> Result<T, Error>
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
pub(crate) fn with<R>(f: impl for<'tcx> FnOnce(&CompilerInterface<'tcx>) -> R) -> R {
    assert!(TLV.is_set());
    TLV.with(|tlv| {
        let ptr = tlv.get();
        assert!(!ptr.is_null());
        f(unsafe { *(ptr as *const &CompilerInterface<'_>) })
    })
}

fn smir_crate<'tcx>(
    cx: &CompilerCtxt<'tcx, BridgeTys>,
    crate_num: rustc_span::def_id::CrateNum,
) -> Crate {
    let name = cx.crate_name(crate_num);
    let is_local = cx.crate_is_local(crate_num);
    let id = CrateNum(cx.crate_num_id(crate_num), ThreadLocalIndex);
    debug!(?name, ?crate_num, "smir_crate");
    Crate { id, name, is_local }
}
