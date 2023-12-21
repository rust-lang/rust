//! Implementation of `[stable_mir::compiler_interface::Context]` trait.
//!
//! This trait is currently the main interface between the Rust compiler,
//! and the `stable_mir` crate.

#![allow(rustc::usage_of_qualified_ty)]

use rustc_abi::HasDataLayout;
use rustc_middle::ty;
use rustc_middle::ty::layout::{
    FnAbiOf, FnAbiOfHelpers, HasParamEnv, HasTyCtxt, LayoutOf, LayoutOfHelpers,
};
use rustc_middle::ty::print::{with_forced_trimmed_paths, with_no_trimmed_paths};
use rustc_middle::ty::{
    GenericPredicates, Instance, List, ParamEnv, ScalarInt, TyCtxt, TypeVisitableExt, ValTree,
};
use rustc_span::def_id::LOCAL_CRATE;
use stable_mir::abi::{FnAbi, Layout, LayoutShape};
use stable_mir::compiler_interface::Context;
use stable_mir::mir::alloc::GlobalAlloc;
use stable_mir::mir::mono::{InstanceDef, StaticDef};
use stable_mir::mir::Body;
use stable_mir::target::{MachineInfo, MachineSize};
use stable_mir::ty::{
    AdtDef, AdtKind, Allocation, ClosureDef, ClosureKind, Const, FieldDef, FnDef, GenericArgs,
    LineInfo, PolyFnSig, RigidTy, Span, Ty, TyKind, VariantDef,
};
use stable_mir::{Crate, CrateItem, DefId, Error, Filename, ItemKind, Symbol};
use std::cell::RefCell;

use crate::rustc_internal::{internal, RustcInternal};
use crate::rustc_smir::builder::BodyBuilder;
use crate::rustc_smir::{alloc, new_item_kind, smir_crate, Stable, Tables};

impl<'tcx> Context for TablesWrapper<'tcx> {
    fn target_info(&self) -> MachineInfo {
        let mut tables = self.0.borrow_mut();
        MachineInfo {
            endian: tables.tcx.data_layout.endian.stable(&mut *tables),
            pointer_width: MachineSize::from_bits(
                tables.tcx.data_layout.pointer_size.bits().try_into().unwrap(),
            ),
        }
    }

    fn entry_fn(&self) -> Option<stable_mir::CrateItem> {
        let mut tables = self.0.borrow_mut();
        let tcx = tables.tcx;
        Some(tables.crate_item(tcx.entry_fn(())?.0))
    }

    fn all_local_items(&self) -> stable_mir::CrateItems {
        let mut tables = self.0.borrow_mut();
        tables.tcx.mir_keys(()).iter().map(|item| tables.crate_item(item.to_def_id())).collect()
    }

    fn mir_body(&self, item: stable_mir::DefId) -> stable_mir::mir::Body {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[item];
        tables.tcx.instance_mir(rustc_middle::ty::InstanceDef::Item(def_id)).stable(&mut tables)
    }

    fn has_body(&self, def: DefId) -> bool {
        let tables = self.0.borrow();
        let def_id = tables[def];
        tables.tcx.is_mir_available(def_id)
    }

    fn all_trait_decls(&self) -> stable_mir::TraitDecls {
        let mut tables = self.0.borrow_mut();
        tables
            .tcx
            .traits(LOCAL_CRATE)
            .iter()
            .map(|trait_def_id| tables.trait_def(*trait_def_id))
            .collect()
    }

    fn trait_decl(&self, trait_def: &stable_mir::ty::TraitDef) -> stable_mir::ty::TraitDecl {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[trait_def.0];
        let trait_def = tables.tcx.trait_def(def_id);
        trait_def.stable(&mut *tables)
    }

    fn all_trait_impls(&self) -> stable_mir::ImplTraitDecls {
        let mut tables = self.0.borrow_mut();
        tables
            .tcx
            .trait_impls_in_crate(LOCAL_CRATE)
            .iter()
            .map(|impl_def_id| tables.impl_def(*impl_def_id))
            .collect()
    }

    fn trait_impl(&self, impl_def: &stable_mir::ty::ImplDef) -> stable_mir::ty::ImplTrait {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[impl_def.0];
        let impl_trait = tables.tcx.impl_trait_ref(def_id).unwrap();
        impl_trait.stable(&mut *tables)
    }

    fn generics_of(&self, def_id: stable_mir::DefId) -> stable_mir::ty::Generics {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[def_id];
        let generics = tables.tcx.generics_of(def_id);
        generics.stable(&mut *tables)
    }

    fn predicates_of(&self, def_id: stable_mir::DefId) -> stable_mir::ty::GenericPredicates {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[def_id];
        let GenericPredicates { parent, predicates } = tables.tcx.predicates_of(def_id);
        stable_mir::ty::GenericPredicates {
            parent: parent.map(|did| tables.trait_def(did)),
            predicates: predicates
                .iter()
                .map(|(clause, span)| {
                    (
                        clause.as_predicate().kind().skip_binder().stable(&mut *tables),
                        span.stable(&mut *tables),
                    )
                })
                .collect(),
        }
    }

    fn explicit_predicates_of(
        &self,
        def_id: stable_mir::DefId,
    ) -> stable_mir::ty::GenericPredicates {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[def_id];
        let GenericPredicates { parent, predicates } = tables.tcx.explicit_predicates_of(def_id);
        stable_mir::ty::GenericPredicates {
            parent: parent.map(|did| tables.trait_def(did)),
            predicates: predicates
                .iter()
                .map(|(clause, span)| {
                    (
                        clause.as_predicate().kind().skip_binder().stable(&mut *tables),
                        span.stable(&mut *tables),
                    )
                })
                .collect(),
        }
    }

    fn local_crate(&self) -> stable_mir::Crate {
        let tables = self.0.borrow();
        smir_crate(tables.tcx, LOCAL_CRATE)
    }

    fn external_crates(&self) -> Vec<stable_mir::Crate> {
        let tables = self.0.borrow();
        tables.tcx.crates(()).iter().map(|crate_num| smir_crate(tables.tcx, *crate_num)).collect()
    }

    fn find_crates(&self, name: &str) -> Vec<stable_mir::Crate> {
        let tables = self.0.borrow();
        let crates: Vec<stable_mir::Crate> = [LOCAL_CRATE]
            .iter()
            .chain(tables.tcx.crates(()).iter())
            .map(|crate_num| {
                let crate_name = tables.tcx.crate_name(*crate_num).to_string();
                (name == crate_name).then(|| smir_crate(tables.tcx, *crate_num))
            })
            .flatten()
            .collect();
        crates
    }

    fn def_name(&self, def_id: stable_mir::DefId, trimmed: bool) -> Symbol {
        let tables = self.0.borrow();
        if trimmed {
            with_forced_trimmed_paths!(tables.tcx.def_path_str(tables[def_id]))
        } else {
            with_no_trimmed_paths!(tables.tcx.def_path_str(tables[def_id]))
        }
    }

    fn span_to_string(&self, span: stable_mir::ty::Span) -> String {
        let tables = self.0.borrow();
        tables.tcx.sess.source_map().span_to_diagnostic_string(tables[span])
    }

    fn get_filename(&self, span: &Span) -> Filename {
        let tables = self.0.borrow();
        tables
            .tcx
            .sess
            .source_map()
            .span_to_filename(tables[*span])
            .display(rustc_span::FileNameDisplayPreference::Local)
            .to_string()
    }

    fn get_lines(&self, span: &Span) -> LineInfo {
        let tables = self.0.borrow();
        let lines = &tables.tcx.sess.source_map().span_to_location_info(tables[*span]);
        LineInfo { start_line: lines.1, start_col: lines.2, end_line: lines.3, end_col: lines.4 }
    }

    fn item_kind(&self, item: CrateItem) -> ItemKind {
        let tables = self.0.borrow();
        new_item_kind(tables.tcx.def_kind(tables[item.0]))
    }

    fn is_foreign_item(&self, item: DefId) -> bool {
        let tables = self.0.borrow();
        tables.tcx.is_foreign_item(tables[item])
    }

    fn adt_kind(&self, def: AdtDef) -> AdtKind {
        let mut tables = self.0.borrow_mut();
        def.internal(&mut *tables).adt_kind().stable(&mut *tables)
    }

    fn adt_is_box(&self, def: AdtDef) -> bool {
        let mut tables = self.0.borrow_mut();
        def.internal(&mut *tables).is_box()
    }

    fn adt_is_simd(&self, def: AdtDef) -> bool {
        let mut tables = self.0.borrow_mut();
        def.internal(&mut *tables).repr().simd()
    }

    fn adt_is_cstr(&self, def: AdtDef) -> bool {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        tables.tcx.lang_items().c_str() == Some(def_id)
    }

    fn fn_sig(&self, def: FnDef, args: &GenericArgs) -> PolyFnSig {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        let sig = tables.tcx.fn_sig(def_id).instantiate(tables.tcx, args.internal(&mut *tables));
        sig.stable(&mut *tables)
    }

    fn closure_sig(&self, args: &GenericArgs) -> PolyFnSig {
        let mut tables = self.0.borrow_mut();
        let args_ref = args.internal(&mut *tables);
        let sig = args_ref.as_closure().sig();
        sig.stable(&mut *tables)
    }

    fn adt_variants_len(&self, def: AdtDef) -> usize {
        let mut tables = self.0.borrow_mut();
        def.internal(&mut *tables).variants().len()
    }

    fn variant_name(&self, def: VariantDef) -> Symbol {
        let mut tables = self.0.borrow_mut();
        def.internal(&mut *tables).name.to_string()
    }

    fn variant_fields(&self, def: VariantDef) -> Vec<FieldDef> {
        let mut tables = self.0.borrow_mut();
        def.internal(&mut *tables).fields.iter().map(|f| f.stable(&mut *tables)).collect()
    }

    fn eval_target_usize(&self, cnst: &Const) -> Result<u64, Error> {
        let mut tables = self.0.borrow_mut();
        let mir_const = cnst.internal(&mut *tables);
        mir_const
            .try_eval_target_usize(tables.tcx, ParamEnv::empty())
            .ok_or_else(|| Error::new(format!("Const `{cnst:?}` cannot be encoded as u64")))
    }

    fn usize_to_const(&self, val: u64) -> Result<Const, Error> {
        let mut tables = self.0.borrow_mut();
        let ty = tables.tcx.types.usize;
        let size = tables.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap().size;

        let scalar = ScalarInt::try_from_uint(val, size).ok_or_else(|| {
            Error::new(format!("Value overflow: cannot convert `{val}` to usize."))
        })?;
        Ok(rustc_middle::ty::Const::new_value(tables.tcx, ValTree::from_scalar_int(scalar), ty)
            .stable(&mut *tables))
    }

    fn new_rigid_ty(&self, kind: RigidTy) -> stable_mir::ty::Ty {
        let mut tables = self.0.borrow_mut();
        let internal_kind = kind.internal(&mut *tables);
        tables.tcx.mk_ty_from_kind(internal_kind).stable(&mut *tables)
    }

    fn new_box_ty(&self, ty: stable_mir::ty::Ty) -> stable_mir::ty::Ty {
        let mut tables = self.0.borrow_mut();
        let inner = ty.internal(&mut *tables);
        ty::Ty::new_box(tables.tcx, inner).stable(&mut *tables)
    }

    fn def_ty(&self, item: stable_mir::DefId) -> stable_mir::ty::Ty {
        let mut tables = self.0.borrow_mut();
        tables.tcx.type_of(item.internal(&mut *tables)).instantiate_identity().stable(&mut *tables)
    }

    fn def_ty_with_args(&self, item: stable_mir::DefId, args: &GenericArgs) -> stable_mir::ty::Ty {
        let mut tables = self.0.borrow_mut();
        let args = args.internal(&mut *tables);
        let def_ty = tables.tcx.type_of(item.internal(&mut *tables));
        def_ty.instantiate(tables.tcx, args).stable(&mut *tables)
    }

    fn const_literal(&self, cnst: &stable_mir::ty::Const) -> String {
        internal(cnst).to_string()
    }

    fn span_of_an_item(&self, def_id: stable_mir::DefId) -> Span {
        let mut tables = self.0.borrow_mut();
        tables.tcx.def_span(tables[def_id]).stable(&mut *tables)
    }

    fn ty_kind(&self, ty: stable_mir::ty::Ty) -> TyKind {
        let mut tables = self.0.borrow_mut();
        tables.types[ty].kind().stable(&mut *tables)
    }

    fn rigid_ty_discriminant_ty(&self, ty: &RigidTy) -> stable_mir::ty::Ty {
        let mut tables = self.0.borrow_mut();
        let internal_kind = ty.internal(&mut *tables);
        let internal_ty = tables.tcx.mk_ty_from_kind(internal_kind);
        internal_ty.discriminant_ty(tables.tcx).stable(&mut *tables)
    }

    fn instance_body(&self, def: InstanceDef) -> Option<Body> {
        let mut tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        tables
            .has_body(instance)
            .then(|| BodyBuilder::new(tables.tcx, instance).build(&mut *tables))
    }

    fn instance_ty(&self, def: InstanceDef) -> stable_mir::ty::Ty {
        let mut tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        assert!(!instance.has_non_region_param(), "{instance:?} needs further substitution");
        instance.ty(tables.tcx, ParamEnv::reveal_all()).stable(&mut *tables)
    }

    fn instance_args(&self, def: InstanceDef) -> GenericArgs {
        let mut tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        instance.args.stable(&mut *tables)
    }

    fn instance_abi(&self, def: InstanceDef) -> Result<FnAbi, Error> {
        let mut tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        Ok(tables.fn_abi_of_instance(instance, List::empty())?.stable(&mut *tables))
    }

    fn instance_def_id(&self, def: InstanceDef) -> stable_mir::DefId {
        let mut tables = self.0.borrow_mut();
        let def_id = tables.instances[def].def_id();
        tables.create_def_id(def_id)
    }

    fn instance_mangled_name(&self, instance: InstanceDef) -> Symbol {
        let tables = self.0.borrow_mut();
        let instance = tables.instances[instance];
        tables.tcx.symbol_name(instance).name.to_string()
    }

    fn is_empty_drop_shim(&self, def: InstanceDef) -> bool {
        let tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        matches!(instance.def, ty::InstanceDef::DropGlue(_, None))
    }

    fn mono_instance(&self, def_id: stable_mir::DefId) -> stable_mir::mir::mono::Instance {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[def_id];
        Instance::mono(tables.tcx, def_id).stable(&mut *tables)
    }

    fn requires_monomorphization(&self, def_id: stable_mir::DefId) -> bool {
        let tables = self.0.borrow();
        let def_id = tables[def_id];
        let generics = tables.tcx.generics_of(def_id);
        let result = generics.requires_monomorphization(tables.tcx);
        result
    }

    fn resolve_instance(
        &self,
        def: stable_mir::ty::FnDef,
        args: &stable_mir::ty::GenericArgs,
    ) -> Option<stable_mir::mir::mono::Instance> {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        let args_ref = args.internal(&mut *tables);
        match Instance::resolve(tables.tcx, ParamEnv::reveal_all(), def_id, args_ref) {
            Ok(Some(instance)) => Some(instance.stable(&mut *tables)),
            Ok(None) | Err(_) => None,
        }
    }

    fn resolve_drop_in_place(&self, ty: stable_mir::ty::Ty) -> stable_mir::mir::mono::Instance {
        let mut tables = self.0.borrow_mut();
        let internal_ty = ty.internal(&mut *tables);
        let instance = Instance::resolve_drop_in_place(tables.tcx, internal_ty);
        instance.stable(&mut *tables)
    }

    fn resolve_for_fn_ptr(
        &self,
        def: FnDef,
        args: &GenericArgs,
    ) -> Option<stable_mir::mir::mono::Instance> {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        let args_ref = args.internal(&mut *tables);
        Instance::resolve_for_fn_ptr(tables.tcx, ParamEnv::reveal_all(), def_id, args_ref)
            .stable(&mut *tables)
    }

    fn resolve_closure(
        &self,
        def: ClosureDef,
        args: &GenericArgs,
        kind: ClosureKind,
    ) -> Option<stable_mir::mir::mono::Instance> {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        let args_ref = args.internal(&mut *tables);
        let closure_kind = kind.internal(&mut *tables);
        Instance::resolve_closure(tables.tcx, def_id, args_ref, closure_kind).stable(&mut *tables)
    }

    fn eval_instance(&self, def: InstanceDef, const_ty: Ty) -> Result<Allocation, Error> {
        let mut tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        let result = tables.tcx.const_eval_instance(
            ParamEnv::reveal_all(),
            instance,
            Some(tables.tcx.def_span(instance.def_id())),
        );
        result
            .map(|const_val| {
                alloc::try_new_allocation(const_ty.internal(&mut *tables), const_val, &mut *tables)
            })
            .map_err(|e| e.stable(&mut *tables))?
    }

    fn eval_static_initializer(&self, def: StaticDef) -> Result<Allocation, Error> {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        tables.tcx.eval_static_initializer(def_id).stable(&mut *tables)
    }

    fn global_alloc(&self, alloc: stable_mir::mir::alloc::AllocId) -> GlobalAlloc {
        let mut tables = self.0.borrow_mut();
        let alloc_id = alloc.internal(&mut *tables);
        tables.tcx.global_alloc(alloc_id).stable(&mut *tables)
    }

    fn vtable_allocation(
        &self,
        global_alloc: &GlobalAlloc,
    ) -> Option<stable_mir::mir::alloc::AllocId> {
        let mut tables = self.0.borrow_mut();
        let GlobalAlloc::VTable(ty, trait_ref) = global_alloc else { return None };
        let alloc_id = tables
            .tcx
            .vtable_allocation((ty.internal(&mut *tables), trait_ref.internal(&mut *tables)));
        Some(alloc_id.stable(&mut *tables))
    }

    fn krate(&self, def_id: stable_mir::DefId) -> Crate {
        let tables = self.0.borrow();
        smir_crate(tables.tcx, tables[def_id].krate)
    }

    /// Retrieve the instance name for diagnostic messages.
    ///
    /// This will return the specialized name, e.g., `Vec<char>::new`.
    fn instance_name(&self, def: InstanceDef, trimmed: bool) -> Symbol {
        let tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        if trimmed {
            with_forced_trimmed_paths!(
                tables.tcx.def_path_str_with_args(instance.def_id(), instance.args)
            )
        } else {
            with_no_trimmed_paths!(
                tables.tcx.def_path_str_with_args(instance.def_id(), instance.args)
            )
        }
    }

    fn ty_layout(&self, ty: Ty) -> Result<Layout, Error> {
        let mut tables = self.0.borrow_mut();
        let ty = ty.internal(&mut *tables);
        let layout = tables.layout_of(ty)?.layout;
        Ok(layout.stable(&mut *tables))
    }

    fn layout_shape(&self, id: Layout) -> LayoutShape {
        let mut tables = self.0.borrow_mut();
        id.internal(&mut *tables).0.stable(&mut *tables)
    }
}

pub struct TablesWrapper<'tcx>(pub RefCell<Tables<'tcx>>);

/// Implement error handling for extracting function ABI information.
impl<'tcx> FnAbiOfHelpers<'tcx> for Tables<'tcx> {
    type FnAbiOfResult = Result<&'tcx rustc_target::abi::call::FnAbi<'tcx, ty::Ty<'tcx>>, Error>;

    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: ty::layout::FnAbiError<'tcx>,
        _span: rustc_span::Span,
        fn_abi_request: ty::layout::FnAbiRequest<'tcx>,
    ) -> Error {
        Error::new(format!("Failed to get ABI for `{fn_abi_request:?}`: {err:?}"))
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for Tables<'tcx> {
    type LayoutOfResult = Result<ty::layout::TyAndLayout<'tcx>, Error>;

    #[inline]
    fn handle_layout_err(
        &self,
        err: ty::layout::LayoutError<'tcx>,
        _span: rustc_span::Span,
        ty: ty::Ty<'tcx>,
    ) -> Error {
        Error::new(format!("Failed to get layout for `{ty}`: {err}"))
    }
}

impl<'tcx> HasParamEnv<'tcx> for Tables<'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        ty::ParamEnv::reveal_all()
    }
}

impl<'tcx> HasTyCtxt<'tcx> for Tables<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> HasDataLayout for Tables<'tcx> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        self.tcx.data_layout()
    }
}
