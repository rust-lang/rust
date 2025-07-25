//! Implementation of CompilerCtxt.

#![allow(rustc::usage_of_qualified_ty)]

use std::iter;

use rustc_abi::{Endian, Layout, ReprOptions};
use rustc_hir::def::DefKind;
use rustc_hir::{Attribute, LangItem};
use rustc_middle::mir::interpret::{AllocId, ConstAllocation, ErrorHandled, GlobalAlloc, Scalar};
use rustc_middle::mir::{BinOp, Body, Const as MirConst, ConstValue, UnOp};
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf};
use rustc_middle::ty::print::{with_forced_trimmed_paths, with_no_trimmed_paths};
use rustc_middle::ty::util::Discr;
use rustc_middle::ty::{
    AdtDef, AdtKind, AssocItem, Binder, ClosureKind, CoroutineArgsExt, EarlyBinder,
    ExistentialTraitRef, FnSig, GenericArgsRef, Instance, InstanceKind, IntrinsicDef, List,
    PolyFnSig, ScalarInt, TraitDef, TraitRef, Ty, TyCtxt, TyKind, TypeVisitableExt, UintTy,
    ValTree, VariantDef,
};
use rustc_middle::{mir, ty};
use rustc_session::cstore::ForeignModule;
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_span::{FileNameDisplayPreference, Span, Symbol};
use rustc_target::callconv::FnAbi;

use super::{AllocRangeHelpers, CompilerCtxt, TyHelpers, TypingEnvHelpers};
use crate::builder::BodyBuilder;
use crate::{Bridge, Error, Tables, filter_def_ids};

impl<'tcx, B: Bridge> TyHelpers<'tcx> for CompilerCtxt<'tcx, B> {
    fn new_foreign(&self, def_id: DefId) -> ty::Ty<'tcx> {
        ty::Ty::new_foreign(self.tcx, def_id)
    }
}

impl<'tcx, B: Bridge> TypingEnvHelpers<'tcx> for CompilerCtxt<'tcx, B> {
    fn fully_monomorphized(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx, B: Bridge> AllocRangeHelpers<'tcx> for CompilerCtxt<'tcx, B> {
    fn alloc_range(
        &self,
        offset: rustc_abi::Size,
        size: rustc_abi::Size,
    ) -> mir::interpret::AllocRange {
        rustc_middle::mir::interpret::alloc_range(offset, size)
    }
}

impl<'tcx, B: Bridge> CompilerCtxt<'tcx, B> {
    pub fn lift<T: ty::Lift<TyCtxt<'tcx>>>(&self, value: T) -> Option<T::Lifted> {
        self.tcx.lift(value)
    }

    pub fn adt_def(&self, def_id: DefId) -> AdtDef<'tcx> {
        self.tcx.adt_def(def_id)
    }

    pub fn coroutine_movability(&self, def_id: DefId) -> ty::Movability {
        self.tcx.coroutine_movability(def_id)
    }

    pub fn valtree_to_const_val(&self, key: ty::Value<'tcx>) -> ConstValue {
        self.tcx.valtree_to_const_val(key)
    }

    /// Return whether the instance as a body available.
    ///
    /// Items and intrinsics may have a body available from its definition.
    /// Shims body may be generated depending on their type.
    pub(crate) fn instance_has_body(&self, instance: Instance<'tcx>) -> bool {
        let def_id = instance.def_id();
        self.item_has_body(def_id)
            || !matches!(
                instance.def,
                ty::InstanceKind::Virtual(..)
                    | ty::InstanceKind::Intrinsic(..)
                    | ty::InstanceKind::Item(..)
            )
    }

    /// Return whether the item has a body defined by the user.
    ///
    /// Note that intrinsics may have a placeholder body that shouldn't be used in practice.
    /// In rustc_public, we handle this case as if the body is not available.
    pub(crate) fn item_has_body(&self, def_id: DefId) -> bool {
        let must_override = if let Some(intrinsic) = self.tcx.intrinsic(def_id) {
            intrinsic.must_be_overridden
        } else {
            false
        };
        !must_override && self.tcx.is_mir_available(def_id)
    }

    fn filter_fn_def(&self, def_id: DefId) -> Option<DefId> {
        if matches!(self.tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
            Some(def_id)
        } else {
            None
        }
    }

    fn filter_static_def(&self, def_id: DefId) -> Option<DefId> {
        matches!(self.tcx.def_kind(def_id), DefKind::Static { .. }).then(|| def_id)
    }

    pub fn target_endian(&self) -> Endian {
        self.tcx.data_layout.endian
    }

    pub fn target_pointer_size(&self) -> usize {
        self.tcx.data_layout.pointer_size().bits().try_into().unwrap()
    }

    pub fn entry_fn(&self) -> Option<DefId> {
        Some(self.tcx.entry_fn(())?.0)
    }

    /// Retrieve all items of the local crate that have a MIR associated with them.
    pub fn all_local_items(&self) -> Vec<DefId> {
        self.tcx.mir_keys(()).iter().map(|item| item.to_def_id()).collect()
    }

    /// Retrieve the body of a function.
    /// This function will panic if the body is not available.
    pub fn mir_body(&self, item: DefId) -> &'tcx Body<'tcx> {
        self.tcx.instance_mir(InstanceKind::Item(item))
    }

    /// Check whether the body of a function is available.
    pub fn has_body(&self, def: DefId) -> bool {
        self.item_has_body(def)
    }

    pub fn foreign_modules(&self, crate_num: CrateNum) -> Vec<DefId> {
        self.tcx.foreign_modules(crate_num).keys().map(|mod_def_id| *mod_def_id).collect()
    }

    /// Retrieve all functions defined in this crate.
    pub fn crate_functions(&self, crate_num: CrateNum) -> Vec<DefId> {
        filter_def_ids(self.tcx, crate_num, |def_id| self.filter_fn_def(def_id))
    }

    /// Retrieve all static items defined in this crate.
    pub fn crate_statics(&self, crate_num: CrateNum) -> Vec<DefId> {
        filter_def_ids(self.tcx, crate_num, |def_id| self.filter_static_def(def_id))
    }

    pub fn foreign_module(&self, mod_def: DefId) -> &ForeignModule {
        self.tcx.foreign_modules(mod_def.krate).get(&mod_def).unwrap()
    }

    pub fn foreign_items(&self, mod_def: DefId) -> Vec<DefId> {
        self.tcx
            .foreign_modules(mod_def.krate)
            .get(&mod_def)
            .unwrap()
            .foreign_items
            .iter()
            .map(|item_def| *item_def)
            .collect()
    }

    pub fn all_trait_decls(&self) -> impl Iterator<Item = DefId> {
        self.tcx.all_traits_including_private()
    }

    pub fn trait_decls(&self, crate_num: CrateNum) -> Vec<DefId> {
        self.tcx.traits(crate_num).iter().map(|trait_def_id| *trait_def_id).collect()
    }

    pub fn trait_decl(&self, trait_def: DefId) -> &'tcx TraitDef {
        self.tcx.trait_def(trait_def)
    }

    pub fn all_trait_impls(&self) -> Vec<DefId> {
        iter::once(LOCAL_CRATE)
            .chain(self.tcx.crates(()).iter().copied())
            .flat_map(|cnum| self.tcx.trait_impls_in_crate(cnum).iter())
            .map(|impl_def_id| *impl_def_id)
            .collect()
    }

    pub fn trait_impls(&self, crate_num: CrateNum) -> Vec<DefId> {
        self.tcx.trait_impls_in_crate(crate_num).iter().map(|impl_def_id| *impl_def_id).collect()
    }

    pub fn trait_impl(&self, impl_def: DefId) -> EarlyBinder<'tcx, TraitRef<'tcx>> {
        self.tcx.impl_trait_ref(impl_def).unwrap()
    }

    pub fn generics_of(&self, def_id: DefId) -> &'tcx ty::Generics {
        self.tcx.generics_of(def_id)
    }

    pub fn predicates_of(
        &self,
        def_id: DefId,
    ) -> (Option<DefId>, Vec<(ty::PredicateKind<'tcx>, Span)>) {
        let ty::GenericPredicates { parent, predicates } = self.tcx.predicates_of(def_id);
        (
            parent,
            predicates
                .iter()
                .map(|(clause, span)| (clause.as_predicate().kind().skip_binder(), *span))
                .collect(),
        )
    }

    pub fn explicit_predicates_of(
        &self,
        def_id: DefId,
    ) -> (Option<DefId>, Vec<(ty::PredicateKind<'tcx>, Span)>) {
        let ty::GenericPredicates { parent, predicates } = self.tcx.explicit_predicates_of(def_id);
        (
            parent,
            predicates
                .iter()
                .map(|(clause, span)| (clause.as_predicate().kind().skip_binder(), *span))
                .collect(),
        )
    }

    pub fn crate_name(&self, crate_num: CrateNum) -> String {
        self.tcx.crate_name(crate_num).to_string()
    }

    pub fn crate_is_local(&self, crate_num: CrateNum) -> bool {
        crate_num == LOCAL_CRATE
    }

    pub fn crate_num_id(&self, crate_num: CrateNum) -> usize {
        crate_num.into()
    }

    pub fn local_crate_num(&self) -> CrateNum {
        LOCAL_CRATE
    }

    /// Retrieve a list of all external crates.
    pub fn external_crates(&self) -> Vec<CrateNum> {
        self.tcx.crates(()).iter().map(|crate_num| *crate_num).collect()
    }

    /// Find a crate with the given name.
    pub fn find_crates(&self, name: &str) -> Vec<CrateNum> {
        let crates: Vec<CrateNum> = [LOCAL_CRATE]
            .iter()
            .chain(self.tcx.crates(()).iter())
            .filter_map(|crate_num| {
                let crate_name = self.tcx.crate_name(*crate_num).to_string();
                (name == crate_name).then(|| *crate_num)
            })
            .collect();
        crates
    }

    /// Returns the name of given `DefId`.
    pub fn def_name(&self, def_id: DefId, trimmed: bool) -> String {
        if trimmed {
            with_forced_trimmed_paths!(self.tcx.def_path_str(def_id))
        } else {
            with_no_trimmed_paths!(self.tcx.def_path_str(def_id))
        }
    }

    /// Return registered tool attributes with the given attribute name.
    ///
    /// FIXME(jdonszelmann): may panic on non-tool attributes. After more attribute work, non-tool
    /// attributes will simply return an empty list.
    ///
    /// Single segmented name like `#[clippy]` is specified as `&["clippy".to_string()]`.
    /// Multi-segmented name like `#[rustfmt::skip]` is specified as `&["rustfmt".to_string(), "skip".to_string()]`.
    pub fn tool_attrs(&self, def_id: DefId, attr: &[String]) -> Vec<(String, Span)> {
        let attr_name: Vec<_> = attr.iter().map(|seg| Symbol::intern(&seg)).collect();
        self.tcx
            .get_attrs_by_path(def_id, &attr_name)
            .filter_map(|attribute| {
                if let Attribute::Unparsed(u) = attribute {
                    let attr_str = rustc_hir_pretty::attribute_to_string(&self.tcx, attribute);
                    Some((attr_str, u.span))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all tool attributes of a definition.
    pub fn all_tool_attrs(&self, did: DefId) -> Vec<(String, Span)> {
        let attrs_iter = if let Some(did) = did.as_local() {
            self.tcx.hir_attrs(self.tcx.local_def_id_to_hir_id(did)).iter()
        } else {
            self.tcx.attrs_for_def(did).iter()
        };
        attrs_iter
            .filter_map(|attribute| {
                if let Attribute::Unparsed(u) = attribute {
                    let attr_str = rustc_hir_pretty::attribute_to_string(&self.tcx, attribute);
                    Some((attr_str, u.span))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns printable, human readable form of `Span`.
    pub fn span_to_string(&self, span: Span) -> String {
        self.tcx.sess.source_map().span_to_diagnostic_string(span)
    }

    /// Return filename from given `Span`, for diagnostic purposes.
    pub fn get_filename(&self, span: Span) -> String {
        self.tcx
            .sess
            .source_map()
            .span_to_filename(span)
            .display(FileNameDisplayPreference::Local)
            .to_string()
    }

    /// Return lines corresponding to this `Span`.
    pub fn get_lines(&self, span: Span) -> (usize, usize, usize, usize) {
        let lines = &self.tcx.sess.source_map().span_to_location_info(span);
        (lines.1, lines.2, lines.3, lines.4)
    }

    /// Returns the `kind` of given `DefId`.
    pub fn def_kind(&self, item: DefId) -> DefKind {
        self.tcx.def_kind(item)
    }

    /// Returns whether this is a foreign item.
    pub fn is_foreign_item(&self, item: DefId) -> bool {
        self.tcx.is_foreign_item(item)
    }

    /// Returns the kind of a given foreign item.
    pub fn foreign_item_kind(&self, def_id: DefId) -> DefKind {
        self.tcx.def_kind(def_id)
    }

    /// Returns the kind of a given algebraic data type.
    pub fn adt_kind(&self, def: AdtDef<'tcx>) -> AdtKind {
        def.adt_kind()
    }

    /// Returns if the ADT is a box.
    pub fn adt_is_box(&self, def: AdtDef<'tcx>) -> bool {
        def.is_box()
    }

    /// Returns whether this ADT is simd.
    pub fn adt_is_simd(&self, def: AdtDef<'tcx>) -> bool {
        def.repr().simd()
    }

    /// Returns whether this definition is a C string.
    pub fn adt_is_cstr(&self, def_id: DefId) -> bool {
        self.tcx.is_lang_item(def_id, LangItem::CStr)
    }

    /// Returns the representation options for this ADT.
    pub fn adt_repr(&self, def: AdtDef<'tcx>) -> ReprOptions {
        def.repr()
    }

    /// Retrieve the function signature for the given generic arguments.
    pub fn fn_sig(
        &self,
        def_id: DefId,
        args_ref: GenericArgsRef<'tcx>,
    ) -> Binder<'tcx, FnSig<'tcx>> {
        let sig = self.tcx.fn_sig(def_id).instantiate(self.tcx, args_ref);
        sig
    }

    /// Retrieve the intrinsic definition if the item corresponds one.
    pub fn intrinsic(&self, def_id: DefId) -> Option<IntrinsicDef> {
        let intrinsic = self.tcx.intrinsic_raw(def_id);
        intrinsic
    }

    /// Retrieve the plain function name of an intrinsic.
    pub fn intrinsic_name(&self, def_id: DefId) -> String {
        self.tcx.intrinsic(def_id).unwrap().name.to_string()
    }

    /// Retrieve the closure signature for the given generic arguments.
    pub fn closure_sig(&self, args_ref: GenericArgsRef<'tcx>) -> Binder<'tcx, FnSig<'tcx>> {
        args_ref.as_closure().sig()
    }

    /// The number of variants in this ADT.
    pub fn adt_variants_len(&self, def: AdtDef<'tcx>) -> usize {
        def.variants().len()
    }

    /// Discriminant for a given variant index of AdtDef.
    pub fn adt_discr_for_variant(
        &self,
        adt: AdtDef<'tcx>,
        variant: rustc_abi::VariantIdx,
    ) -> Discr<'tcx> {
        adt.discriminant_for_variant(self.tcx, variant)
    }

    /// Discriminant for a given variand index and args of a coroutine.
    pub fn coroutine_discr_for_variant(
        &self,
        coroutine: DefId,
        args: GenericArgsRef<'tcx>,
        variant: rustc_abi::VariantIdx,
    ) -> Discr<'tcx> {
        args.as_coroutine().discriminant_for_variant(coroutine, self.tcx, variant)
    }

    /// The name of a variant.
    pub fn variant_name(&self, def: &'tcx VariantDef) -> String {
        def.name.to_string()
    }

    /// Evaluate constant as a target usize.
    pub fn eval_target_usize(&self, cnst: MirConst<'tcx>) -> Result<u64, B::Error> {
        use crate::context::TypingEnvHelpers;
        cnst.try_eval_target_usize(self.tcx, self.fully_monomorphized())
            .ok_or_else(|| B::Error::new(format!("Const `{cnst:?}` cannot be encoded as u64")))
    }

    pub fn eval_target_usize_ty(&self, cnst: ty::Const<'tcx>) -> Result<u64, B::Error> {
        cnst.try_to_target_usize(self.tcx)
            .ok_or_else(|| B::Error::new(format!("Const `{cnst:?}` cannot be encoded as u64")))
    }

    pub fn try_new_const_zst(&self, ty_internal: Ty<'tcx>) -> Result<MirConst<'tcx>, B::Error> {
        let size = self
            .tcx
            .layout_of(self.fully_monomorphized().as_query_input(ty_internal))
            .map_err(|err| {
                B::Error::new(format!(
                    "Cannot create a zero-sized constant for type `{ty_internal}`: {err}"
                ))
            })?
            .size;
        if size.bytes() != 0 {
            return Err(B::Error::new(format!(
                "Cannot create a zero-sized constant for type `{ty_internal}`: \
                Type `{ty_internal}` has {} bytes",
                size.bytes()
            )));
        }

        Ok(MirConst::Ty(ty_internal, self.const_zero_sized(ty_internal)))
    }

    pub fn const_zero_sized(&self, ty_internal: Ty<'tcx>) -> ty::Const<'tcx> {
        ty::Const::zero_sized(self.tcx, ty_internal)
    }

    /// Create a new constant that represents the given string value.
    pub fn new_const_str(&self, value: &str) -> MirConst<'tcx> {
        let ty = Ty::new_static_str(self.tcx);
        let bytes = value.as_bytes();
        let valtree = ValTree::from_raw_bytes(self.tcx, bytes);
        let cv = ty::Value { ty, valtree };
        let val = self.tcx.valtree_to_const_val(cv);
        MirConst::from_value(val, ty)
    }

    /// Create a new constant that represents the given boolean value.
    pub fn new_const_bool(&self, value: bool) -> MirConst<'tcx> {
        MirConst::from_bool(self.tcx, value)
    }

    pub fn try_new_const_uint(
        &self,
        value: u128,
        ty_internal: Ty<'tcx>,
    ) -> Result<MirConst<'tcx>, B::Error> {
        let size = self
            .tcx
            .layout_of(self.fully_monomorphized().as_query_input(ty_internal))
            .unwrap()
            .size;
        let scalar = ScalarInt::try_from_uint(value, size).ok_or_else(|| {
            B::Error::new(format!("Value overflow: cannot convert `{value}` to `{ty_internal}`."))
        })?;
        Ok(self.mir_const_from_scalar(Scalar::Int(scalar), ty_internal))
    }

    pub fn try_new_ty_const_uint(
        &self,
        value: u128,
        ty_internal: Ty<'tcx>,
    ) -> Result<ty::Const<'tcx>, B::Error> {
        let size = self
            .tcx
            .layout_of(self.fully_monomorphized().as_query_input(ty_internal))
            .unwrap()
            .size;
        let scalar = ScalarInt::try_from_uint(value, size).ok_or_else(|| {
            B::Error::new(format!("Value overflow: cannot convert `{value}` to `{ty_internal}`."))
        })?;

        Ok(self.ty_const_new_value(ValTree::from_scalar_int(self.tcx, scalar), ty_internal))
    }

    pub fn ty_new_uint(&self, ty: UintTy) -> Ty<'tcx> {
        Ty::new_uint(self.tcx, ty)
    }

    pub fn mir_const_from_scalar(&self, s: Scalar, ty: Ty<'tcx>) -> MirConst<'tcx> {
        MirConst::from_scalar(self.tcx, s, ty)
    }

    pub fn ty_const_new_value(&self, valtree: ValTree<'tcx>, ty: Ty<'tcx>) -> ty::Const<'tcx> {
        ty::Const::new_value(self.tcx, valtree, ty)
    }

    pub fn ty_valtree_from_scalar_int(&self, i: ScalarInt) -> ValTree<'tcx> {
        ValTree::from_scalar_int(self.tcx, i)
    }

    /// Create a new type from the given kind.
    pub fn new_rigid_ty(&self, internal_kind: TyKind<'tcx>) -> Ty<'tcx> {
        self.tcx.mk_ty_from_kind(internal_kind)
    }

    /// Create a new box type, `Box<T>`, for the given inner type `T`.
    pub fn new_box_ty(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        ty::Ty::new_box(self.tcx, ty)
    }

    /// Returns the type of given crate item.
    pub fn def_ty(&self, item: DefId) -> Ty<'tcx> {
        self.tcx.type_of(item).instantiate_identity()
    }

    /// Returns the type of given definition instantiated with the given arguments.
    pub fn def_ty_with_args(&self, item: DefId, args_ref: GenericArgsRef<'tcx>) -> Ty<'tcx> {
        let def_ty = self.tcx.type_of(item);
        self.tcx.instantiate_and_normalize_erasing_regions(
            args_ref,
            self.fully_monomorphized(),
            def_ty,
        )
    }

    /// `Span` of an item.
    pub fn span_of_an_item(&self, def_id: DefId) -> Span {
        self.tcx.def_span(def_id)
    }

    pub fn ty_const_pretty(&self, ct: ty::Const<'tcx>) -> String {
        ct.to_string()
    }

    /// Obtain the representation of a type.
    pub fn ty_pretty(&self, ty: Ty<'tcx>) -> String {
        ty.to_string()
    }

    /// Obtain the kind of a type.
    pub fn ty_kind(&self, ty: Ty<'tcx>) -> &'tcx TyKind<'tcx> {
        ty.kind()
    }

    /// Get the discriminant Ty for this Ty if there's one.
    pub fn rigid_ty_discriminant_ty(&self, internal_kind: TyKind<'tcx>) -> Ty<'tcx> {
        let internal_ty = self.tcx.mk_ty_from_kind(internal_kind);
        internal_ty.discriminant_ty(self.tcx)
    }

    /// Get the body of an Instance which is already monomorphized.
    pub fn instance_body(&self, instance: ty::Instance<'tcx>) -> Option<Body<'tcx>> {
        self.instance_has_body(instance).then(|| BodyBuilder::new(self.tcx, instance).build())
    }

    /// Get the instance type with generic instantiations applied and lifetimes erased.
    pub fn instance_ty(&self, instance: ty::Instance<'tcx>) -> Ty<'tcx> {
        assert!(!instance.has_non_region_param(), "{instance:?} needs further instantiation");
        instance.ty(self.tcx, self.fully_monomorphized())
    }

    /// Get the instantiation types.
    pub fn instance_args(&self, instance: ty::Instance<'tcx>) -> GenericArgsRef<'tcx> {
        instance.args
    }

    /// Get an instance ABI.
    pub fn instance_abi(
        &self,
        instance: ty::Instance<'tcx>,
    ) -> Result<&FnAbi<'tcx, Ty<'tcx>>, B::Error> {
        Ok(self.fn_abi_of_instance(instance, List::empty())?)
    }

    /// Get the ABI of a function pointer.
    pub fn fn_ptr_abi(&self, sig: PolyFnSig<'tcx>) -> Result<&FnAbi<'tcx, Ty<'tcx>>, B::Error> {
        Ok(self.fn_abi_of_fn_ptr(sig, List::empty())?)
    }

    /// Get the instance.
    pub fn instance_def_id(
        &self,
        instances: ty::Instance<'tcx>,
        tables: &mut Tables<'_, B>,
    ) -> B::DefId {
        let def_id = instances.def_id();
        tables.create_def_id(def_id)
    }

    /// Get the instance mangled name.
    pub fn instance_mangled_name(&self, instance: ty::Instance<'tcx>) -> String {
        self.tcx.symbol_name(instance).name.to_string()
    }

    /// Check if this is an empty DropGlue shim.
    pub fn is_empty_drop_shim(&self, instance: ty::Instance<'tcx>) -> bool {
        matches!(instance.def, ty::InstanceKind::DropGlue(_, None))
    }

    /// Convert a non-generic crate item into an instance.
    /// This function will panic if the item is generic.
    pub fn mono_instance(&self, def_id: DefId) -> Instance<'tcx> {
        Instance::mono(self.tcx, def_id)
    }

    /// Item requires monomorphization.
    pub fn requires_monomorphization(&self, def_id: DefId) -> bool {
        let generics = self.tcx.generics_of(def_id);
        let result = generics.requires_monomorphization(self.tcx);
        result
    }

    /// Resolve an instance from the given function definition and generic arguments.
    pub fn resolve_instance(
        &self,
        def_id: DefId,
        args_ref: GenericArgsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        match Instance::try_resolve(self.tcx, self.fully_monomorphized(), def_id, args_ref) {
            Ok(Some(instance)) => Some(instance),
            Ok(None) | Err(_) => None,
        }
    }

    /// Resolve an instance for drop_in_place for the given type.
    pub fn resolve_drop_in_place(&self, internal_ty: Ty<'tcx>) -> Instance<'tcx> {
        let instance = Instance::resolve_drop_in_place(self.tcx, internal_ty);
        instance
    }

    /// Resolve instance for a function pointer.
    pub fn resolve_for_fn_ptr(
        &self,
        def_id: DefId,
        args_ref: GenericArgsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        Instance::resolve_for_fn_ptr(self.tcx, self.fully_monomorphized(), def_id, args_ref)
    }

    /// Resolve instance for a closure with the requested type.
    pub fn resolve_closure(
        &self,
        def_id: DefId,
        args_ref: GenericArgsRef<'tcx>,
        closure_kind: ClosureKind,
    ) -> Option<Instance<'tcx>> {
        Some(Instance::resolve_closure(self.tcx, def_id, args_ref, closure_kind))
    }

    /// Try to evaluate an instance into a constant.
    pub fn eval_instance(&self, instance: ty::Instance<'tcx>) -> Result<ConstValue, ErrorHandled> {
        self.tcx.const_eval_instance(
            self.fully_monomorphized(),
            instance,
            self.tcx.def_span(instance.def_id()),
        )
    }

    /// Evaluate a static's initializer.
    pub fn eval_static_initializer(
        &self,
        def_id: DefId,
    ) -> Result<ConstAllocation<'tcx>, ErrorHandled> {
        self.tcx.eval_static_initializer(def_id)
    }

    /// Retrieve global allocation for the given allocation ID.
    pub fn global_alloc(&self, alloc_id: AllocId) -> GlobalAlloc<'tcx> {
        self.tcx.global_alloc(alloc_id)
    }

    /// Retrieve the id for the virtual table.
    pub fn vtable_allocation(
        &self,
        ty: Ty<'tcx>,
        trait_ref: Option<Binder<'tcx, ExistentialTraitRef<'tcx>>>,
    ) -> AllocId {
        let alloc_id = self.tcx.vtable_allocation((
            ty,
            trait_ref.map(|principal| self.tcx.instantiate_bound_regions_with_erased(principal)),
        ));
        alloc_id
    }

    /// Retrieve the instance name for diagnostic messages.
    ///
    /// This will return the specialized name, e.g., `Vec<char>::new`.
    pub fn instance_name(&self, instance: ty::Instance<'tcx>, trimmed: bool) -> String {
        if trimmed {
            with_forced_trimmed_paths!(
                self.tcx.def_path_str_with_args(instance.def_id(), instance.args)
            )
        } else {
            with_no_trimmed_paths!(
                self.tcx.def_path_str_with_args(instance.def_id(), instance.args)
            )
        }
    }

    /// Get the layout of a type.
    pub fn ty_layout(&self, ty: Ty<'tcx>) -> Result<Layout<'tcx>, B::Error> {
        let layout = self.layout_of(ty)?.layout;
        Ok(layout)
    }

    /// Get the resulting type of binary operation.
    pub fn binop_ty(&self, bin_op: BinOp, rhs: Ty<'tcx>, lhs: Ty<'tcx>) -> Ty<'tcx> {
        bin_op.ty(self.tcx, rhs, lhs)
    }

    /// Get the resulting type of unary operation.
    pub fn unop_ty(&self, un_op: UnOp, arg: Ty<'tcx>) -> Ty<'tcx> {
        un_op.ty(self.tcx, arg)
    }

    /// Get all associated items of a definition.
    pub fn associated_items(&self, def_id: DefId) -> Vec<AssocItem> {
        let assoc_items = if self.tcx.is_trait_alias(def_id) {
            Vec::new()
        } else {
            self.tcx
                .associated_item_def_ids(def_id)
                .iter()
                .map(|did| self.tcx.associated_item(*did))
                .collect()
        };
        assoc_items
    }
}
