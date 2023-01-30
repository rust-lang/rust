#![allow(rustc::usage_of_ty_tykind)]

use std::cmp::Ordering;
use std::iter;

use rustc_error_messages::MultiSpan;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::{def_id::DefId, LangItem};
use rustc_index::vec::IndexVec;
use rustc_span::{Symbol, DUMMY_SP};
use rustc_target::abi::{Layout, LayoutS, ReprOptions, VariantIdx};
use rustc_target::spec::abi::{self};
use rustc_type_ir::CollectAndApply;
use rustc_type_ir::{sty::TyKind::*, DynKind, FloatVid, InferTy::*, IntVid, TyVid};

use crate::infer::canonical::CanonicalVarInfo;
use crate::mir::interpret::{Allocation, ConstAllocation};
use crate::mir::ProjectionKind;
use crate::traits::solve::{ExternalConstraints, ExternalConstraintsData};
use crate::ty::{AdtDefData, AdtKind, InternalSubsts};
use crate::{
    mir::{Field, Local, Place, PlaceElem},
    ty::{
        self, AdtDef, Binder, Const, FloatTy, GenericArg, GenericParamDefKind, IntTy, List,
        ParamConst, ParamTy, PolyExistentialPredicate, PolyFnSig, Predicate, PredicateKind, Region,
        SubstsRef, Ty, TyCtxt, TyKind, TypeAndMut, UintTy,
    },
};

/// This structure provides convenience functions that make stuff.
///
/// This replaces `tcx.mk_stuff()` with `tcx.mk().stuff()`, making documentation a bit more bearable.
#[derive(Copy, Clone)]
pub struct MkCtxt<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> TyCtxt<'tcx> {
    /// Returns the "make context" with some extension methods.
    ///
    /// ## Examples
    ///
    /// ```rust,no_run
    /// # rustc_middle::ty::TyCtxt;
    /// let tcx: TyCtxt = todo!();
    /// let unit_ty = tcx.mk().unit();
    /// ```
    #[inline]
    pub fn mk(self) -> MkCtxt<'tcx> {
        MkCtxt { tcx: self }
    }
}

impl<'tcx> MkCtxt<'tcx> {
    pub fn adt_def(
        self,
        did: DefId,
        kind: AdtKind,
        variants: IndexVec<VariantIdx, ty::VariantDef>,
        repr: ReprOptions,
    ) -> ty::AdtDef<'tcx> {
        self.adt_def_from_data(ty::AdtDefData::new(self.tcx, did, kind, variants, repr))
    }

    /// Constructs a `RegionKind::ReError` lifetime.
    pub fn re_error(self, reported: ErrorGuaranteed) -> Region<'tcx> {
        self.tcx.intern_region(ty::ReError(reported))
    }

    /// Constructs a `RegionKind::ReError` lifetime and registers a `delay_span_bug` to ensure it
    /// gets used.
    #[track_caller]
    pub fn re_error_misc(self) -> Region<'tcx> {
        self.re_error_with_message(
            DUMMY_SP,
            "RegionKind::ReError constructed but no error reported",
        )
    }

    /// Constructs a `RegionKind::ReError` lifetime and registers a `delay_span_bug` with the given
    /// `msg` to ensure it gets used.
    #[track_caller]
    pub fn re_error_with_message<S: Into<MultiSpan>>(self, span: S, msg: &str) -> Region<'tcx> {
        let reported = self.tcx.sess.delay_span_bug(span, msg);
        self.re_error(reported)
    }

    // Avoid this in favour of more specific `mk().*` methods, where possible.
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline]
    pub fn ty_from_kind(self, st: TyKind<'tcx>) -> Ty<'tcx> {
        self.tcx.interners.intern_ty(
            st,
            self.tcx.sess,
            // This is only used to create a stable hashing context.
            &self.tcx.untracked,
        )
    }

    #[inline]
    pub fn predicate(self, binder: Binder<'tcx, PredicateKind<'tcx>>) -> Predicate<'tcx> {
        self.tcx.interners.intern_predicate(
            binder,
            self.tcx.sess,
            // This is only used to create a stable hashing context.
            &self.tcx.untracked,
        )
    }

    pub fn mach_int(self, tm: IntTy) -> Ty<'tcx> {
        match tm {
            IntTy::Isize => self.tcx.types.isize,
            IntTy::I8 => self.tcx.types.i8,
            IntTy::I16 => self.tcx.types.i16,
            IntTy::I32 => self.tcx.types.i32,
            IntTy::I64 => self.tcx.types.i64,
            IntTy::I128 => self.tcx.types.i128,
        }
    }

    pub fn mach_uint(self, tm: UintTy) -> Ty<'tcx> {
        match tm {
            UintTy::Usize => self.tcx.types.usize,
            UintTy::U8 => self.tcx.types.u8,
            UintTy::U16 => self.tcx.types.u16,
            UintTy::U32 => self.tcx.types.u32,
            UintTy::U64 => self.tcx.types.u64,
            UintTy::U128 => self.tcx.types.u128,
        }
    }

    pub fn mach_float(self, tm: FloatTy) -> Ty<'tcx> {
        match tm {
            FloatTy::F32 => self.tcx.types.f32,
            FloatTy::F64 => self.tcx.types.f64,
        }
    }

    #[inline]
    pub fn static_str(self) -> Ty<'tcx> {
        self.imm_ref(self.tcx.lifetimes.re_static, self.tcx.types.str_)
    }

    #[inline]
    pub fn adt(self, def: AdtDef<'tcx>, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        // Take a copy of substs so that we own the vectors inside.
        self.ty_from_kind(Adt(def, substs))
    }

    #[inline]
    pub fn foreign(self, def_id: DefId) -> Ty<'tcx> {
        self.ty_from_kind(Foreign(def_id))
    }

    fn generic_adt(self, wrapper_def_id: DefId, ty_param: Ty<'tcx>) -> Ty<'tcx> {
        let adt_def = self.tcx.adt_def(wrapper_def_id);
        let substs =
            InternalSubsts::for_item(self.tcx, wrapper_def_id, |param, substs| match param.kind {
                GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => bug!(),
                GenericParamDefKind::Type { has_default, .. } => {
                    if param.index == 0 {
                        ty_param.into()
                    } else {
                        assert!(has_default);
                        self.tcx.type_of(param.def_id).subst(self.tcx, substs).into()
                    }
                }
            });

        self.ty_from_kind(Adt(adt_def, substs))
    }

    #[inline]
    pub fn box_(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.tcx.require_lang_item(LangItem::OwnedBox, None);
        self.generic_adt(def_id, ty)
    }

    #[inline]
    pub fn lang_item(self, ty: Ty<'tcx>, item: LangItem) -> Option<Ty<'tcx>> {
        let def_id = self.tcx.lang_items().get(item)?;
        Some(self.generic_adt(def_id, ty))
    }

    #[inline]
    pub fn diagnostic_item(self, ty: Ty<'tcx>, name: Symbol) -> Option<Ty<'tcx>> {
        let def_id = self.tcx.get_diagnostic_item(name)?;
        Some(self.generic_adt(def_id, ty))
    }

    #[inline]
    pub fn maybe_uninit(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.tcx.require_lang_item(LangItem::MaybeUninit, None);
        self.generic_adt(def_id, ty)
    }

    #[inline]
    pub fn ptr(self, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(RawPtr(tm))
    }

    #[inline]
    pub fn ref_(self, r: Region<'tcx>, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(Ref(r, tm.ty, tm.mutbl))
    }

    #[inline]
    pub fn mut_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.ref_(r, TypeAndMut { ty, mutbl: hir::Mutability::Mut })
    }

    #[inline]
    pub fn imm_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.ref_(r, TypeAndMut { ty, mutbl: hir::Mutability::Not })
    }

    #[inline]
    pub fn mut_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.ptr(TypeAndMut { ty, mutbl: hir::Mutability::Mut })
    }

    #[inline]
    pub fn imm_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.ptr(TypeAndMut { ty, mutbl: hir::Mutability::Not })
    }

    #[inline]
    pub fn array(self, ty: Ty<'tcx>, n: u64) -> Ty<'tcx> {
        self.ty_from_kind(Array(ty, ty::Const::from_target_usize(self.tcx, n)))
    }

    #[inline]
    pub fn array_with_const_len(self, ty: Ty<'tcx>, ct: Const<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(Array(ty, ct))
    }

    #[inline]
    pub fn slice(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(Slice(ty))
    }

    pub fn tup(self, ts: &[Ty<'tcx>]) -> Ty<'tcx> {
        if ts.is_empty() {
            self.tcx.types.unit
        } else {
            self.ty_from_kind(Tuple(self.type_list(&ts)))
        }
    }

    pub fn tup_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<Ty<'tcx>, Ty<'tcx>>,
    {
        T::collect_and_apply(iter, |ts| self.tup(ts))
    }

    #[inline]
    pub fn unit(self) -> Ty<'tcx> {
        self.tcx.types.unit
    }

    #[inline]
    pub fn diverging_default(self) -> Ty<'tcx> {
        if self.tcx.features().never_type_fallback {
            self.tcx.types.never
        } else {
            self.tcx.types.unit
        }
    }

    #[inline]
    pub fn fn_def(
        self,
        def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> Ty<'tcx> {
        let substs = self.tcx.check_and_mk_substs(def_id, substs);
        self.ty_from_kind(FnDef(def_id, substs))
    }

    #[inline]
    pub fn fn_ptr(self, fty: PolyFnSig<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(FnPtr(fty))
    }

    #[inline]
    pub fn dynamic(
        self,
        obj: &'tcx List<PolyExistentialPredicate<'tcx>>,
        reg: ty::Region<'tcx>,
        repr: DynKind,
    ) -> Ty<'tcx> {
        self.ty_from_kind(Dynamic(obj, reg, repr))
    }

    #[inline]
    pub fn projection(
        self,
        item_def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> Ty<'tcx> {
        self.alias(ty::Projection, self.alias_ty(item_def_id, substs))
    }

    #[inline]
    pub fn closure(self, closure_id: DefId, closure_substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(Closure(closure_id, closure_substs))
    }

    #[inline]
    pub fn generator(
        self,
        id: DefId,
        generator_substs: SubstsRef<'tcx>,
        movability: hir::Movability,
    ) -> Ty<'tcx> {
        self.ty_from_kind(Generator(id, generator_substs, movability))
    }

    #[inline]
    pub fn generator_witness(self, types: ty::Binder<'tcx, &'tcx List<Ty<'tcx>>>) -> Ty<'tcx> {
        self.ty_from_kind(GeneratorWitness(types))
    }

    /// Creates a `&mut Context<'_>` [`Ty`] with erased lifetimes.
    pub fn task_context(self) -> Ty<'tcx> {
        let context_did = self.tcx.require_lang_item(LangItem::Context, None);
        let context_adt_ref = self.tcx.adt_def(context_did);
        let context_substs = self.substs(&[self.tcx.lifetimes.re_erased.into()]);
        let context_ty = self.adt(context_adt_ref, context_substs);
        self.mut_ref(self.tcx.lifetimes.re_erased, context_ty)
    }

    #[inline]
    pub fn generator_witness_mir(self, id: DefId, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(GeneratorWitnessMIR(id, substs))
    }

    #[inline]
    pub fn const_(self, kind: impl Into<ty::ConstKind<'tcx>>, ty: Ty<'tcx>) -> Const<'tcx> {
        self.tcx.intern_const(ty::ConstData { kind: kind.into(), ty })
    }

    #[inline]
    pub fn ty_var(self, v: TyVid) -> Ty<'tcx> {
        // Use a pre-interned one when possible.
        self.tcx
            .types
            .ty_vars
            .get(v.as_usize())
            .copied()
            .unwrap_or_else(|| self.ty_from_kind(Infer(TyVar(v))))
    }

    #[inline]
    pub fn int_var(self, v: IntVid) -> Ty<'tcx> {
        self.ty_from_kind(Infer(IntVar(v)))
    }

    #[inline]
    pub fn float_var(self, v: FloatVid) -> Ty<'tcx> {
        self.ty_from_kind(Infer(FloatVar(v)))
    }

    #[inline]
    pub fn fresh_ty(self, n: u32) -> Ty<'tcx> {
        // Use a pre-interned one when possible.
        self.tcx
            .types
            .fresh_tys
            .get(n as usize)
            .copied()
            .unwrap_or_else(|| self.ty_from_kind(Infer(ty::FreshTy(n))))
    }

    #[inline]
    pub fn fresh_int_ty(self, n: u32) -> Ty<'tcx> {
        // Use a pre-interned one when possible.
        self.tcx
            .types
            .fresh_int_tys
            .get(n as usize)
            .copied()
            .unwrap_or_else(|| self.ty_from_kind(Infer(ty::FreshIntTy(n))))
    }

    #[inline]
    pub fn fresh_float_ty(self, n: u32) -> Ty<'tcx> {
        // Use a pre-interned one when possible.
        self.tcx
            .types
            .fresh_float_tys
            .get(n as usize)
            .copied()
            .unwrap_or_else(|| self.ty_from_kind(Infer(ty::FreshFloatTy(n))))
    }

    #[inline]
    pub fn ty_param(self, index: u32, name: Symbol) -> Ty<'tcx> {
        self.ty_from_kind(Param(ParamTy { index, name }))
    }

    pub fn param_from_def(self, param: &ty::GenericParamDef) -> GenericArg<'tcx> {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                self.re_early_bound(param.to_early_bound_region_data()).into()
            }
            GenericParamDefKind::Type { .. } => self.ty_param(param.index, param.name).into(),
            GenericParamDefKind::Const { .. } => self
                .const_(
                    ParamConst { index: param.index, name: param.name },
                    self.tcx
                        .type_of(param.def_id)
                        .no_bound_vars()
                        .expect("const parameter types cannot be generic"),
                )
                .into(),
        }
    }

    #[inline]
    pub fn bound(self, index: ty::DebruijnIndex, bound_ty: ty::BoundTy) -> Ty<'tcx> {
        self.ty_from_kind(Bound(index, bound_ty))
    }

    #[inline]
    pub fn placeholder(self, placeholder: ty::PlaceholderType) -> Ty<'tcx> {
        self.ty_from_kind(Placeholder(placeholder))
    }

    #[inline]
    pub fn alias(self, kind: ty::AliasKind, alias_ty: ty::AliasTy<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(Alias(kind, alias_ty))
    }

    #[inline]
    pub fn opaque(self, def_id: DefId, substs: SubstsRef<'tcx>) -> Ty<'tcx> {
        self.ty_from_kind(Alias(ty::Opaque, self.alias_ty(def_id, substs)))
    }

    #[inline]
    pub fn re_early_bound(self, early_bound_region: ty::EarlyBoundRegion) -> Region<'tcx> {
        self.tcx.intern_region(ty::ReEarlyBound(early_bound_region))
    }

    #[inline]
    pub fn re_late_bound(
        self,
        debruijn: ty::DebruijnIndex,
        bound_region: ty::BoundRegion,
    ) -> Region<'tcx> {
        // Use a pre-interned one when possible.
        if let ty::BoundRegion { var, kind: ty::BrAnon(v, None) } = bound_region
            && var.as_u32() == v
            && let Some(inner) = self.tcx.lifetimes.re_late_bounds.get(debruijn.as_usize())
            && let Some(re) = inner.get(v as usize).copied()
        {
            re
        } else {
            self.tcx.intern_region(ty::ReLateBound(debruijn, bound_region))
        }
    }

    #[inline]
    pub fn re_free(self, scope: DefId, bound_region: ty::BoundRegionKind) -> Region<'tcx> {
        self.tcx.intern_region(ty::ReFree(ty::FreeRegion { scope, bound_region }))
    }

    #[inline]
    pub fn re_var(self, v: ty::RegionVid) -> Region<'tcx> {
        // Use a pre-interned one when possible.
        self.tcx
            .lifetimes
            .re_vars
            .get(v.as_usize())
            .copied()
            .unwrap_or_else(|| self.tcx.intern_region(ty::ReVar(v)))
    }

    #[inline]
    pub fn re_placeholder(self, placeholder: ty::PlaceholderRegion) -> Region<'tcx> {
        self.tcx.intern_region(ty::RePlaceholder(placeholder))
    }

    // Avoid this in favour of more specific `re_*` methods, where possible,
    // to avoid the cost of the `match`.
    pub fn region_from_kind(self, kind: ty::RegionKind<'tcx>) -> Region<'tcx> {
        match kind {
            ty::ReEarlyBound(region) => self.re_early_bound(region),
            ty::ReLateBound(debruijn, region) => self.re_late_bound(debruijn, region),
            ty::ReFree(ty::FreeRegion { scope, bound_region }) => self.re_free(scope, bound_region),
            ty::ReStatic => self.tcx.lifetimes.re_static,
            ty::ReVar(vid) => self.re_var(vid),
            ty::RePlaceholder(region) => self.re_placeholder(region),
            ty::ReErased => self.tcx.lifetimes.re_erased,
            ty::ReError(reported) => self.re_error(reported),
        }
    }

    pub fn place_field(self, place: Place<'tcx>, f: Field, ty: Ty<'tcx>) -> Place<'tcx> {
        self.place_elem(place, PlaceElem::Field(f, ty))
    }

    pub fn place_deref(self, place: Place<'tcx>) -> Place<'tcx> {
        self.place_elem(place, PlaceElem::Deref)
    }

    pub fn place_downcast(
        self,
        place: Place<'tcx>,
        adt_def: AdtDef<'tcx>,
        variant_index: VariantIdx,
    ) -> Place<'tcx> {
        self.place_elem(
            place,
            PlaceElem::Downcast(Some(adt_def.variant(variant_index).name), variant_index),
        )
    }

    pub fn place_downcast_unnamed(
        self,
        place: Place<'tcx>,
        variant_index: VariantIdx,
    ) -> Place<'tcx> {
        self.place_elem(place, PlaceElem::Downcast(None, variant_index))
    }

    pub fn place_index(self, place: Place<'tcx>, index: Local) -> Place<'tcx> {
        self.place_elem(place, PlaceElem::Index(index))
    }

    /// This method copies `Place`'s projection, add an element and reintern it. Should not be used
    /// to build a full `Place` it's just a convenient way to grab a projection and modify it in
    /// flight.
    pub fn place_elem(self, place: Place<'tcx>, elem: PlaceElem<'tcx>) -> Place<'tcx> {
        let mut projection = place.projection.to_vec();
        projection.push(elem);

        Place { local: place.local, projection: self.place_elems(&projection) }
    }

    pub fn poly_existential_predicates(
        self,
        eps: &[PolyExistentialPredicate<'tcx>],
    ) -> &'tcx List<PolyExistentialPredicate<'tcx>> {
        assert!(!eps.is_empty());
        assert!(
            eps.array_windows()
                .all(|[a, b]| a.skip_binder().stable_cmp(self.tcx, &b.skip_binder())
                    != Ordering::Greater)
        );
        self.tcx.intern_poly_existential_predicates(eps)
    }

    pub fn predicates(self, preds: &[Predicate<'tcx>]) -> &'tcx List<Predicate<'tcx>> {
        // FIXME consider asking the input slice to be sorted to avoid
        // re-interning permutations, in which case that would be asserted
        // here.
        self.tcx.intern_predicates(preds)
    }

    pub fn const_list_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<ty::Const<'tcx>, &'tcx List<ty::Const<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.const_list(xs))
    }

    pub fn type_list(self, ts: &[Ty<'tcx>]) -> &'tcx List<Ty<'tcx>> {
        // Actually intern type lists as lists of `GenericArg`s.
        //
        // Transmuting from `Ty<'tcx>` to `GenericArg<'tcx>` is sound
        // as explained in `ty_slice_as_generic_arg`. With this,
        // we guarantee that even when transmuting between `List<Ty<'tcx>>`
        // and `List<GenericArg<'tcx>>`, the uniqueness requirement for
        // lists is upheld.
        let substs = self.substs(ty::subst::ty_slice_as_generic_args(ts));
        substs.try_as_type_list().unwrap()
    }

    // Unlike various other `*_from_iter` functions, this one uses `I:
    // IntoIterator` instead of `I: Iterator`, and it doesn't have a slice
    // variant, because of the need to combine `inputs` and `output`. This
    // explains the lack of `_from_iter` suffix.
    pub fn fn_sig<I, T>(
        self,
        inputs: I,
        output: I::Item,
        c_variadic: bool,
        unsafety: hir::Unsafety,
        abi: abi::Abi,
    ) -> T::Output
    where
        I: IntoIterator<Item = T>,
        T: CollectAndApply<Ty<'tcx>, ty::FnSig<'tcx>>,
    {
        T::collect_and_apply(inputs.into_iter().chain(iter::once(output)), |xs| ty::FnSig {
            inputs_and_output: self.type_list(xs),
            c_variadic,
            unsafety,
            abi,
        })
    }

    pub fn poly_existential_predicates_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<
                PolyExistentialPredicate<'tcx>,
                &'tcx List<PolyExistentialPredicate<'tcx>>,
            >,
    {
        T::collect_and_apply(iter, |xs| self.poly_existential_predicates(xs))
    }

    pub fn predicates_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<Predicate<'tcx>, &'tcx List<Predicate<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.predicates(xs))
    }

    pub fn type_list_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<Ty<'tcx>, &'tcx List<Ty<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.type_list(xs))
    }

    pub fn substs_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<GenericArg<'tcx>, &'tcx List<GenericArg<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.substs(xs))
    }

    pub fn canonical_var_infos_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<CanonicalVarInfo<'tcx>, &'tcx List<CanonicalVarInfo<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.canonical_var_infos(xs))
    }

    pub fn place_elems_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<PlaceElem<'tcx>, &'tcx List<PlaceElem<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.place_elems(xs))
    }

    pub fn substs_trait(
        self,
        self_ty: Ty<'tcx>,
        rest: impl IntoIterator<Item = GenericArg<'tcx>>,
    ) -> SubstsRef<'tcx> {
        self.substs_from_iter(iter::once(self_ty.into()).chain(rest))
    }

    pub fn trait_ref(
        self,
        trait_def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> ty::TraitRef<'tcx> {
        let substs = self.tcx.check_and_mk_substs(trait_def_id, substs);
        ty::TraitRef { def_id: trait_def_id, substs, _use_mk_trait_ref_instead: () }
    }

    pub fn alias_ty(
        self,
        def_id: DefId,
        substs: impl IntoIterator<Item = impl Into<GenericArg<'tcx>>>,
    ) -> ty::AliasTy<'tcx> {
        let substs = self.tcx.check_and_mk_substs(def_id, substs);
        ty::AliasTy { def_id, substs, _use_mk_alias_ty_instead: () }
    }

    pub fn bound_variable_kinds_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<ty::BoundVariableKind, &'tcx List<ty::BoundVariableKind>>,
    {
        T::collect_and_apply(iter, |xs| self.bound_variable_kinds(xs))
    }

    #[inline]
    pub fn const_list(self, v: &[Const<'tcx>]) -> &'tcx List<Const<'tcx>> {
        self.tcx.mk_const_list(v)
    }

    #[inline]
    pub fn substs(self, v: &[GenericArg<'tcx>]) -> &'tcx List<GenericArg<'tcx>> {
        self.tcx.mk_substs(v)
    }

    #[inline]
    pub fn canonical_var_infos(
        self,
        v: &[CanonicalVarInfo<'tcx>],
    ) -> &'tcx List<CanonicalVarInfo<'tcx>> {
        self.tcx.mk_canonical_var_infos(v)
    }

    #[inline]
    pub fn projs(self, v: &[ProjectionKind]) -> &'tcx List<ProjectionKind> {
        self.tcx.mk_projs(v)
    }

    #[inline]
    pub fn place_elems(self, v: &[PlaceElem<'tcx>]) -> &'tcx List<PlaceElem<'tcx>> {
        self.tcx.mk_place_elems(v)
    }

    #[inline]
    pub fn bound_variable_kinds(
        self,
        v: &[ty::BoundVariableKind],
    ) -> &'tcx List<ty::BoundVariableKind> {
        self.tcx.mk_bound_variable_kinds(v)
    }

    #[inline]
    pub fn const_alloc(self, v: Allocation) -> ConstAllocation<'tcx> {
        self.tcx.mk_const_alloc(v)
    }

    #[inline]
    pub fn layout(self, v: LayoutS) -> Layout<'tcx> {
        self.tcx.mk_layout(v)
    }

    #[inline]
    pub fn adt_def_from_data(self, v: AdtDefData) -> AdtDef<'tcx> {
        self.tcx.mk_adt_def_from_data(v)
    }

    #[inline]
    pub fn external_constraints(
        self,
        v: ExternalConstraintsData<'tcx>,
    ) -> ExternalConstraints<'tcx> {
        self.tcx.mk_external_constraints(v)
    }
}
