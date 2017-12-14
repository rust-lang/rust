use rustc::hir::{self, Mutability};
use rustc::hir::Mutability::*;
use rustc::mir::{self, ValidationOp, ValidationOperand};
use rustc::ty::{self, Ty, TypeFoldable, TyCtxt};
use rustc::ty::layout::LayoutOf;
use rustc::ty::subst::{Substs, Subst};
use rustc::traits;
use rustc::infer::InferCtxt;
use rustc::traits::Reveal;
use rustc::middle::region;
use rustc_data_structures::indexed_vec::Idx;
use rustc_mir::interpret::HasMemory;

use super::{EvalContext, Place, PlaceExtra, ValTy};
use rustc::mir::interpret::{DynamicLifetime, AccessKind, EvalErrorKind, Value, EvalError, EvalResult};
use locks::MemoryExt;

pub type ValidationQuery<'tcx> = ValidationOperand<'tcx, (AbsPlace<'tcx>, Place)>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum ValidationMode {
    Acquire,
    /// Recover because the given region ended
    Recover(region::Scope),
    ReleaseUntil(Option<region::Scope>),
}

impl ValidationMode {
    fn acquiring(self) -> bool {
        use self::ValidationMode::*;
        match self {
            Acquire | Recover(_) => true,
            ReleaseUntil(_) => false,
        }
    }
}

// Abstract places
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbsPlace<'tcx> {
    Local(mir::Local),
    Static(hir::def_id::DefId),
    Projection(Box<AbsPlaceProjection<'tcx>>),
}

type AbsPlaceProjection<'tcx> = mir::Projection<'tcx, AbsPlace<'tcx>, u64, ()>;
type AbsPlaceElem<'tcx> = mir::ProjectionElem<'tcx, u64, ()>;

impl<'tcx> AbsPlace<'tcx> {
    pub fn field(self, f: mir::Field) -> AbsPlace<'tcx> {
        self.elem(mir::ProjectionElem::Field(f, ()))
    }

    pub fn deref(self) -> AbsPlace<'tcx> {
        self.elem(mir::ProjectionElem::Deref)
    }

    pub fn downcast(self, adt_def: &'tcx ty::AdtDef, variant_index: usize) -> AbsPlace<'tcx> {
        self.elem(mir::ProjectionElem::Downcast(adt_def, variant_index))
    }

    pub fn index(self, index: u64) -> AbsPlace<'tcx> {
        self.elem(mir::ProjectionElem::Index(index))
    }

    fn elem(self, elem: AbsPlaceElem<'tcx>) -> AbsPlace<'tcx> {
        AbsPlace::Projection(Box::new(AbsPlaceProjection {
            base: self,
            elem,
        }))
    }
}

pub(crate) trait EvalContextExt<'tcx> {
    fn abstract_place_projection(&self, proj: &mir::PlaceProjection<'tcx>) -> EvalResult<'tcx, AbsPlaceProjection<'tcx>>;
    fn abstract_place(&self, place: &mir::Place<'tcx>) -> EvalResult<'tcx, AbsPlace<'tcx>>;
    fn validation_op(
        &mut self,
        op: ValidationOp,
        operand: &ValidationOperand<'tcx, mir::Place<'tcx>>,
    ) -> EvalResult<'tcx>;
    fn end_region(&mut self, scope: Option<region::Scope>) -> EvalResult<'tcx>;
    fn normalize_type_unerased(&self, ty: Ty<'tcx>) -> Ty<'tcx>;
    fn field_with_lifetimes(
        &mut self,
        base: Place,
        layout: ty::layout::TyLayout<'tcx>,
        i: usize,
    ) -> EvalResult<'tcx, Ty<'tcx>>;
    fn validate_fields(
        &mut self,
        query: ValidationQuery<'tcx>,
        mode: ValidationMode,
    ) -> EvalResult<'tcx>;
    fn validate_ptr(
        &mut self,
        val: Value,
        abs_place: AbsPlace<'tcx>,
        pointee_ty: Ty<'tcx>,
        re: Option<region::Scope>,
        mutbl: Mutability,
        mode: ValidationMode,
    ) -> EvalResult<'tcx>;
    fn validate(
        &mut self,
        query: ValidationQuery<'tcx>,
        mode: ValidationMode,
    ) -> EvalResult<'tcx>;
}

impl<'a, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'tcx, super::Evaluator<'tcx>> {
    fn abstract_place_projection(&self, proj: &mir::PlaceProjection<'tcx>) -> EvalResult<'tcx, AbsPlaceProjection<'tcx>> {
        use self::mir::ProjectionElem::*;

        let elem = match proj.elem {
            Deref => Deref,
            Field(f, _) => Field(f, ()),
            Index(v) => {
                let value = self.frame().get_local(v)?;
                let ty = self.tcx.types.usize;
                let n = self.value_to_primval(ValTy { value, ty })?.to_u64()?;
                Index(n)
            },
            ConstantIndex { offset, min_length, from_end } =>
                ConstantIndex { offset, min_length, from_end },
            Subslice { from, to } =>
                Subslice { from, to },
            Downcast(adt, sz) => Downcast(adt, sz),
        };
        Ok(AbsPlaceProjection {
            base: self.abstract_place(&proj.base)?,
            elem
        })
    }

    fn abstract_place(&self, place: &mir::Place<'tcx>) -> EvalResult<'tcx, AbsPlace<'tcx>> {
        Ok(match place {
            &mir::Place::Local(l) => AbsPlace::Local(l),
            &mir::Place::Static(ref s) => AbsPlace::Static(s.def_id),
            &mir::Place::Projection(ref p) =>
                AbsPlace::Projection(Box::new(self.abstract_place_projection(&*p)?)),
        })
    }

    // Validity checks
    fn validation_op(
        &mut self,
        op: ValidationOp,
        operand: &ValidationOperand<'tcx, mir::Place<'tcx>>,
    ) -> EvalResult<'tcx> {
        // If mir-emit-validate is set to 0 (i.e., disabled), we may still see validation commands
        // because other crates may have been compiled with mir-emit-validate > 0.  Ignore those
        // commands.  This makes mir-emit-validate also a flag to control whether miri will do
        // validation or not.
        if self.tcx.sess.opts.debugging_opts.mir_emit_validate == 0 {
            return Ok(());
        }
        debug_assert!(self.memory.cur_frame == self.cur_frame());

        // HACK: Determine if this method is whitelisted and hence we do not perform any validation.
        // We currently insta-UB on anything passing around uninitialized memory, so we have to whitelist
        // the places that are allowed to do that.
        // The second group is stuff libstd does that is forbidden even under relaxed validation.
        {
            // The regexp we use for filtering
            use regex::Regex;
            lazy_static! {
                static ref RE: Regex = Regex::new("^(\
                    (std|alloc::heap::__core)::mem::(uninitialized|forget)::|\
                    <(std|alloc)::heap::Heap as (std::heap|alloc::allocator)::Alloc>::|\
                    <(std|alloc::heap::__core)::mem::ManuallyDrop<T>><.*>::new$|\
                    <(std|alloc::heap::__core)::mem::ManuallyDrop<T> as std::ops::DerefMut><.*>::deref_mut$|\
                    (std|alloc::heap::__core)::ptr::read::|\
                    \
                    <std::sync::Arc<T>><.*>::inner$|\
                    <std::sync::Arc<T>><.*>::drop_slow$|\
                    (std::heap|alloc::allocator)::Layout::for_value::|\
                    (std|alloc::heap::__core)::mem::(size|align)_of_val::\
                )").unwrap();
            }
            // Now test
            let name = self.frame().instance.to_string();
            if RE.is_match(&name) {
                return Ok(());
            }
        }

        // We need to monomorphize ty *without* erasing lifetimes
        trace!("validation_op1: {:?}", operand.ty.sty);
        let ty = operand.ty.subst(self.tcx, self.substs());
        trace!("validation_op2: {:?}", operand.ty.sty);
        let place = self.eval_place(&operand.place)?;
        let abs_place = self.abstract_place(&operand.place)?;
        let query = ValidationQuery {
            place: (abs_place, place),
            ty,
            re: operand.re,
            mutbl: operand.mutbl,
        };

        // Check the mode, and also perform mode-specific operations
        let mode = match op {
            ValidationOp::Acquire => ValidationMode::Acquire,
            ValidationOp::Release => ValidationMode::ReleaseUntil(None),
            ValidationOp::Suspend(scope) => {
                if query.mutbl == MutMutable {
                    let lft = DynamicLifetime {
                        frame: self.cur_frame(),
                        region: Some(scope), // Notably, we only ever suspend things for given regions.
                        // Suspending for the entire function does not make any sense.
                    };
                    trace!("Suspending {:?} until {:?}", query, scope);
                    self.machine.suspended.entry(lft).or_insert_with(Vec::new).push(
                        query.clone(),
                    );
                }
                ValidationMode::ReleaseUntil(Some(scope))
            }
        };
        self.validate(query, mode)
    }

    /// Release locks and executes suspensions of the given region (or the entire fn, in case of None).
    fn end_region(&mut self, scope: Option<region::Scope>) -> EvalResult<'tcx> {
        debug_assert!(self.memory.cur_frame == self.cur_frame());
        self.memory.locks_lifetime_ended(scope);
        match scope {
            Some(scope) => {
                // Recover suspended places
                let lft = DynamicLifetime {
                    frame: self.cur_frame(),
                    region: Some(scope),
                };
                if let Some(queries) = self.machine.suspended.remove(&lft) {
                    for query in queries {
                        trace!("Recovering {:?} from suspension", query);
                        self.validate(query, ValidationMode::Recover(scope))?;
                    }
                }
            }
            None => {
                // Clean suspension table of current frame
                let cur_frame = self.cur_frame();
                self.machine.suspended.retain(|lft, _| {
                    lft.frame != cur_frame // keep only what is in the other (lower) frames
                });
            }
        }
        Ok(())
    }

    fn normalize_type_unerased(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        return normalize_associated_type(self.tcx, &ty);

        use syntax::codemap::{Span, DUMMY_SP};

        // We copy a bunch of stuff from rustc/infer/mod.rs to be able to tweak its behavior
        fn normalize_projections_in<'a, 'gcx, 'tcx, T>(
            self_: &InferCtxt<'a, 'gcx, 'tcx>,
            param_env: ty::ParamEnv<'tcx>,
            value: &T,
        ) -> T::Lifted
        where
            T: TypeFoldable<'tcx> + ty::Lift<'gcx>,
        {
            let mut selcx = traits::SelectionContext::new(self_);
            let cause = traits::ObligationCause::dummy();
            let traits::Normalized {
                value: result,
                obligations,
            } = traits::normalize(&mut selcx, param_env, cause, value);

            let mut fulfill_cx = traits::FulfillmentContext::new();

            for obligation in obligations {
                fulfill_cx.register_predicate_obligation(self_, obligation);
            }

            drain_fulfillment_cx_or_panic(self_, DUMMY_SP, &mut fulfill_cx, &result)
        }

        fn drain_fulfillment_cx_or_panic<'a, 'gcx, 'tcx, T>(
            self_: &InferCtxt<'a, 'gcx, 'tcx>,
            span: Span,
            fulfill_cx: &mut traits::FulfillmentContext<'tcx>,
            result: &T,
        ) -> T::Lifted
        where
            T: TypeFoldable<'tcx> + ty::Lift<'gcx>,
        {
            // In principle, we only need to do this so long as `result`
            // contains unbound type parameters. It could be a slight
            // optimization to stop iterating early.
            match fulfill_cx.select_all_or_error(self_) {
                Ok(()) => { }
                Err(errors) => {
                    span_bug!(
                        span,
                        "Encountered errors `{:?}` resolving bounds after type-checking",
                        errors
                    );
                }
            }

            let result = self_.resolve_type_vars_if_possible(result);
            let result = self_.tcx.fold_regions(
                &result,
                &mut false,
                |r, _| match *r {
                    ty::ReVar(_) => self_.tcx.types.re_erased,
                    _ => r,
                },
            );

            match self_.tcx.lift_to_global(&result) {
                Some(result) => result,
                None => {
                    span_bug!(span, "Uninferred types/regions in `{:?}`", result);
                }
            }
        }

        trait MyTransNormalize<'gcx>: TypeFoldable<'gcx> {
            fn my_trans_normalize<'a, 'tcx>(
                &self,
                infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                param_env: ty::ParamEnv<'tcx>,
            ) -> Self;
        }

        macro_rules! items { ($($item:item)+) => ($($item)+) }
        macro_rules! impl_trans_normalize {
            ($lt_gcx:tt, $($ty:ty),+) => {
                items!($(impl<$lt_gcx> MyTransNormalize<$lt_gcx> for $ty {
                    fn my_trans_normalize<'a, 'tcx>(&self,
                                                infcx: &InferCtxt<'a, $lt_gcx, 'tcx>,
                                                param_env: ty::ParamEnv<'tcx>)
                                                -> Self {
                        normalize_projections_in(infcx, param_env, self)
                    }
                })+);
            }
        }

        impl_trans_normalize!('gcx,
            Ty<'gcx>,
            &'gcx Substs<'gcx>,
            ty::FnSig<'gcx>,
            ty::PolyFnSig<'gcx>,
            ty::ClosureSubsts<'gcx>,
            ty::PolyTraitRef<'gcx>,
            ty::ExistentialTraitRef<'gcx>
        );

        fn normalize_associated_type<'a, 'tcx, T>(self_: TyCtxt<'a, 'tcx, 'tcx>, value: &T) -> T
        where
            T: MyTransNormalize<'tcx>,
        {
            let param_env = ty::ParamEnv::empty(Reveal::All);

            if !value.has_projections() {
                return value.clone();
            }

            self_.infer_ctxt().enter(|infcx| {
                value.my_trans_normalize(&infcx, param_env)
            })
        }
    }

    // This is a copy of `Layout::field`
    //
    // FIXME: remove once validation does not depend on lifetimes
    fn field_with_lifetimes(
        &mut self,
        base: Place,
        mut layout: ty::layout::TyLayout<'tcx>,
        i: usize,
    ) -> EvalResult<'tcx, Ty<'tcx>> {
        match base {
            Place::Ptr { extra: PlaceExtra::DowncastVariant(variant_index), .. } => {
                layout = layout.for_variant(&self, variant_index);
            }
            _ => {}
        }
        let tcx = self.tcx;
        Ok(match layout.ty.sty {
            ty::TyBool |
            ty::TyChar |
            ty::TyInt(_) |
            ty::TyUint(_) |
            ty::TyFloat(_) |
            ty::TyFnPtr(_) |
            ty::TyNever |
            ty::TyFnDef(..) |
            ty::TyDynamic(..) |
            ty::TyForeign(..) => {
                bug!("TyLayout::field_type({:?}): not applicable", layout)
            }

            // Potentially-fat pointers.
            ty::TyRef(_, ty::TypeAndMut { ty: pointee, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty: pointee, .. }) => {
                assert!(i < 2);

                // Reuse the fat *T type as its own thin pointer data field.
                // This provides information about e.g. DST struct pointees
                // (which may have no non-DST form), and will work as long
                // as the `Abi` or `FieldPlacement` is checked by users.
                if i == 0 {
                    return Ok(layout.ty);
                }

                match tcx.struct_tail(pointee).sty {
                    ty::TySlice(_) |
                    ty::TyStr => tcx.types.usize,
                    ty::TyDynamic(..) => {
                        // FIXME(eddyb) use an usize/fn() array with
                        // the correct number of vtables slots.
                        tcx.mk_imm_ref(tcx.types.re_static, tcx.mk_nil())
                    }
                    _ => bug!("TyLayout::field_type({:?}): not applicable", layout)
                }
            }

            // Arrays and slices.
            ty::TyArray(element, _) |
            ty::TySlice(element) => element,
            ty::TyStr => tcx.types.u8,

            // Tuples, generators and closures.
            ty::TyClosure(def_id, ref substs) => {
                substs.upvar_tys(def_id, tcx).nth(i).unwrap()
            }

            ty::TyGenerator(def_id, ref substs, _) => {
                substs.field_tys(def_id, tcx).nth(i).unwrap()
            }

            ty::TyTuple(tys, _) => tys[i],

            // SIMD vector types.
            ty::TyAdt(def, ..) if def.repr.simd() => {
                layout.ty.simd_type(tcx)
            }

            // ADTs.
            ty::TyAdt(def, substs) => {
                use rustc::ty::layout::Variants;
                match layout.variants {
                    Variants::Single { index } => {
                        def.variants[index].fields[i].ty(tcx, substs)
                    }

                    // Discriminant field for enums (where applicable).
                    Variants::Tagged { ref discr, .. } |
                    Variants::NicheFilling { niche: ref discr, .. } => {
                        assert_eq!(i, 0);
                        return Ok(discr.value.to_ty(tcx))
                    }
                }
            }

            ty::TyProjection(_) | ty::TyAnon(..) | ty::TyParam(_) |
            ty::TyInfer(_) | ty::TyError => {
                bug!("TyLayout::field_type: unexpected type `{}`", layout.ty)
            }
        })
    }

    fn validate_fields(
        &mut self,
        query: ValidationQuery<'tcx>,
        mode: ValidationMode,
    ) -> EvalResult<'tcx> {
        let mut layout = self.layout_of(query.ty)?;
        layout.ty = query.ty;

        // TODO: Maybe take visibility/privacy into account.
        for idx in 0..layout.fields.count() {
            let field = mir::Field::new(idx);
            let (field_place, field_layout) =
                self.place_field(query.place.1, field, layout)?;
            // layout stuff erases lifetimes, get the field ourselves
            let field_ty = self.field_with_lifetimes(query.place.1, layout, idx)?;
            trace!("assuming \n{:?}\n == \n{:?}\n except for lifetimes", field_layout.ty, field_ty);
            self.validate(
                ValidationQuery {
                    place: (query.place.0.clone().field(field), field_place),
                    ty: field_ty,
                    ..query
                },
                mode,
            )?;
        }

        Ok(())
    }

    fn validate_ptr(
        &mut self,
        val: Value,
        abs_place: AbsPlace<'tcx>,
        pointee_ty: Ty<'tcx>,
        re: Option<region::Scope>,
        mutbl: Mutability,
        mode: ValidationMode,
    ) -> EvalResult<'tcx> {
        // Check alignment and non-NULLness
        let (_, align) = self.size_and_align_of_dst(pointee_ty, val)?;
        let ptr = self.into_ptr(val)?;
        self.memory.check_align(ptr, align.abi(), None)?;

        // Recurse
        let pointee_place = self.val_to_place(val, pointee_ty)?;
        self.validate(
            ValidationQuery {
                place: (abs_place.deref(), pointee_place),
                ty: pointee_ty,
                re,
                mutbl,
            },
            mode,
        )
    }

    /// Validate the place at the given type. If `acquire` is false, just do a release of all write locks
    fn validate(
        &mut self,
        mut query: ValidationQuery<'tcx>,
        mode: ValidationMode,
    ) -> EvalResult<'tcx> {
        use rustc::ty::TypeVariants::*;
        use rustc::ty::RegionKind::*;
        use rustc::ty::AdtKind;

        // No point releasing shared stuff.
        if !mode.acquiring() && query.mutbl == MutImmutable {
            return Ok(());
        }
        // When we recover, we may see data whose validity *just* ended.  Do not acquire it.
        if let ValidationMode::Recover(ending_ce) = mode {
            if query.re == Some(ending_ce) {
                return Ok(());
            }
        }

        query.ty = self.normalize_type_unerased(&query.ty);
        trace!("{:?} on {:#?}", mode, query);
        trace!("{:#?}", query.ty.sty);

        // Decide whether this type *owns* the memory it covers (like integers), or whether it
        // just assembles pieces (that each own their memory) together to a larger whole.
        // TODO: Currently, we don't acquire locks for padding and discriminants. We should.
        let is_owning = match query.ty.sty {
            TyInt(_) | TyUint(_) | TyRawPtr(_) | TyBool | TyFloat(_) | TyChar | TyStr |
            TyRef(..) | TyFnPtr(..) | TyFnDef(..) | TyNever => true,
            TyAdt(adt, _) if adt.is_box() => true,
            TySlice(_) | TyAdt(_, _) | TyTuple(..) | TyClosure(..) | TyArray(..) |
            TyDynamic(..) | TyGenerator(..) | TyForeign(_) => false,
            TyParam(_) | TyInfer(_) | TyProjection(_) | TyAnon(..) | TyError => {
                bug!("I got an incomplete/unnormalized type for validation")
            }
        };
        if is_owning {
            // We need to lock.  So we need memory.  So we have to force_acquire.
            // Tracking the same state for locals not backed by memory would just duplicate too
            // much machinery.
            // FIXME: We ignore alignment.
            let (ptr, extra) = self.force_allocation(query.place.1)?.to_ptr_extra_aligned();
            // Determine the size
            // FIXME: Can we reuse size_and_align_of_dst for Places?
            let layout = self.layout_of(query.ty)?;
            let len = if !layout.is_unsized() {
                assert_eq!(extra, PlaceExtra::None, "Got a fat ptr to a sized type");
                layout.size.bytes()
            } else {
                // The only unsized typ we concider "owning" is TyStr.
                assert_eq!(
                    query.ty.sty,
                    TyStr,
                    "Found a surprising unsized owning type"
                );
                // The extra must be the length, in bytes.
                match extra {
                    PlaceExtra::Length(len) => len,
                    _ => bug!("TyStr must have a length as extra"),
                }
            };
            // Handle locking
            if len > 0 {
                let ptr = ptr.to_ptr()?;
                match query.mutbl {
                    MutImmutable => {
                        if mode.acquiring() {
                            self.memory.acquire_lock(
                                ptr,
                                len,
                                query.re,
                                AccessKind::Read,
                            )?;
                        }
                    }
                    // No releasing of read locks, ever.
                    MutMutable => {
                        match mode {
                            ValidationMode::Acquire => {
                                self.memory.acquire_lock(
                                    ptr,
                                    len,
                                    query.re,
                                    AccessKind::Write,
                                )?
                            }
                            ValidationMode::Recover(ending_ce) => {
                                self.memory.recover_write_lock(
                                    ptr,
                                    len,
                                    &query.place.0,
                                    query.re,
                                    ending_ce,
                                )?
                            }
                            ValidationMode::ReleaseUntil(suspended_ce) => {
                                self.memory.suspend_write_lock(
                                    ptr,
                                    len,
                                    &query.place.0,
                                    suspended_ce,
                                )?
                            }
                        }
                    }
                }
            }
        }

        let res = do catch {
            match query.ty.sty {
                TyInt(_) | TyUint(_) | TyRawPtr(_) => {
                    if mode.acquiring() {
                        // Make sure we can read this.
                        let val = self.read_place(query.place.1)?;
                        self.follow_by_ref_value(val, query.ty)?;
                        // FIXME: It would be great to rule out Undef here, but that doesn't actually work.
                        // Passing around undef data is a thing that e.g. Vec::extend_with does.
                    }
                    Ok(())
                }
                TyBool | TyFloat(_) | TyChar => {
                    if mode.acquiring() {
                        let val = self.read_place(query.place.1)?;
                        let val = self.value_to_primval(ValTy { value: val, ty: query.ty })?;
                        val.to_bytes()?;
                        // TODO: Check if these are valid bool/float/codepoint/UTF-8
                    }
                    Ok(())
                }
                TyNever => err!(ValidationFailure(format!("The empty type is never valid."))),
                TyRef(region,
                    ty::TypeAndMut {
                        ty: pointee_ty,
                        mutbl,
                    }) => {
                    let val = self.read_place(query.place.1)?;
                    // Sharing restricts our context
                    if mutbl == MutImmutable {
                        query.mutbl = MutImmutable;
                    }
                    // Inner lifetimes *outlive* outer ones, so only if we have no lifetime restriction yet,
                    // we record the region of this borrow to the context.
                    if query.re == None {
                        match *region {
                            ReScope(scope) => query.re = Some(scope),
                            // It is possible for us to encounter erased lifetimes here because the lifetimes in
                            // this functions' Subst will be erased.
                            _ => {}
                        }
                    }
                    self.validate_ptr(val, query.place.0, pointee_ty, query.re, query.mutbl, mode)
                }
                TyAdt(adt, _) if adt.is_box() => {
                    let val = self.read_place(query.place.1)?;
                    self.validate_ptr(val, query.place.0, query.ty.boxed_ty(), query.re, query.mutbl, mode)
                }
                TyFnPtr(_sig) => {
                    let ptr = self.read_place(query.place.1)?;
                    let ptr = self.into_ptr(ptr)?.to_ptr()?;
                    self.memory.get_fn(ptr)?;
                    // TODO: Check if the signature matches (should be the same check as what terminator/mod.rs already does on call?).
                    Ok(())
                }
                TyFnDef(..) => {
                    // This is a zero-sized type with all relevant data sitting in the type.
                    // There is nothing to validate.
                    Ok(())
                }

                // Compound types
                TyStr => {
                    // TODO: Validate strings
                    Ok(())
                }
                TySlice(elem_ty) => {
                    let len = match query.place.1 {
                        Place::Ptr { extra: PlaceExtra::Length(len), .. } => len,
                        _ => {
                            bug!(
                                "acquire_valid of a TySlice given non-slice place: {:?}",
                                query.place
                            )
                        }
                    };
                    for i in 0..len {
                        let inner_place = self.place_index(query.place.1, query.ty, i)?;
                        self.validate(
                            ValidationQuery {
                                place: (query.place.0.clone().index(i), inner_place),
                                ty: elem_ty,
                                ..query
                            },
                            mode,
                        )?;
                    }
                    Ok(())
                }
                TyArray(elem_ty, len) => {
                    let len = len.val.to_const_int().unwrap().to_u64().unwrap();
                    for i in 0..len {
                        let inner_place = self.place_index(query.place.1, query.ty, i as u64)?;
                        self.validate(
                            ValidationQuery {
                                place: (query.place.0.clone().index(i as u64), inner_place),
                                ty: elem_ty,
                                ..query
                            },
                            mode,
                        )?;
                    }
                    Ok(())
                }
                TyDynamic(_data, _region) => {
                    // Check that this is a valid vtable
                    let vtable = match query.place.1 {
                        Place::Ptr { extra: PlaceExtra::Vtable(vtable), .. } => vtable,
                        _ => {
                            bug!(
                                "acquire_valid of a TyDynamic given non-trait-object place: {:?}",
                                query.place
                            )
                        }
                    };
                    self.read_size_and_align_from_vtable(vtable)?;
                    // TODO: Check that the vtable contains all the function pointers we expect it to have.
                    // Trait objects cannot have any operations performed
                    // on them directly.  We cannot, in general, even acquire any locks as the trait object *could*
                    // contain an UnsafeCell.  If we call functions to get access to data, we will validate
                    // their return values.  So, it doesn't seem like there's anything else to do.
                    Ok(())
                }
                TyAdt(adt, _) => {
                    if Some(adt.did) == self.tcx.lang_items().unsafe_cell_type() &&
                        query.mutbl == MutImmutable
                    {
                        // No locks for shared unsafe cells.  Also no other validation, the only field is private anyway.
                        return Ok(());
                    }

                    match adt.adt_kind() {
                        AdtKind::Enum => {
                            let discr = self.read_discriminant_value(query.place.1, query.ty)?;

                            // Get variant index for discriminant
                            let variant_idx = adt.discriminants(self.tcx).position(|variant_discr| {
                                variant_discr.to_u128_unchecked() == discr
                            });
                            let variant_idx = match variant_idx {
                                Some(val) => val,
                                None => return err!(InvalidDiscriminant),
                            };
                            let variant = &adt.variants[variant_idx];

                            if variant.fields.len() > 0 {
                                // Downcast to this variant, if needed
                                let place = if adt.is_enum() {
                                    (
                                        query.place.0.downcast(adt, variant_idx),
                                        self.eval_place_projection(
                                            query.place.1,
                                            query.ty,
                                            &mir::ProjectionElem::Downcast(adt, variant_idx),
                                        )?,
                                    )
                                } else {
                                    query.place
                                };

                                // Recursively validate the fields
                                self.validate_fields(
                                    ValidationQuery { place, ..query },
                                    mode,
                                )
                            } else {
                                // No fields, nothing left to check.  Downcasting may fail, e.g. in case of a CEnum.
                                Ok(())
                            }
                        }
                        AdtKind::Struct => {
                            self.validate_fields(query, mode)
                        }
                        AdtKind::Union => {
                            // No guarantees are provided for union types.
                            // TODO: Make sure that all access to union fields is unsafe; otherwise, we may have some checking to do (but what exactly?)
                            Ok(())
                        }
                    }
                }
                TyTuple(..) |
                TyClosure(..) => {
                    // TODO: Check if the signature matches for `TyClosure`
                    // (should be the same check as what terminator/mod.rs already does on call?).
                    // Is there other things we can/should check?  Like vtable pointers?
                    self.validate_fields(query, mode)
                }
                // FIXME: generators aren't validated right now
                TyGenerator(..) => Ok(()),
                _ => bug!("We already established that this is a type we support. ({})", query.ty),
            }
        };
        match res {
            // ReleaseUntil(None) of an uninitalized variable is a NOP.  This is needed because
            // we have to release the return value of a function; due to destination-passing-style
            // the callee may directly write there.
            // TODO: Ideally we would know whether the destination is already initialized, and only
            // release if it is.  But of course that can't even always be statically determined.
            Err(EvalError { kind: EvalErrorKind::ReadUndefBytes, .. })
                if mode == ValidationMode::ReleaseUntil(None) => {
                return Ok(());
            }
            res => res,
        }
    }
}
