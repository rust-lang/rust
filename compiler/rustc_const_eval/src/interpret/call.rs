//! Manages calling a concrete function (with known MIR body) with argument passing,
//! and returning the return value to the caller.
use std::assert_matches::assert_matches;
use std::borrow::Cow;

use either::{Left, Right};
use rustc_abi::{self as abi, ExternAbi, FieldIdx, Integer, VariantIdx};
use rustc_hir::def_id::DefId;
use rustc_middle::ty::layout::{FnAbiOf, IntegerExt, TyAndLayout};
use rustc_middle::ty::{self, AdtDef, Instance, Ty, VariantDef};
use rustc_middle::{bug, mir, span_bug};
use rustc_span::sym;
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use tracing::{info, instrument, trace};

use super::{
    CtfeProvenance, FnVal, ImmTy, InterpCx, InterpResult, MPlaceTy, Machine, OpTy, PlaceTy,
    Projectable, Provenance, ReturnAction, Scalar, StackPopCleanup, StackPopInfo, interp_ok,
    throw_ub, throw_ub_custom, throw_unsup_format,
};
use crate::fluent_generated as fluent;

/// An argument passed to a function.
#[derive(Clone, Debug)]
pub enum FnArg<'tcx, Prov: Provenance = CtfeProvenance> {
    /// Pass a copy of the given operand.
    Copy(OpTy<'tcx, Prov>),
    /// Allow for the argument to be passed in-place: destroy the value originally stored at that place and
    /// make the place inaccessible for the duration of the function call.
    InPlace(MPlaceTy<'tcx, Prov>),
}

impl<'tcx, Prov: Provenance> FnArg<'tcx, Prov> {
    pub fn layout(&self) -> &TyAndLayout<'tcx> {
        match self {
            FnArg::Copy(op) => &op.layout,
            FnArg::InPlace(mplace) => &mplace.layout,
        }
    }
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Make a copy of the given fn_arg. Any `InPlace` are degenerated to copies, no protection of the
    /// original memory occurs.
    pub fn copy_fn_arg(&self, arg: &FnArg<'tcx, M::Provenance>) -> OpTy<'tcx, M::Provenance> {
        match arg {
            FnArg::Copy(op) => op.clone(),
            FnArg::InPlace(mplace) => mplace.clone().into(),
        }
    }

    /// Make a copy of the given fn_args. Any `InPlace` are degenerated to copies, no protection of the
    /// original memory occurs.
    pub fn copy_fn_args(
        &self,
        args: &[FnArg<'tcx, M::Provenance>],
    ) -> Vec<OpTy<'tcx, M::Provenance>> {
        args.iter().map(|fn_arg| self.copy_fn_arg(fn_arg)).collect()
    }

    /// Helper function for argument untupling.
    pub(super) fn fn_arg_field(
        &self,
        arg: &FnArg<'tcx, M::Provenance>,
        field: FieldIdx,
    ) -> InterpResult<'tcx, FnArg<'tcx, M::Provenance>> {
        interp_ok(match arg {
            FnArg::Copy(op) => FnArg::Copy(self.project_field(op, field)?),
            FnArg::InPlace(mplace) => FnArg::InPlace(self.project_field(mplace, field)?),
        })
    }

    /// Find the wrapped inner type of a transparent wrapper.
    /// Must not be called on 1-ZST (as they don't have a uniquely defined "wrapped field").
    ///
    /// We work with `TyAndLayout` here since that makes it much easier to iterate over all fields.
    fn unfold_transparent(
        &self,
        layout: TyAndLayout<'tcx>,
        may_unfold: impl Fn(AdtDef<'tcx>) -> bool,
    ) -> TyAndLayout<'tcx> {
        match layout.ty.kind() {
            ty::Adt(adt_def, _) if adt_def.repr().transparent() && may_unfold(*adt_def) => {
                assert!(!adt_def.is_enum());
                // Find the non-1-ZST field, and recurse.
                let (_, field) = layout.non_1zst_field(self).unwrap();
                self.unfold_transparent(field, may_unfold)
            }
            // Not a transparent type, no further unfolding.
            _ => layout,
        }
    }

    /// Unwrap types that are guaranteed a null-pointer-optimization
    fn unfold_npo(&self, layout: TyAndLayout<'tcx>) -> InterpResult<'tcx, TyAndLayout<'tcx>> {
        // Check if this is an option-like type wrapping some type.
        let ty::Adt(def, args) = layout.ty.kind() else {
            // Not an ADT, so definitely no NPO.
            return interp_ok(layout);
        };
        if def.variants().len() != 2 {
            // Not a 2-variant enum, so no NPO.
            return interp_ok(layout);
        }
        assert!(def.is_enum());

        let all_fields_1zst = |variant: &VariantDef| -> InterpResult<'tcx, _> {
            for field in &variant.fields {
                let ty = field.ty(*self.tcx, args);
                let layout = self.layout_of(ty)?;
                if !layout.is_1zst() {
                    return interp_ok(false);
                }
            }
            interp_ok(true)
        };

        // If one variant consists entirely of 1-ZST, then the other variant
        // is the only "relevant" one for this check.
        let var0 = VariantIdx::from_u32(0);
        let var1 = VariantIdx::from_u32(1);
        let relevant_variant = if all_fields_1zst(def.variant(var0))? {
            def.variant(var1)
        } else if all_fields_1zst(def.variant(var1))? {
            def.variant(var0)
        } else {
            // No varant is all-1-ZST, so no NPO.
            return interp_ok(layout);
        };
        // The "relevant" variant must have exactly one field, and its type is the "inner" type.
        if relevant_variant.fields.len() != 1 {
            return interp_ok(layout);
        }
        let inner = relevant_variant.fields[FieldIdx::from_u32(0)].ty(*self.tcx, args);
        let inner = self.layout_of(inner)?;

        // Check if the inner type is one of the NPO-guaranteed ones.
        // For that we first unpeel transparent *structs* (but not unions).
        let is_npo = |def: AdtDef<'tcx>| {
            self.tcx.has_attr(def.did(), sym::rustc_nonnull_optimization_guaranteed)
        };
        let inner = self.unfold_transparent(inner, /* may_unfold */ |def| {
            // Stop at NPO types so that we don't miss that attribute in the check below!
            def.is_struct() && !is_npo(def)
        });
        interp_ok(match inner.ty.kind() {
            ty::Ref(..) | ty::FnPtr(..) => {
                // Option<&T> behaves like &T, and same for fn()
                inner
            }
            ty::Adt(def, _) if is_npo(*def) => {
                // Once we found a `nonnull_optimization_guaranteed` type, further strip off
                // newtype structs from it to find the underlying ABI type.
                self.unfold_transparent(inner, /* may_unfold */ |def| def.is_struct())
            }
            _ => {
                // Everything else we do not unfold.
                layout
            }
        })
    }

    /// Check if these two layouts look like they are fn-ABI-compatible.
    /// (We also compare the `PassMode`, so this doesn't have to check everything. But it turns out
    /// that only checking the `PassMode` is insufficient.)
    fn layout_compat(
        &self,
        caller: TyAndLayout<'tcx>,
        callee: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, bool> {
        // Fast path: equal types are definitely compatible.
        if caller.ty == callee.ty {
            return interp_ok(true);
        }
        // 1-ZST are compatible with all 1-ZST (and with nothing else).
        if caller.is_1zst() || callee.is_1zst() {
            return interp_ok(caller.is_1zst() && callee.is_1zst());
        }
        // Unfold newtypes and NPO optimizations.
        let unfold = |layout: TyAndLayout<'tcx>| {
            self.unfold_npo(self.unfold_transparent(layout, /* may_unfold */ |_def| true))
        };
        let caller = unfold(caller)?;
        let callee = unfold(callee)?;
        // Now see if these inner types are compatible.

        // Compatible pointer types. For thin pointers, we have to accept even non-`repr(transparent)`
        // things as compatible due to `DispatchFromDyn`. For instance, `Rc<i32>` and `*mut i32`
        // must be compatible. So we just accept everything with Pointer ABI as compatible,
        // even if this will accept some code that is not stably guaranteed to work.
        // This also handles function pointers.
        let thin_pointer = |layout: TyAndLayout<'tcx>| match layout.backend_repr {
            abi::BackendRepr::Scalar(s) => match s.primitive() {
                abi::Primitive::Pointer(addr_space) => Some(addr_space),
                _ => None,
            },
            _ => None,
        };
        if let (Some(caller), Some(callee)) = (thin_pointer(caller), thin_pointer(callee)) {
            return interp_ok(caller == callee);
        }
        // For wide pointers we have to get the pointee type.
        let pointee_ty = |ty: Ty<'tcx>| -> InterpResult<'tcx, Option<Ty<'tcx>>> {
            // We cannot use `builtin_deref` here since we need to reject `Box<T, MyAlloc>`.
            interp_ok(Some(match ty.kind() {
                ty::Ref(_, ty, _) => *ty,
                ty::RawPtr(ty, _) => *ty,
                // We only accept `Box` with the default allocator.
                _ if ty.is_box_global(*self.tcx) => ty.expect_boxed_ty(),
                _ => return interp_ok(None),
            }))
        };
        if let (Some(caller), Some(callee)) = (pointee_ty(caller.ty)?, pointee_ty(callee.ty)?) {
            // This is okay if they have the same metadata type.
            let meta_ty = |ty: Ty<'tcx>| {
                // Even if `ty` is normalized, the search for the unsized tail will project
                // to fields, which can yield non-normalized types. So we need to provide a
                // normalization function.
                let normalize = |ty| self.tcx.normalize_erasing_regions(self.typing_env, ty);
                ty.ptr_metadata_ty(*self.tcx, normalize)
            };
            return interp_ok(meta_ty(caller) == meta_ty(callee));
        }

        // Compatible integer types (in particular, usize vs ptr-sized-u32/u64).
        // `char` counts as `u32.`
        let int_ty = |ty: Ty<'tcx>| {
            Some(match ty.kind() {
                ty::Int(ity) => (Integer::from_int_ty(&self.tcx, *ity), /* signed */ true),
                ty::Uint(uty) => (Integer::from_uint_ty(&self.tcx, *uty), /* signed */ false),
                ty::Char => (Integer::I32, /* signed */ false),
                _ => return None,
            })
        };
        if let (Some(caller), Some(callee)) = (int_ty(caller.ty), int_ty(callee.ty)) {
            // This is okay if they are the same integer type.
            return interp_ok(caller == callee);
        }

        // Fall back to exact equality.
        interp_ok(caller == callee)
    }

    /// Returns a `bool` saying whether the two arguments are ABI-compatible.
    pub fn check_argument_compat(
        &self,
        caller_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        callee_abi: &ArgAbi<'tcx, Ty<'tcx>>,
    ) -> InterpResult<'tcx, bool> {
        // We do not want to accept things as ABI-compatible that just "happen to be" compatible on the current target,
        // so we implement a type-based check that reflects the guaranteed rules for ABI compatibility.
        if self.layout_compat(caller_abi.layout, callee_abi.layout)? {
            // Ensure that our checks imply actual ABI compatibility for this concrete call.
            // (This can fail e.g. if `#[rustc_nonnull_optimization_guaranteed]` is used incorrectly.)
            assert!(caller_abi.eq_abi(callee_abi));
            interp_ok(true)
        } else {
            trace!(
                "check_argument_compat: incompatible ABIs:\ncaller: {:?}\ncallee: {:?}",
                caller_abi, callee_abi
            );
            interp_ok(false)
        }
    }

    /// Initialize a single callee argument, checking the types for compatibility.
    fn pass_argument<'x, 'y>(
        &mut self,
        caller_args: &mut impl Iterator<
            Item = (&'x FnArg<'tcx, M::Provenance>, &'y ArgAbi<'tcx, Ty<'tcx>>),
        >,
        callee_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        callee_arg: &mir::Place<'tcx>,
        callee_ty: Ty<'tcx>,
        already_live: bool,
    ) -> InterpResult<'tcx>
    where
        'tcx: 'x,
        'tcx: 'y,
    {
        assert_eq!(callee_ty, callee_abi.layout.ty);
        if matches!(callee_abi.mode, PassMode::Ignore) {
            // This one is skipped. Still must be made live though!
            if !already_live {
                self.storage_live(callee_arg.as_local().unwrap())?;
            }
            return interp_ok(());
        }
        // Find next caller arg.
        let Some((caller_arg, caller_abi)) = caller_args.next() else {
            throw_ub_custom!(fluent::const_eval_not_enough_caller_args);
        };
        assert_eq!(caller_arg.layout().layout, caller_abi.layout.layout);
        // Sadly we cannot assert that `caller_arg.layout().ty` and `caller_abi.layout.ty` are
        // equal; in closures the types sometimes differ. We just hope that `caller_abi` is the
        // right type to print to the user.

        // Check compatibility
        if !self.check_argument_compat(caller_abi, callee_abi)? {
            throw_ub!(AbiMismatchArgument {
                caller_ty: caller_abi.layout.ty,
                callee_ty: callee_abi.layout.ty
            });
        }
        // We work with a copy of the argument for now; if this is in-place argument passing, we
        // will later protect the source it comes from. This means the callee cannot observe if we
        // did in-place of by-copy argument passing, except for pointer equality tests.
        let caller_arg_copy = self.copy_fn_arg(caller_arg);
        if !already_live {
            let local = callee_arg.as_local().unwrap();
            let meta = caller_arg_copy.meta();
            // `check_argument_compat` ensures that if metadata is needed, both have the same type,
            // so we know they will use the metadata the same way.
            assert!(!meta.has_meta() || caller_arg_copy.layout.ty == callee_ty);

            self.storage_live_dyn(local, meta)?;
        }
        // Now we can finally actually evaluate the callee place.
        let callee_arg = self.eval_place(*callee_arg)?;
        // We allow some transmutes here.
        // FIXME: Depending on the PassMode, this should reset some padding to uninitialized. (This
        // is true for all `copy_op`, but there are a lot of special cases for argument passing
        // specifically.)
        self.copy_op_allow_transmute(&caller_arg_copy, &callee_arg)?;
        // If this was an in-place pass, protect the place it comes from for the duration of the call.
        if let FnArg::InPlace(mplace) = caller_arg {
            M::protect_in_place_function_argument(self, mplace)?;
        }
        interp_ok(())
    }

    /// The main entry point for creating a new stack frame: performs ABI checks and initializes
    /// arguments.
    #[instrument(skip(self), level = "trace")]
    pub fn init_stack_frame(
        &mut self,
        instance: Instance<'tcx>,
        body: &'tcx mir::Body<'tcx>,
        caller_fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[FnArg<'tcx, M::Provenance>],
        with_caller_location: bool,
        destination: &PlaceTy<'tcx, M::Provenance>,
        mut stack_pop: StackPopCleanup,
    ) -> InterpResult<'tcx> {
        // Compute callee information.
        // FIXME: for variadic support, do we have to somehow determine callee's extra_args?
        let callee_fn_abi = self.fn_abi_of_instance(instance, ty::List::empty())?;

        if callee_fn_abi.c_variadic || caller_fn_abi.c_variadic {
            throw_unsup_format!("calling a c-variadic function is not supported");
        }

        if caller_fn_abi.conv != callee_fn_abi.conv {
            throw_ub_custom!(
                fluent::const_eval_incompatible_calling_conventions,
                callee_conv = format!("{}", callee_fn_abi.conv),
                caller_conv = format!("{}", caller_fn_abi.conv),
            )
        }

        // Check that all target features required by the callee (i.e., from
        // the attribute `#[target_feature(enable = ...)]`) are enabled at
        // compile time.
        M::check_fn_target_features(self, instance)?;

        if !callee_fn_abi.can_unwind {
            // The callee cannot unwind, so force the `Unreachable` unwind handling.
            match &mut stack_pop {
                StackPopCleanup::Root { .. } => {}
                StackPopCleanup::Goto { unwind, .. } => {
                    *unwind = mir::UnwindAction::Unreachable;
                }
            }
        }

        self.push_stack_frame_raw(instance, body, destination, stack_pop)?;

        // If an error is raised here, pop the frame again to get an accurate backtrace.
        // To this end, we wrap it all in a `try` block.
        let res: InterpResult<'tcx> = try {
            trace!(
                "caller ABI: {:#?}, args: {:#?}",
                caller_fn_abi,
                args.iter()
                    .map(|arg| (
                        arg.layout().ty,
                        match arg {
                            FnArg::Copy(op) => format!("copy({op:?})"),
                            FnArg::InPlace(mplace) => format!("in-place({mplace:?})"),
                        }
                    ))
                    .collect::<Vec<_>>()
            );
            trace!(
                "spread_arg: {:?}, locals: {:#?}",
                body.spread_arg,
                body.args_iter()
                    .map(|local| (
                        local,
                        self.layout_of_local(self.frame(), local, None).unwrap().ty,
                    ))
                    .collect::<Vec<_>>()
            );

            // In principle, we have two iterators: Where the arguments come from, and where
            // they go to.

            // The "where they come from" part is easy, we expect the caller to do any special handling
            // that might be required here (e.g. for untupling).
            // If `with_caller_location` is set we pretend there is an extra argument (that
            // we will not pass; our `caller_location` intrinsic implementation walks the stack instead).
            assert_eq!(
                args.len() + if with_caller_location { 1 } else { 0 },
                caller_fn_abi.args.len(),
                "mismatch between caller ABI and caller arguments",
            );
            let mut caller_args = args
                .iter()
                .zip(caller_fn_abi.args.iter())
                .filter(|arg_and_abi| !matches!(arg_and_abi.1.mode, PassMode::Ignore));

            // Now we have to spread them out across the callee's locals,
            // taking into account the `spread_arg`. If we could write
            // this is a single iterator (that handles `spread_arg`), then
            // `pass_argument` would be the loop body. It takes care to
            // not advance `caller_iter` for ignored arguments.
            let mut callee_args_abis = callee_fn_abi.args.iter();
            for local in body.args_iter() {
                // Construct the destination place for this argument. At this point all
                // locals are still dead, so we cannot construct a `PlaceTy`.
                let dest = mir::Place::from(local);
                // `layout_of_local` does more than just the instantiation we need to get the
                // type, but the result gets cached so this avoids calling the instantiation
                // query *again* the next time this local is accessed.
                let ty = self.layout_of_local(self.frame(), local, None)?.ty;
                if Some(local) == body.spread_arg {
                    // Make the local live once, then fill in the value field by field.
                    self.storage_live(local)?;
                    // Must be a tuple
                    let ty::Tuple(fields) = ty.kind() else {
                        span_bug!(self.cur_span(), "non-tuple type for `spread_arg`: {ty}")
                    };
                    for (i, field_ty) in fields.iter().enumerate() {
                        let dest = dest.project_deeper(
                            &[mir::ProjectionElem::Field(FieldIdx::from_usize(i), field_ty)],
                            *self.tcx,
                        );
                        let callee_abi = callee_args_abis.next().unwrap();
                        self.pass_argument(
                            &mut caller_args,
                            callee_abi,
                            &dest,
                            field_ty,
                            /* already_live */ true,
                        )?;
                    }
                } else {
                    // Normal argument. Cannot mark it as live yet, it might be unsized!
                    let callee_abi = callee_args_abis.next().unwrap();
                    self.pass_argument(
                        &mut caller_args,
                        callee_abi,
                        &dest,
                        ty,
                        /* already_live */ false,
                    )?;
                }
            }
            // If the callee needs a caller location, pretend we consume one more argument from the ABI.
            if instance.def.requires_caller_location(*self.tcx) {
                callee_args_abis.next().unwrap();
            }
            // Now we should have no more caller args or callee arg ABIs
            assert!(
                callee_args_abis.next().is_none(),
                "mismatch between callee ABI and callee body arguments"
            );
            if caller_args.next().is_some() {
                throw_ub_custom!(fluent::const_eval_too_many_caller_args);
            }
            // Don't forget to check the return type!
            if !self.check_argument_compat(&caller_fn_abi.ret, &callee_fn_abi.ret)? {
                throw_ub!(AbiMismatchReturn {
                    caller_ty: caller_fn_abi.ret.layout.ty,
                    callee_ty: callee_fn_abi.ret.layout.ty
                });
            }

            // Protect return place for in-place return value passing.
            // We only need to protect anything if this is actually an in-memory place.
            if let Left(mplace) = destination.as_mplace_or_local() {
                M::protect_in_place_function_argument(self, &mplace)?;
            }

            // Don't forget to mark "initially live" locals as live.
            self.storage_live_for_always_live_locals()?;
        };
        res.inspect_err_kind(|_| {
            // Don't show the incomplete stack frame in the error stacktrace.
            self.stack_mut().pop();
        })
    }

    /// Initiate a call to this function -- pushing the stack frame and initializing the arguments.
    ///
    /// `caller_fn_abi` is used to determine if all the arguments are passed the proper way.
    /// However, we also need `caller_abi` to determine if we need to do untupling of arguments.
    ///
    /// `with_caller_location` indicates whether the caller passed a caller location. Miri
    /// implements caller locations without argument passing, but to match `FnAbi` we need to know
    /// when those arguments are present.
    pub(super) fn init_fn_call(
        &mut self,
        fn_val: FnVal<'tcx, M::ExtraFnVal>,
        (caller_abi, caller_fn_abi): (ExternAbi, &FnAbi<'tcx, Ty<'tcx>>),
        args: &[FnArg<'tcx, M::Provenance>],
        with_caller_location: bool,
        destination: &PlaceTy<'tcx, M::Provenance>,
        target: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        trace!("init_fn_call: {:#?}", fn_val);

        let instance = match fn_val {
            FnVal::Instance(instance) => instance,
            FnVal::Other(extra) => {
                return M::call_extra_fn(
                    self,
                    extra,
                    caller_fn_abi,
                    args,
                    destination,
                    target,
                    unwind,
                );
            }
        };

        match instance.def {
            ty::InstanceKind::Intrinsic(def_id) => {
                assert!(self.tcx.intrinsic(def_id).is_some());
                // FIXME: Should `InPlace` arguments be reset to uninit?
                if let Some(fallback) = M::call_intrinsic(
                    self,
                    instance,
                    &self.copy_fn_args(args),
                    destination,
                    target,
                    unwind,
                )? {
                    assert!(!self.tcx.intrinsic(fallback.def_id()).unwrap().must_be_overridden);
                    assert_matches!(fallback.def, ty::InstanceKind::Item(_));
                    return self.init_fn_call(
                        FnVal::Instance(fallback),
                        (caller_abi, caller_fn_abi),
                        args,
                        with_caller_location,
                        destination,
                        target,
                        unwind,
                    );
                } else {
                    interp_ok(())
                }
            }
            ty::InstanceKind::VTableShim(..)
            | ty::InstanceKind::ReifyShim(..)
            | ty::InstanceKind::ClosureOnceShim { .. }
            | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
            | ty::InstanceKind::FnPtrShim(..)
            | ty::InstanceKind::DropGlue(..)
            | ty::InstanceKind::CloneShim(..)
            | ty::InstanceKind::FnPtrAddrShim(..)
            | ty::InstanceKind::ThreadLocalShim(..)
            | ty::InstanceKind::AsyncDropGlueCtorShim(..)
            | ty::InstanceKind::AsyncDropGlue(..)
            | ty::InstanceKind::FutureDropPollShim(..)
            | ty::InstanceKind::Item(_) => {
                // We need MIR for this fn.
                // Note that this can be an intrinsic, if we are executing its fallback body.
                let Some((body, instance)) = M::find_mir_or_eval_fn(
                    self,
                    instance,
                    caller_fn_abi,
                    args,
                    destination,
                    target,
                    unwind,
                )?
                else {
                    return interp_ok(());
                };

                // Special handling for the closure ABI: untuple the last argument.
                let args: Cow<'_, [FnArg<'tcx, M::Provenance>]> =
                    if caller_abi == ExternAbi::RustCall && !args.is_empty() {
                        // Untuple
                        let (untuple_arg, args) = args.split_last().unwrap();
                        trace!("init_fn_call: Will pass last argument by untupling");
                        Cow::from(
                            args.iter()
                                .map(|a| interp_ok(a.clone()))
                                .chain((0..untuple_arg.layout().fields.count()).map(|i| {
                                    self.fn_arg_field(untuple_arg, FieldIdx::from_usize(i))
                                }))
                                .collect::<InterpResult<'_, Vec<_>>>()?,
                        )
                    } else {
                        // Plain arg passing
                        Cow::from(args)
                    };

                self.init_stack_frame(
                    instance,
                    body,
                    caller_fn_abi,
                    &args,
                    with_caller_location,
                    destination,
                    StackPopCleanup::Goto { ret: target, unwind },
                )
            }
            // `InstanceKind::Virtual` does not have callable MIR. Calls to `Virtual` instances must be
            // codegen'd / interpreted as virtual calls through the vtable.
            ty::InstanceKind::Virtual(def_id, idx) => {
                let mut args = args.to_vec();
                // We have to implement all "dyn-compatible receivers". So we have to go search for a
                // pointer or `dyn Trait` type, but it could be wrapped in newtypes. So recursively
                // unwrap those newtypes until we are there.
                // An `InPlace` does nothing here, we keep the original receiver intact. We can't
                // really pass the argument in-place anyway, and we are constructing a new
                // `Immediate` receiver.
                let mut receiver = self.copy_fn_arg(&args[0]);
                let receiver_place = loop {
                    match receiver.layout.ty.kind() {
                        ty::Ref(..) | ty::RawPtr(..) => {
                            // We do *not* use `deref_pointer` here: we don't want to conceptually
                            // create a place that must be dereferenceable, since the receiver might
                            // be a raw pointer and (for `*const dyn Trait`) we don't need to
                            // actually access memory to resolve this method.
                            // Also see <https://github.com/rust-lang/miri/issues/2786>.
                            let val = self.read_immediate(&receiver)?;
                            break self.ref_to_mplace(&val)?;
                        }
                        ty::Dynamic(.., ty::Dyn) => break receiver.assert_mem_place(), // no immediate unsized values
                        ty::Dynamic(.., ty::DynStar) => {
                            // Not clear how to handle this, so far we assume the receiver is always a pointer.
                            span_bug!(
                                self.cur_span(),
                                "by-value calls on a `dyn*`... are those a thing?"
                            );
                        }
                        _ => {
                            // Not there yet, search for the only non-ZST field.
                            // (The rules for `DispatchFromDyn` ensure there's exactly one such field.)
                            let (idx, _) = receiver.layout.non_1zst_field(self).expect(
                                "not exactly one non-1-ZST field in a `DispatchFromDyn` type",
                            );
                            receiver = self.project_field(&receiver, idx)?;
                        }
                    }
                };

                // Obtain the underlying trait we are working on, and the adjusted receiver argument.
                let (trait_, dyn_ty, adjusted_recv) = if let ty::Dynamic(data, _, ty::DynStar) =
                    receiver_place.layout.ty.kind()
                {
                    let recv = self.unpack_dyn_star(&receiver_place, data)?;

                    (data.principal(), recv.layout.ty, recv.ptr())
                } else {
                    // Doesn't have to be a `dyn Trait`, but the unsized tail must be `dyn Trait`.
                    // (For that reason we also cannot use `unpack_dyn_trait`.)
                    let receiver_tail =
                        self.tcx.struct_tail_for_codegen(receiver_place.layout.ty, self.typing_env);
                    let ty::Dynamic(receiver_trait, _, ty::Dyn) = receiver_tail.kind() else {
                        span_bug!(
                            self.cur_span(),
                            "dynamic call on non-`dyn` type {}",
                            receiver_tail
                        )
                    };
                    assert!(receiver_place.layout.is_unsized());

                    // Get the required information from the vtable.
                    let vptr = receiver_place.meta().unwrap_meta().to_pointer(self)?;
                    let dyn_ty = self.get_ptr_vtable_ty(vptr, Some(receiver_trait))?;

                    // It might be surprising that we use a pointer as the receiver even if this
                    // is a by-val case; this works because by-val passing of an unsized `dyn
                    // Trait` to a function is actually desugared to a pointer.
                    (receiver_trait.principal(), dyn_ty, receiver_place.ptr())
                };

                // Now determine the actual method to call. Usually we use the easy way of just
                // looking up the method at index `idx`.
                let vtable_entries = self.vtable_entries(trait_, dyn_ty);
                let Some(ty::VtblEntry::Method(fn_inst)) = vtable_entries.get(idx).copied() else {
                    // FIXME(fee1-dead) these could be variants of the UB info enum instead of this
                    throw_ub_custom!(fluent::const_eval_dyn_call_not_a_method);
                };
                trace!("Virtual call dispatches to {fn_inst:#?}");
                // We can also do the lookup based on `def_id` and `dyn_ty`, and check that that
                // produces the same result.
                self.assert_virtual_instance_matches_concrete(dyn_ty, def_id, instance, fn_inst);

                // Adjust receiver argument. Layout can be any (thin) ptr.
                let receiver_ty = Ty::new_mut_ptr(self.tcx.tcx, dyn_ty);
                args[0] = FnArg::Copy(
                    ImmTy::from_immediate(
                        Scalar::from_maybe_pointer(adjusted_recv, self).into(),
                        self.layout_of(receiver_ty)?,
                    )
                    .into(),
                );
                trace!("Patched receiver operand to {:#?}", args[0]);
                // Need to also adjust the type in the ABI. Strangely, the layout there is actually
                // already fine! Just the type is bogus. This is due to what `force_thin_self_ptr`
                // does in `fn_abi_new_uncached`; supposedly, codegen relies on having the bogus
                // type, so we just patch this up locally.
                let mut caller_fn_abi = caller_fn_abi.clone();
                caller_fn_abi.args[0].layout.ty = receiver_ty;

                // recurse with concrete function
                self.init_fn_call(
                    FnVal::Instance(fn_inst),
                    (caller_abi, &caller_fn_abi),
                    &args,
                    with_caller_location,
                    destination,
                    target,
                    unwind,
                )
            }
        }
    }

    fn assert_virtual_instance_matches_concrete(
        &self,
        dyn_ty: Ty<'tcx>,
        def_id: DefId,
        virtual_instance: ty::Instance<'tcx>,
        concrete_instance: ty::Instance<'tcx>,
    ) {
        let tcx = *self.tcx;

        let trait_def_id = tcx.trait_of_item(def_id).unwrap();
        let virtual_trait_ref = ty::TraitRef::from_method(tcx, trait_def_id, virtual_instance.args);
        let existential_trait_ref = ty::ExistentialTraitRef::erase_self_ty(tcx, virtual_trait_ref);
        let concrete_trait_ref = existential_trait_ref.with_self_ty(tcx, dyn_ty);

        let concrete_method = Instance::expect_resolve_for_vtable(
            tcx,
            self.typing_env,
            def_id,
            virtual_instance.args.rebase_onto(tcx, trait_def_id, concrete_trait_ref.args),
            self.cur_span(),
        );
        assert_eq!(concrete_instance, concrete_method);
    }

    /// Initiate a tail call to this function -- popping the current stack frame, pushing the new
    /// stack frame and initializing the arguments.
    pub(super) fn init_fn_tail_call(
        &mut self,
        fn_val: FnVal<'tcx, M::ExtraFnVal>,
        (caller_abi, caller_fn_abi): (ExternAbi, &FnAbi<'tcx, Ty<'tcx>>),
        args: &[FnArg<'tcx, M::Provenance>],
        with_caller_location: bool,
    ) -> InterpResult<'tcx> {
        trace!("init_fn_tail_call: {:#?}", fn_val);

        // This is the "canonical" implementation of tails calls,
        // a pop of the current stack frame, followed by a normal call
        // which pushes a new stack frame, with the return address from
        // the popped stack frame.
        //
        // Note that we are using `pop_stack_frame_raw` and not `return_from_current_stack_frame`,
        // as the latter "executes" the goto to the return block, but we don't want to,
        // only the tail called function should return to the current return block.
        let StackPopInfo { return_action, return_to_block, return_place } = self
            .pop_stack_frame_raw(false, |_this, _return_place| {
                // This function's return value is just discarded, the tail-callee will fill in the return place instead.
                interp_ok(())
            })?;

        assert_eq!(return_action, ReturnAction::Normal);

        // Take the "stack pop cleanup" info, and use that to initiate the next call.
        let StackPopCleanup::Goto { ret, unwind } = return_to_block else {
            bug!("can't tailcall as root");
        };

        // FIXME(explicit_tail_calls):
        //   we should check if both caller&callee can/n't unwind,
        //   see <https://github.com/rust-lang/rust/pull/113128#issuecomment-1614979803>

        self.init_fn_call(
            fn_val,
            (caller_abi, caller_fn_abi),
            args,
            with_caller_location,
            &return_place,
            ret,
            unwind,
        )
    }

    pub(super) fn init_drop_in_place_call(
        &mut self,
        place: &PlaceTy<'tcx, M::Provenance>,
        instance: ty::Instance<'tcx>,
        target: mir::BasicBlock,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        trace!("init_drop_in_place_call: {:?},\n  instance={:?}", place, instance);
        // We take the address of the object. This may well be unaligned, which is fine
        // for us here. However, unaligned accesses will probably make the actual drop
        // implementation fail -- a problem shared by rustc.
        let place = self.force_allocation(place)?;

        // We behave a bit different from codegen here.
        // Codegen creates an `InstanceKind::Virtual` with index 0 (the slot of the drop method) and
        // then dispatches that to the normal call machinery. However, our call machinery currently
        // only supports calling `VtblEntry::Method`; it would choke on a `MetadataDropInPlace`. So
        // instead we do the virtual call stuff ourselves. It's easier here than in `eval_fn_call`
        // since we can just get a place of the underlying type and use `mplace_to_ref`.
        let place = match place.layout.ty.kind() {
            ty::Dynamic(data, _, ty::Dyn) => {
                // Dropping a trait object. Need to find actual drop fn.
                self.unpack_dyn_trait(&place, data)?
            }
            ty::Dynamic(data, _, ty::DynStar) => {
                // Dropping a `dyn*`. Need to find actual drop fn.
                self.unpack_dyn_star(&place, data)?
            }
            _ => {
                debug_assert_eq!(
                    instance,
                    ty::Instance::resolve_drop_in_place(*self.tcx, place.layout.ty)
                );
                place
            }
        };
        let instance = ty::Instance::resolve_drop_in_place(*self.tcx, place.layout.ty);
        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty())?;

        let arg = self.mplace_to_ref(&place)?;
        let ret = MPlaceTy::fake_alloc_zst(self.layout_of(self.tcx.types.unit)?);

        self.init_fn_call(
            FnVal::Instance(instance),
            (ExternAbi::Rust, fn_abi),
            &[FnArg::Copy(arg.into())],
            false,
            &ret.into(),
            Some(target),
            unwind,
        )
    }

    /// Pops the current frame from the stack, copies the return value to the caller, deallocates
    /// the memory for allocated locals, and jumps to an appropriate place.
    ///
    /// If `unwinding` is `false`, then we are performing a normal return
    /// from a function. In this case, we jump back into the frame of the caller,
    /// and continue execution as normal.
    ///
    /// If `unwinding` is `true`, then we are in the middle of a panic,
    /// and need to unwind this frame. In this case, we jump to the
    /// `cleanup` block for the function, which is responsible for running
    /// `Drop` impls for any locals that have been initialized at this point.
    /// The cleanup block ends with a special `Resume` terminator, which will
    /// cause us to continue unwinding.
    #[instrument(skip(self), level = "trace")]
    pub(super) fn return_from_current_stack_frame(
        &mut self,
        unwinding: bool,
    ) -> InterpResult<'tcx> {
        info!(
            "popping stack frame ({})",
            if unwinding { "during unwinding" } else { "returning from function" }
        );

        // Check `unwinding`.
        assert_eq!(
            unwinding,
            match self.frame().loc {
                Left(loc) => self.body().basic_blocks[loc.block].is_cleanup,
                Right(_) => true,
            }
        );
        if unwinding && self.frame_idx() == 0 {
            throw_ub_custom!(fluent::const_eval_unwind_past_top);
        }

        // Get out the return value. Must happen *before* the frame is popped as we have to get the
        // local's value out.
        let return_op =
            self.local_to_op(mir::RETURN_PLACE, None).expect("return place should always be live");
        // Do the actual pop + copy.
        let stack_pop_info = self.pop_stack_frame_raw(unwinding, |this, return_place| {
            this.copy_op_allow_transmute(&return_op, return_place)?;
            trace!("return value: {:?}", this.dump_place(return_place));
            interp_ok(())
        })?;

        match stack_pop_info.return_action {
            ReturnAction::Normal => {}
            ReturnAction::NoJump => {
                // The hook already did everything.
                return interp_ok(());
            }
            ReturnAction::NoCleanup => {
                // If we are not doing cleanup, also skip everything else.
                assert!(self.stack().is_empty(), "only the topmost frame should ever be leaked");
                assert!(!unwinding, "tried to skip cleanup during unwinding");
                // Don't jump anywhere.
                return interp_ok(());
            }
        }

        // Normal return, figure out where to jump.
        if unwinding {
            // Follow the unwind edge.
            match stack_pop_info.return_to_block {
                StackPopCleanup::Goto { unwind, .. } => {
                    // This must be the very last thing that happens, since it can in fact push a new stack frame.
                    self.unwind_to_block(unwind)
                }
                StackPopCleanup::Root { .. } => {
                    panic!("encountered StackPopCleanup::Root when unwinding!")
                }
            }
        } else {
            // Follow the normal return edge.
            match stack_pop_info.return_to_block {
                StackPopCleanup::Goto { ret, .. } => self.return_to_block(ret),
                StackPopCleanup::Root { .. } => {
                    assert!(
                        self.stack().is_empty(),
                        "only the bottommost frame can have StackPopCleanup::Root"
                    );
                    interp_ok(())
                }
            }
        }
    }
}
