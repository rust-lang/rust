use std::borrow::Cow;

use either::Either;
use rustc_ast::ast::InlineAsmOptions;
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf, TyAndLayout};
use rustc_middle::ty::Instance;
use rustc_middle::{
    mir,
    ty::{self, Ty},
};
use rustc_target::abi;
use rustc_target::abi::call::{ArgAbi, ArgAttribute, ArgAttributes, FnAbi, PassMode};
use rustc_target::spec::abi::Abi;

use super::{
    AllocId, FnVal, ImmTy, Immediate, InterpCx, InterpResult, MPlaceTy, Machine, MemoryKind, OpTy,
    Operand, PlaceTy, Provenance, Scalar, StackPopCleanup,
};
use crate::fluent_generated as fluent;

/// An argment passed to a function.
#[derive(Clone, Debug)]
pub enum FnArg<'tcx, Prov: Provenance = AllocId> {
    /// Pass a copy of the given operand.
    Copy(OpTy<'tcx, Prov>),
    /// Allow for the argument to be passed in-place: destroy the value originally stored at that place and
    /// make the place inaccessible for the duration of the function call.
    InPlace(PlaceTy<'tcx, Prov>),
}

impl<'tcx, Prov: Provenance> FnArg<'tcx, Prov> {
    pub fn layout(&self) -> &TyAndLayout<'tcx> {
        match self {
            FnArg::Copy(op) => &op.layout,
            FnArg::InPlace(place) => &place.layout,
        }
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Make a copy of the given fn_arg. Any `InPlace` are degenerated to copies, no protection of the
    /// original memory occurs.
    pub fn copy_fn_arg(
        &self,
        arg: &FnArg<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        match arg {
            FnArg::Copy(op) => Ok(op.clone()),
            FnArg::InPlace(place) => self.place_to_op(&place),
        }
    }

    /// Make a copy of the given fn_args. Any `InPlace` are degenerated to copies, no protection of the
    /// original memory occurs.
    pub fn copy_fn_args(
        &self,
        args: &[FnArg<'tcx, M::Provenance>],
    ) -> InterpResult<'tcx, Vec<OpTy<'tcx, M::Provenance>>> {
        args.iter().map(|fn_arg| self.copy_fn_arg(fn_arg)).collect()
    }

    pub fn fn_arg_field(
        &mut self,
        arg: &FnArg<'tcx, M::Provenance>,
        field: usize,
    ) -> InterpResult<'tcx, FnArg<'tcx, M::Provenance>> {
        Ok(match arg {
            FnArg::Copy(op) => FnArg::Copy(self.operand_field(op, field)?),
            FnArg::InPlace(place) => FnArg::InPlace(self.place_field(place, field)?),
        })
    }

    pub(super) fn eval_terminator(
        &mut self,
        terminator: &mir::Terminator<'tcx>,
    ) -> InterpResult<'tcx> {
        use rustc_middle::mir::TerminatorKind::*;
        match terminator.kind {
            Return => {
                self.pop_stack_frame(/* unwinding */ false)?
            }

            Goto { target } => self.go_to_block(target),

            SwitchInt { ref discr, ref targets } => {
                let discr = self.read_immediate(&self.eval_operand(discr, None)?)?;
                trace!("SwitchInt({:?})", *discr);

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets.otherwise();

                for (const_int, target) in targets.iter() {
                    // Compare using MIR BinOp::Eq, to also support pointer values.
                    // (Avoiding `self.binary_op` as that does some redundant layout computation.)
                    let res = self
                        .overflowing_binary_op(
                            mir::BinOp::Eq,
                            &discr,
                            &ImmTy::from_uint(const_int, discr.layout),
                        )?
                        .0;
                    if res.to_bool()? {
                        target_block = target;
                        break;
                    }
                }

                self.go_to_block(target_block);
            }

            Call {
                ref func,
                ref args,
                destination,
                target,
                unwind,
                call_source: _,
                fn_span: _,
            } => {
                let old_stack = self.frame_idx();
                let old_loc = self.frame().loc;
                let func = self.eval_operand(func, None)?;
                let args = self.eval_fn_call_arguments(args)?;

                let fn_sig_binder = func.layout.ty.fn_sig(*self.tcx);
                let fn_sig =
                    self.tcx.normalize_erasing_late_bound_regions(self.param_env, fn_sig_binder);
                let extra_args = &args[fn_sig.inputs().len()..];
                let extra_args =
                    self.tcx.mk_type_list_from_iter(extra_args.iter().map(|arg| arg.layout().ty));

                let (fn_val, fn_abi, with_caller_location) = match *func.layout.ty.kind() {
                    ty::FnPtr(_sig) => {
                        let fn_ptr = self.read_pointer(&func)?;
                        let fn_val = self.get_ptr_fn(fn_ptr)?;
                        (fn_val, self.fn_abi_of_fn_ptr(fn_sig_binder, extra_args)?, false)
                    }
                    ty::FnDef(def_id, args) => {
                        let instance = self.resolve(def_id, args)?;
                        (
                            FnVal::Instance(instance),
                            self.fn_abi_of_instance(instance, extra_args)?,
                            instance.def.requires_caller_location(*self.tcx),
                        )
                    }
                    _ => span_bug!(
                        terminator.source_info.span,
                        "invalid callee of type {:?}",
                        func.layout.ty
                    ),
                };

                let destination = self.eval_place(destination)?;
                self.eval_fn_call(
                    fn_val,
                    (fn_sig.abi, fn_abi),
                    &args,
                    with_caller_location,
                    &destination,
                    target,
                    if fn_abi.can_unwind { unwind } else { mir::UnwindAction::Unreachable },
                )?;
                // Sanity-check that `eval_fn_call` either pushed a new frame or
                // did a jump to another block.
                if self.frame_idx() == old_stack && self.frame().loc == old_loc {
                    span_bug!(terminator.source_info.span, "evaluating this call made no progress");
                }
            }

            Drop { place, target, unwind, replace: _ } => {
                let frame = self.frame();
                let ty = place.ty(&frame.body.local_decls, *self.tcx).ty;
                let ty = self.subst_from_frame_and_normalize_erasing_regions(frame, ty)?;
                let instance = Instance::resolve_drop_in_place(*self.tcx, ty);
                if let ty::InstanceDef::DropGlue(_, None) = instance.def {
                    // This is the branch we enter if and only if the dropped type has no drop glue
                    // whatsoever. This can happen as a result of monomorphizing a drop of a
                    // generic. In order to make sure that generic and non-generic code behaves
                    // roughly the same (and in keeping with Mir semantics) we do nothing here.
                    self.go_to_block(target);
                    return Ok(());
                }
                let place = self.eval_place(place)?;
                trace!("TerminatorKind::drop: {:?}, type {}", place, ty);
                self.drop_in_place(&place, instance, target, unwind)?;
            }

            Assert { ref cond, expected, ref msg, target, unwind } => {
                let ignored =
                    M::ignore_optional_overflow_checks(self) && msg.is_optional_overflow_check();
                let cond_val = self.read_scalar(&self.eval_operand(cond, None)?)?.to_bool()?;
                if ignored || expected == cond_val {
                    self.go_to_block(target);
                } else {
                    M::assert_panic(self, msg, unwind)?;
                }
            }

            Terminate => {
                // FIXME: maybe should call `panic_no_unwind` lang item instead.
                M::abort(self, "panic in a function that cannot unwind".to_owned())?;
            }

            // When we encounter Resume, we've finished unwinding
            // cleanup for the current stack frame. We pop it in order
            // to continue unwinding the next frame
            Resume => {
                trace!("unwinding: resuming from cleanup");
                // By definition, a Resume terminator means
                // that we're unwinding
                self.pop_stack_frame(/* unwinding */ true)?;
                return Ok(());
            }

            // It is UB to ever encounter this.
            Unreachable => throw_ub!(Unreachable),

            // These should never occur for MIR we actually run.
            FalseEdge { .. } | FalseUnwind { .. } | Yield { .. } | GeneratorDrop => span_bug!(
                terminator.source_info.span,
                "{:#?} should have been eliminated by MIR pass",
                terminator.kind
            ),

            InlineAsm { template, ref operands, options, destination, .. } => {
                M::eval_inline_asm(self, template, operands, options)?;
                if options.contains(InlineAsmOptions::NORETURN) {
                    throw_ub_custom!(fluent::const_eval_noreturn_asm_returned);
                }
                self.go_to_block(
                    destination
                        .expect("InlineAsm terminators without noreturn must have a destination"),
                )
            }
        }

        Ok(())
    }

    /// Evaluate the arguments of a function call
    pub(super) fn eval_fn_call_arguments(
        &mut self,
        ops: &[mir::Operand<'tcx>],
    ) -> InterpResult<'tcx, Vec<FnArg<'tcx, M::Provenance>>> {
        ops.iter()
            .map(|op| {
                Ok(match op {
                    mir::Operand::Move(place) => FnArg::InPlace(self.eval_place(*place)?),
                    _ => FnArg::Copy(self.eval_operand(op, None)?),
                })
            })
            .collect()
    }

    fn check_argument_compat(
        caller_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        callee_abi: &ArgAbi<'tcx, Ty<'tcx>>,
    ) -> bool {
        // Heuristic for type comparison.
        let layout_compat = || {
            if caller_abi.layout.ty == callee_abi.layout.ty {
                // No question
                return true;
            }
            if caller_abi.layout.is_unsized() || callee_abi.layout.is_unsized() {
                // No, no, no. We require the types to *exactly* match for unsized arguments. If
                // these are somehow unsized "in a different way" (say, `dyn Trait` vs `[i32]`),
                // then who knows what happens.
                return false;
            }
            if caller_abi.layout.size != callee_abi.layout.size
                || caller_abi.layout.align.abi != callee_abi.layout.align.abi
            {
                // This cannot go well...
                return false;
            }
            // The rest *should* be okay, but we are extra conservative.
            match (caller_abi.layout.abi, callee_abi.layout.abi) {
                // Different valid ranges are okay (once we enforce validity,
                // that will take care to make it UB to leave the range, just
                // like for transmute).
                (abi::Abi::Scalar(caller), abi::Abi::Scalar(callee)) => {
                    caller.primitive() == callee.primitive()
                }
                (
                    abi::Abi::ScalarPair(caller1, caller2),
                    abi::Abi::ScalarPair(callee1, callee2),
                ) => {
                    caller1.primitive() == callee1.primitive()
                        && caller2.primitive() == callee2.primitive()
                }
                // Be conservative
                _ => false,
            }
        };
        // When comparing the PassMode, we have to be smart about comparing the attributes.
        let arg_attr_compat = |a1: &ArgAttributes, a2: &ArgAttributes| {
            // There's only one regular attribute that matters for the call ABI: InReg.
            // Everything else is things like noalias, dereferenceable, nonnull, ...
            // (This also applies to pointee_size, pointee_align.)
            if a1.regular.contains(ArgAttribute::InReg) != a2.regular.contains(ArgAttribute::InReg)
            {
                return false;
            }
            // We also compare the sign extension mode -- this could let the callee make assumptions
            // about bits that conceptually were not even passed.
            if a1.arg_ext != a2.arg_ext {
                return false;
            }
            return true;
        };
        let mode_compat = || match (&caller_abi.mode, &callee_abi.mode) {
            (PassMode::Ignore, PassMode::Ignore) => true,
            (PassMode::Direct(a1), PassMode::Direct(a2)) => arg_attr_compat(a1, a2),
            (PassMode::Pair(a1, b1), PassMode::Pair(a2, b2)) => {
                arg_attr_compat(a1, a2) && arg_attr_compat(b1, b2)
            }
            (PassMode::Cast(c1, pad1), PassMode::Cast(c2, pad2)) => c1 == c2 && pad1 == pad2,
            (
                PassMode::Indirect { attrs: a1, extra_attrs: None, on_stack: s1 },
                PassMode::Indirect { attrs: a2, extra_attrs: None, on_stack: s2 },
            ) => arg_attr_compat(a1, a2) && s1 == s2,
            (
                PassMode::Indirect { attrs: a1, extra_attrs: Some(e1), on_stack: s1 },
                PassMode::Indirect { attrs: a2, extra_attrs: Some(e2), on_stack: s2 },
            ) => arg_attr_compat(a1, a2) && arg_attr_compat(e1, e2) && s1 == s2,
            _ => false,
        };

        if layout_compat() && mode_compat() {
            return true;
        }
        trace!(
            "check_argument_compat: incompatible ABIs:\ncaller: {:?}\ncallee: {:?}",
            caller_abi,
            callee_abi
        );
        return false;
    }

    /// Initialize a single callee argument, checking the types for compatibility.
    fn pass_argument<'x, 'y>(
        &mut self,
        caller_args: &mut impl Iterator<
            Item = (&'x FnArg<'tcx, M::Provenance>, &'y ArgAbi<'tcx, Ty<'tcx>>),
        >,
        callee_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        callee_arg: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx>
    where
        'tcx: 'x,
        'tcx: 'y,
    {
        if matches!(callee_abi.mode, PassMode::Ignore) {
            // This one is skipped.
            return Ok(());
        }
        // Find next caller arg.
        let Some((caller_arg, caller_abi)) = caller_args.next() else {
            throw_ub_custom!(fluent::const_eval_not_enough_caller_args);
        };
        // Now, check
        if !Self::check_argument_compat(caller_abi, callee_abi) {
            let callee_ty = format!("{}", callee_arg.layout.ty);
            let caller_ty = format!("{}", caller_arg.layout().ty);
            throw_ub_custom!(
                fluent::const_eval_incompatible_types,
                callee_ty = callee_ty,
                caller_ty = caller_ty,
            )
        }
        // We work with a copy of the argument for now; if this is in-place argument passing, we
        // will later protect the source it comes from. This means the callee cannot observe if we
        // did in-place of by-copy argument passing, except for pointer equality tests.
        let caller_arg_copy = self.copy_fn_arg(&caller_arg)?;
        // Special handling for unsized parameters.
        if caller_arg_copy.layout.is_unsized() {
            // `check_argument_compat` ensures that both have the same type, so we know they will use the metadata the same way.
            assert_eq!(caller_arg_copy.layout.ty, callee_arg.layout.ty);
            // We have to properly pre-allocate the memory for the callee.
            // So let's tear down some abstractions.
            // This all has to be in memory, there are no immediate unsized values.
            let src = caller_arg_copy.assert_mem_place();
            // The destination cannot be one of these "spread args".
            let (dest_frame, dest_local) = callee_arg.assert_local();
            // We are just initializing things, so there can't be anything here yet.
            assert!(matches!(
                *self.local_to_op(&self.stack()[dest_frame], dest_local, None)?,
                Operand::Immediate(Immediate::Uninit)
            ));
            // Allocate enough memory to hold `src`.
            let Some((size, align)) = self.size_and_align_of_mplace(&src)? else {
                span_bug!(
                    self.cur_span(),
                    "unsized fn arg with `extern` type tail should not be allowed"
                )
            };
            let ptr = self.allocate_ptr(size, align, MemoryKind::Stack)?;
            let dest_place =
                MPlaceTy::from_aligned_ptr_with_meta(ptr.into(), callee_arg.layout, src.meta);
            // Update the local to be that new place.
            *M::access_local_mut(self, dest_frame, dest_local)? = Operand::Indirect(*dest_place);
        }
        // We allow some transmutes here.
        // FIXME: Depending on the PassMode, this should reset some padding to uninitialized. (This
        // is true for all `copy_op`, but there are a lot of special cases for argument passing
        // specifically.)
        self.copy_op(&caller_arg_copy, callee_arg, /*allow_transmute*/ true)?;
        // If this was an in-place pass, protect the place it comes from for the duration of the call.
        if let FnArg::InPlace(place) = caller_arg {
            M::protect_in_place_function_argument(self, place)?;
        }
        Ok(())
    }

    /// Call this function -- pushing the stack frame and initializing the arguments.
    ///
    /// `caller_fn_abi` is used to determine if all the arguments are passed the proper way.
    /// However, we also need `caller_abi` to determine if we need to do untupling of arguments.
    ///
    /// `with_caller_location` indicates whether the caller passed a caller location. Miri
    /// implements caller locations without argument passing, but to match `FnAbi` we need to know
    /// when those arguments are present.
    pub(crate) fn eval_fn_call(
        &mut self,
        fn_val: FnVal<'tcx, M::ExtraFnVal>,
        (caller_abi, caller_fn_abi): (Abi, &FnAbi<'tcx, Ty<'tcx>>),
        args: &[FnArg<'tcx, M::Provenance>],
        with_caller_location: bool,
        destination: &PlaceTy<'tcx, M::Provenance>,
        target: Option<mir::BasicBlock>,
        mut unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        trace!("eval_fn_call: {:#?}", fn_val);

        let instance = match fn_val {
            FnVal::Instance(instance) => instance,
            FnVal::Other(extra) => {
                return M::call_extra_fn(
                    self,
                    extra,
                    caller_abi,
                    args,
                    destination,
                    target,
                    unwind,
                );
            }
        };

        match instance.def {
            ty::InstanceDef::Intrinsic(def_id) => {
                assert!(self.tcx.is_intrinsic(def_id));
                // FIXME: Should `InPlace` arguments be reset to uninit?
                M::call_intrinsic(
                    self,
                    instance,
                    &self.copy_fn_args(args)?,
                    destination,
                    target,
                    unwind,
                )
            }
            ty::InstanceDef::VTableShim(..)
            | ty::InstanceDef::ReifyShim(..)
            | ty::InstanceDef::ClosureOnceShim { .. }
            | ty::InstanceDef::FnPtrShim(..)
            | ty::InstanceDef::DropGlue(..)
            | ty::InstanceDef::CloneShim(..)
            | ty::InstanceDef::FnPtrAddrShim(..)
            | ty::InstanceDef::ThreadLocalShim(..)
            | ty::InstanceDef::Item(_) => {
                // We need MIR for this fn
                let Some((body, instance)) = M::find_mir_or_eval_fn(
                    self,
                    instance,
                    caller_abi,
                    args,
                    destination,
                    target,
                    unwind,
                )?
                else {
                    return Ok(());
                };

                // Compute callee information using the `instance` returned by
                // `find_mir_or_eval_fn`.
                // FIXME: for variadic support, do we have to somehow determine callee's extra_args?
                let callee_fn_abi = self.fn_abi_of_instance(instance, ty::List::empty())?;

                if callee_fn_abi.c_variadic || caller_fn_abi.c_variadic {
                    throw_unsup_format!("calling a c-variadic function is not supported");
                }

                if M::enforce_abi(self) {
                    if caller_fn_abi.conv != callee_fn_abi.conv {
                        throw_ub_custom!(
                            fluent::const_eval_incompatible_calling_conventions,
                            callee_conv = format!("{:?}", callee_fn_abi.conv),
                            caller_conv = format!("{:?}", caller_fn_abi.conv),
                        )
                    }
                }

                // Check that all target features required by the callee (i.e., from
                // the attribute `#[target_feature(enable = ...)]`) are enabled at
                // compile time.
                self.check_fn_target_features(instance)?;

                if !callee_fn_abi.can_unwind {
                    // The callee cannot unwind, so force the `Unreachable` unwind handling.
                    unwind = mir::UnwindAction::Unreachable;
                }

                self.push_stack_frame(
                    instance,
                    body,
                    destination,
                    StackPopCleanup::Goto { ret: target, unwind },
                )?;

                // If an error is raised here, pop the frame again to get an accurate backtrace.
                // To this end, we wrap it all in a `try` block.
                let res: InterpResult<'tcx> = try {
                    trace!(
                        "caller ABI: {:?}, args: {:#?}",
                        caller_abi,
                        args.iter()
                            .map(|arg| (
                                arg.layout().ty,
                                match arg {
                                    FnArg::Copy(op) => format!("copy({:?})", *op),
                                    FnArg::InPlace(place) => format!("in-place({:?})", *place),
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
                                self.layout_of_local(self.frame(), local, None).unwrap().ty
                            ))
                            .collect::<Vec<_>>()
                    );

                    // In principle, we have two iterators: Where the arguments come from, and where
                    // they go to.

                    // For where they come from: If the ABI is RustCall, we untuple the
                    // last incoming argument. These two iterators do not have the same type,
                    // so to keep the code paths uniform we accept an allocation
                    // (for RustCall ABI only).
                    let caller_args: Cow<'_, [FnArg<'tcx, M::Provenance>]> =
                        if caller_abi == Abi::RustCall && !args.is_empty() {
                            // Untuple
                            let (untuple_arg, args) = args.split_last().unwrap();
                            trace!("eval_fn_call: Will pass last argument by untupling");
                            Cow::from(
                                args.iter()
                                    .map(|a| Ok(a.clone()))
                                    .chain(
                                        (0..untuple_arg.layout().fields.count())
                                            .map(|i| self.fn_arg_field(untuple_arg, i)),
                                    )
                                    .collect::<InterpResult<'_, Vec<_>>>()?,
                            )
                        } else {
                            // Plain arg passing
                            Cow::from(args)
                        };
                    // If `with_caller_location` is set we pretend there is an extra argument (that
                    // we will not pass).
                    assert_eq!(
                        caller_args.len() + if with_caller_location { 1 } else { 0 },
                        caller_fn_abi.args.len(),
                        "mismatch between caller ABI and caller arguments",
                    );
                    let mut caller_args = caller_args
                        .iter()
                        .zip(caller_fn_abi.args.iter())
                        .filter(|arg_and_abi| !matches!(arg_and_abi.1.mode, PassMode::Ignore));

                    // Now we have to spread them out across the callee's locals,
                    // taking into account the `spread_arg`. If we could write
                    // this is a single iterator (that handles `spread_arg`), then
                    // `pass_argument` would be the loop body. It takes care to
                    // not advance `caller_iter` for ZSTs.
                    let mut callee_args_abis = callee_fn_abi.args.iter();
                    for local in body.args_iter() {
                        let dest = self.eval_place(mir::Place::from(local))?;
                        if Some(local) == body.spread_arg {
                            // Must be a tuple
                            for i in 0..dest.layout.fields.count() {
                                let dest = self.place_field(&dest, i)?;
                                let callee_abi = callee_args_abis.next().unwrap();
                                self.pass_argument(&mut caller_args, callee_abi, &dest)?;
                            }
                        } else {
                            // Normal argument
                            let callee_abi = callee_args_abis.next().unwrap();
                            self.pass_argument(&mut caller_args, callee_abi, &dest)?;
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
                    if !Self::check_argument_compat(&caller_fn_abi.ret, &callee_fn_abi.ret) {
                        let callee_ty = format!("{}", callee_fn_abi.ret.layout.ty);
                        let caller_ty = format!("{}", caller_fn_abi.ret.layout.ty);
                        throw_ub_custom!(
                            fluent::const_eval_incompatible_return_types,
                            callee_ty = callee_ty,
                            caller_ty = caller_ty,
                        )
                    }
                    // Ensure the return place is aligned and dereferenceable, and protect it for
                    // in-place return value passing.
                    if let Either::Left(mplace) = destination.as_mplace_or_local() {
                        self.check_mplace(mplace)?;
                    } else {
                        // Nothing to do for locals, they are always properly allocated and aligned.
                    }
                    M::protect_in_place_function_argument(self, destination)?;
                };
                match res {
                    Err(err) => {
                        self.stack_mut().pop();
                        Err(err)
                    }
                    Ok(()) => Ok(()),
                }
            }
            // cannot use the shim here, because that will only result in infinite recursion
            ty::InstanceDef::Virtual(def_id, idx) => {
                let mut args = args.to_vec();
                // We have to implement all "object safe receivers". So we have to go search for a
                // pointer or `dyn Trait` type, but it could be wrapped in newtypes. So recursively
                // unwrap those newtypes until we are there.
                // An `InPlace` does nothing here, we keep the original receiver intact. We can't
                // really pass the argument in-place anyway, and we are constructing a new
                // `Immediate` receiver.
                let mut receiver = self.copy_fn_arg(&args[0])?;
                let receiver_place = loop {
                    match receiver.layout.ty.kind() {
                        ty::Ref(..) | ty::RawPtr(..) => {
                            // We do *not* use `deref_operand` here: we don't want to conceptually
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
                            let mut non_zst_field = None;
                            for i in 0..receiver.layout.fields.count() {
                                let field = self.operand_field(&receiver, i)?;
                                let zst =
                                    field.layout.is_zst() && field.layout.align.abi.bytes() == 1;
                                if !zst {
                                    assert!(
                                        non_zst_field.is_none(),
                                        "multiple non-ZST fields in dyn receiver type {}",
                                        receiver.layout.ty
                                    );
                                    non_zst_field = Some(field);
                                }
                            }
                            receiver = non_zst_field.unwrap_or_else(|| {
                                panic!(
                                    "no non-ZST fields in dyn receiver type {}",
                                    receiver.layout.ty
                                )
                            });
                        }
                    }
                };

                // Obtain the underlying trait we are working on, and the adjusted receiver argument.
                let (vptr, dyn_ty, adjusted_receiver) = if let ty::Dynamic(data, _, ty::DynStar) =
                    receiver_place.layout.ty.kind()
                {
                    let (recv, vptr) = self.unpack_dyn_star(&receiver_place.into())?;
                    let (dyn_ty, dyn_trait) = self.get_ptr_vtable(vptr)?;
                    if dyn_trait != data.principal() {
                        throw_ub_custom!(fluent::const_eval_dyn_star_call_vtable_mismatch);
                    }
                    let recv = recv.assert_mem_place(); // we passed an MPlaceTy to `unpack_dyn_star` so we definitely still have one

                    (vptr, dyn_ty, recv.ptr)
                } else {
                    // Doesn't have to be a `dyn Trait`, but the unsized tail must be `dyn Trait`.
                    // (For that reason we also cannot use `unpack_dyn_trait`.)
                    let receiver_tail = self
                        .tcx
                        .struct_tail_erasing_lifetimes(receiver_place.layout.ty, self.param_env);
                    let ty::Dynamic(data, _, ty::Dyn) = receiver_tail.kind() else {
                        span_bug!(
                            self.cur_span(),
                            "dynamic call on non-`dyn` type {}",
                            receiver_tail
                        )
                    };
                    assert!(receiver_place.layout.is_unsized());

                    // Get the required information from the vtable.
                    let vptr = receiver_place.meta.unwrap_meta().to_pointer(self)?;
                    let (dyn_ty, dyn_trait) = self.get_ptr_vtable(vptr)?;
                    if dyn_trait != data.principal() {
                        throw_ub_custom!(fluent::const_eval_dyn_call_vtable_mismatch);
                    }

                    // It might be surprising that we use a pointer as the receiver even if this
                    // is a by-val case; this works because by-val passing of an unsized `dyn
                    // Trait` to a function is actually desugared to a pointer.
                    (vptr, dyn_ty, receiver_place.ptr)
                };

                // Now determine the actual method to call. We can do that in two different ways and
                // compare them to ensure everything fits.
                let Some(ty::VtblEntry::Method(fn_inst)) =
                    self.get_vtable_entries(vptr)?.get(idx).copied()
                else {
                    // FIXME(fee1-dead) these could be variants of the UB info enum instead of this
                    throw_ub_custom!(fluent::const_eval_dyn_call_not_a_method);
                };
                trace!("Virtual call dispatches to {fn_inst:#?}");
                if cfg!(debug_assertions) {
                    let tcx = *self.tcx;

                    let trait_def_id = tcx.trait_of_item(def_id).unwrap();
                    let virtual_trait_ref =
                        ty::TraitRef::from_method(tcx, trait_def_id, instance.args);
                    let existential_trait_ref =
                        ty::ExistentialTraitRef::erase_self_ty(tcx, virtual_trait_ref);
                    let concrete_trait_ref = existential_trait_ref.with_self_ty(tcx, dyn_ty);

                    let concrete_method = Instance::resolve_for_vtable(
                        tcx,
                        self.param_env,
                        def_id,
                        instance.args.rebase_onto(tcx, trait_def_id, concrete_trait_ref.args),
                    )
                    .unwrap();
                    assert_eq!(fn_inst, concrete_method);
                }

                // Adjust receiver argument. Layout can be any (thin) ptr.
                args[0] = FnArg::Copy(
                    ImmTy::from_immediate(
                        Scalar::from_maybe_pointer(adjusted_receiver, self).into(),
                        self.layout_of(Ty::new_mut_ptr(self.tcx.tcx, dyn_ty))?,
                    )
                    .into(),
                );
                trace!("Patched receiver operand to {:#?}", args[0]);
                // recurse with concrete function
                self.eval_fn_call(
                    FnVal::Instance(fn_inst),
                    (caller_abi, caller_fn_abi),
                    &args,
                    with_caller_location,
                    destination,
                    target,
                    unwind,
                )
            }
        }
    }

    fn check_fn_target_features(&self, instance: ty::Instance<'tcx>) -> InterpResult<'tcx, ()> {
        let attrs = self.tcx.codegen_fn_attrs(instance.def_id());
        if attrs
            .target_features
            .iter()
            .any(|feature| !self.tcx.sess.target_features.contains(feature))
        {
            throw_ub_custom!(
                fluent::const_eval_unavailable_target_features_for_fn,
                unavailable_feats = attrs
                    .target_features
                    .iter()
                    .filter(|&feature| !self.tcx.sess.target_features.contains(feature))
                    .fold(String::new(), |mut s, feature| {
                        if !s.is_empty() {
                            s.push_str(", ");
                        }
                        s.push_str(feature.as_str());
                        s
                    }),
            );
        }
        Ok(())
    }

    fn drop_in_place(
        &mut self,
        place: &PlaceTy<'tcx, M::Provenance>,
        instance: ty::Instance<'tcx>,
        target: mir::BasicBlock,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        trace!("drop_in_place: {:?},\n  {:?}, {:?}", *place, place.layout.ty, instance);
        // We take the address of the object. This may well be unaligned, which is fine
        // for us here. However, unaligned accesses will probably make the actual drop
        // implementation fail -- a problem shared by rustc.
        let place = self.force_allocation(place)?;

        let place = match place.layout.ty.kind() {
            ty::Dynamic(_, _, ty::Dyn) => {
                // Dropping a trait object. Need to find actual drop fn.
                self.unpack_dyn_trait(&place)?.0
            }
            ty::Dynamic(_, _, ty::DynStar) => {
                // Dropping a `dyn*`. Need to find actual drop fn.
                self.unpack_dyn_star(&place.into())?.0.assert_mem_place()
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

        let arg = ImmTy::from_immediate(
            place.to_ref(self),
            self.layout_of(Ty::new_mut_ptr(self.tcx.tcx, place.layout.ty))?,
        );
        let ret = MPlaceTy::fake_alloc_zst(self.layout_of(self.tcx.types.unit)?);

        self.eval_fn_call(
            FnVal::Instance(instance),
            (Abi::Rust, fn_abi),
            &[FnArg::Copy(arg.into())],
            false,
            &ret.into(),
            Some(target),
            unwind,
        )
    }
}
