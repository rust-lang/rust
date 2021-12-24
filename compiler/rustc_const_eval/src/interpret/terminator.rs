use std::borrow::Cow;
use std::convert::TryFrom;

use rustc_middle::ty::layout::{FnAbiOf, LayoutOf};
use rustc_middle::ty::Instance;
use rustc_middle::{
    mir,
    ty::{self, Ty},
};
use rustc_target::abi;
use rustc_target::abi::call::{ArgAbi, ArgAttribute, ArgAttributes, FnAbi, PassMode};
use rustc_target::spec::abi::Abi;

use super::{
    FnVal, ImmTy, InterpCx, InterpResult, MPlaceTy, Machine, OpTy, PlaceTy, Scalar,
    StackPopCleanup, StackPopUnwind,
};

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
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

            SwitchInt { ref discr, ref targets, switch_ty } => {
                let discr = self.read_immediate(&self.eval_operand(discr, None)?)?;
                trace!("SwitchInt({:?})", *discr);
                assert_eq!(discr.layout.ty, switch_ty);

                // Branch to the `otherwise` case by default, if no match is found.
                assert!(!targets.iter().is_empty());
                let mut target_block = targets.otherwise();

                for (const_int, target) in targets.iter() {
                    // Compare using binary_op, to also support pointer values
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

            Call { ref func, ref args, destination, ref cleanup, from_hir_call: _, fn_span: _ } => {
                let old_stack = self.frame_idx();
                let old_loc = self.frame().loc;
                let func = self.eval_operand(func, None)?;
                let args = self.eval_operands(args)?;

                let fn_sig_binder = func.layout.ty.fn_sig(*self.tcx);
                let fn_sig =
                    self.tcx.normalize_erasing_late_bound_regions(self.param_env, fn_sig_binder);
                let extra_args = &args[fn_sig.inputs().len()..];
                let extra_args = self.tcx.mk_type_list(extra_args.iter().map(|arg| arg.layout.ty));

                let (fn_val, fn_abi, with_caller_location) = match *func.layout.ty.kind() {
                    ty::FnPtr(_sig) => {
                        let fn_ptr = self.read_pointer(&func)?;
                        let fn_val = self.memory.get_fn(fn_ptr)?;
                        (fn_val, self.fn_abi_of_fn_ptr(fn_sig_binder, extra_args)?, false)
                    }
                    ty::FnDef(def_id, substs) => {
                        let instance =
                            self.resolve(ty::WithOptConstParam::unknown(def_id), substs)?;
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

                let dest_place;
                let ret = match destination {
                    Some((dest, ret)) => {
                        dest_place = self.eval_place(dest)?;
                        Some((&dest_place, ret))
                    }
                    None => None,
                };
                self.eval_fn_call(
                    fn_val,
                    (fn_sig.abi, fn_abi),
                    &args,
                    with_caller_location,
                    ret,
                    match (cleanup, fn_abi.can_unwind) {
                        (Some(cleanup), true) => StackPopUnwind::Cleanup(*cleanup),
                        (None, true) => StackPopUnwind::Skip,
                        (_, false) => StackPopUnwind::NotAllowed,
                    },
                )?;
                // Sanity-check that `eval_fn_call` either pushed a new frame or
                // did a jump to another block.
                if self.frame_idx() == old_stack && self.frame().loc == old_loc {
                    span_bug!(terminator.source_info.span, "evaluating this call made no progress");
                }
            }

            Drop { place, target, unwind } => {
                let place = self.eval_place(place)?;
                let ty = place.layout.ty;
                trace!("TerminatorKind::drop: {:?}, type {}", place, ty);

                let instance = Instance::resolve_drop_in_place(*self.tcx, ty);
                self.drop_in_place(&place, instance, target, unwind)?;
            }

            Assert { ref cond, expected, ref msg, target, cleanup } => {
                let cond_val =
                    self.read_immediate(&self.eval_operand(cond, None)?)?.to_scalar()?.to_bool()?;
                if expected == cond_val {
                    self.go_to_block(target);
                } else {
                    M::assert_panic(self, msg, cleanup)?;
                }
            }

            Abort => {
                M::abort(self, "the program aborted execution".to_owned())?;
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
            DropAndReplace { .. }
            | FalseEdge { .. }
            | FalseUnwind { .. }
            | Yield { .. }
            | GeneratorDrop => span_bug!(
                terminator.source_info.span,
                "{:#?} should have been eliminated by MIR pass",
                terminator.kind
            ),

            // Inline assembly can't be interpreted.
            InlineAsm { .. } => throw_unsup_format!("inline assembly is not supported"),
        }

        Ok(())
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
            // Compare layout
            match (caller_abi.layout.abi, callee_abi.layout.abi) {
                // Different valid ranges are okay (once we enforce validity,
                // that will take care to make it UB to leave the range, just
                // like for transmute).
                (abi::Abi::Scalar(caller), abi::Abi::Scalar(callee)) => {
                    caller.value == callee.value
                }
                (
                    abi::Abi::ScalarPair(caller1, caller2),
                    abi::Abi::ScalarPair(callee1, callee2),
                ) => caller1.value == callee1.value && caller2.value == callee2.value,
                // Be conservative
                _ => false,
            }
        };
        // Padding must be fully equal.
        let pad_compat = || caller_abi.pad == callee_abi.pad;
        // When comparing the PassMode, we have to be smart about comparing the attributes.
        let arg_attr_compat = |a1: ArgAttributes, a2: ArgAttributes| {
            // There's only one regular attribute that matters for the call ABI: InReg.
            // Everything else is things like noalias, dereferencable, nonnull, ...
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
        let mode_compat = || match (caller_abi.mode, callee_abi.mode) {
            (PassMode::Ignore, PassMode::Ignore) => true,
            (PassMode::Direct(a1), PassMode::Direct(a2)) => arg_attr_compat(a1, a2),
            (PassMode::Pair(a1, b1), PassMode::Pair(a2, b2)) => {
                arg_attr_compat(a1, a2) && arg_attr_compat(b1, b2)
            }
            (PassMode::Cast(c1), PassMode::Cast(c2)) => c1 == c2,
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

        if layout_compat() && pad_compat() && mode_compat() {
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
            Item = (&'x OpTy<'tcx, M::PointerTag>, &'y ArgAbi<'tcx, Ty<'tcx>>),
        >,
        callee_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        callee_arg: &PlaceTy<'tcx, M::PointerTag>,
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
        let (caller_arg, caller_abi) = caller_args.next().ok_or_else(|| {
            err_ub_format!("calling a function with fewer arguments than it requires")
        })?;
        // Now, check
        if !Self::check_argument_compat(caller_abi, callee_abi) {
            throw_ub_format!(
                "calling a function with argument of type {:?} passing data of type {:?}",
                callee_arg.layout.ty,
                caller_arg.layout.ty
            )
        }
        // We allow some transmutes here.
        // FIXME: Depending on the PassMode, this should reset some padding to uninitialized. (This
        // is true for all `copy_op`, but there are a lot of special cases for argument passing
        // specifically.)
        self.copy_op_transmute(&caller_arg, callee_arg)
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
        args: &[OpTy<'tcx, M::PointerTag>],
        with_caller_location: bool,
        ret: Option<(&PlaceTy<'tcx, M::PointerTag>, mir::BasicBlock)>,
        mut unwind: StackPopUnwind,
    ) -> InterpResult<'tcx> {
        trace!("eval_fn_call: {:#?}", fn_val);

        let instance = match fn_val {
            FnVal::Instance(instance) => instance,
            FnVal::Other(extra) => {
                return M::call_extra_fn(self, extra, caller_abi, args, ret, unwind);
            }
        };

        match instance.def {
            ty::InstanceDef::Intrinsic(..) => {
                assert!(caller_abi == Abi::RustIntrinsic || caller_abi == Abi::PlatformIntrinsic);
                // caller_fn_abi is not relevant here, we interpret the arguments directly for each intrinsic.
                M::call_intrinsic(self, instance, args, ret, unwind)
            }
            ty::InstanceDef::VtableShim(..)
            | ty::InstanceDef::ReifyShim(..)
            | ty::InstanceDef::ClosureOnceShim { .. }
            | ty::InstanceDef::FnPtrShim(..)
            | ty::InstanceDef::DropGlue(..)
            | ty::InstanceDef::CloneShim(..)
            | ty::InstanceDef::Item(_) => {
                // We need MIR for this fn
                let (body, instance) =
                    match M::find_mir_or_eval_fn(self, instance, caller_abi, args, ret, unwind)? {
                        Some(body) => body,
                        None => return Ok(()),
                    };

                // Compute callee information using the `instance` returned by
                // `find_mir_or_eval_fn`.
                // FIXME: for variadic support, do we have to somehow determine calle's extra_args?
                let callee_fn_abi = self.fn_abi_of_instance(instance, ty::List::empty())?;

                if callee_fn_abi.c_variadic != caller_fn_abi.c_variadic {
                    throw_ub_format!(
                        "calling a c-variadic function via a non-variadic call site, or vice versa"
                    );
                }
                if callee_fn_abi.c_variadic {
                    throw_unsup_format!("calling a c-variadic function is not supported");
                }

                if M::enforce_abi(self) {
                    if caller_fn_abi.conv != callee_fn_abi.conv {
                        throw_ub_format!(
                            "calling a function with calling convention {:?} using calling convention {:?}",
                            callee_fn_abi.conv,
                            caller_fn_abi.conv
                        )
                    }
                }

                if !matches!(unwind, StackPopUnwind::NotAllowed) && !callee_fn_abi.can_unwind {
                    // The callee cannot unwind.
                    unwind = StackPopUnwind::NotAllowed;
                }

                self.push_stack_frame(
                    instance,
                    body,
                    ret.map(|p| p.0),
                    StackPopCleanup::Goto { ret: ret.map(|p| p.1), unwind },
                )?;

                // If an error is raised here, pop the frame again to get an accurate backtrace.
                // To this end, we wrap it all in a `try` block.
                let res: InterpResult<'tcx> = try {
                    trace!(
                        "caller ABI: {:?}, args: {:#?}",
                        caller_abi,
                        args.iter()
                            .map(|arg| (arg.layout.ty, format!("{:?}", **arg)))
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
                    // last incoming argument.  These two iterators do not have the same type,
                    // so to keep the code paths uniform we accept an allocation
                    // (for RustCall ABI only).
                    let caller_args: Cow<'_, [OpTy<'tcx, M::PointerTag>]> =
                        if caller_abi == Abi::RustCall && !args.is_empty() {
                            // Untuple
                            let (untuple_arg, args) = args.split_last().unwrap();
                            trace!("eval_fn_call: Will pass last argument by untupling");
                            Cow::from(
                                args.iter()
                                    .map(|&a| Ok(a))
                                    .chain(
                                        (0..untuple_arg.layout.fields.count())
                                            .map(|i| self.operand_field(untuple_arg, i)),
                                    )
                                    .collect::<InterpResult<'_, Vec<OpTy<'tcx, M::PointerTag>>>>(
                                    )?,
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
                    // taking into account the `spread_arg`.  If we could write
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
                        throw_ub_format!("calling a function with more arguments than it expected")
                    }
                    // Don't forget to check the return type!
                    if !Self::check_argument_compat(&caller_fn_abi.ret, &callee_fn_abi.ret) {
                        throw_ub_format!(
                            "calling a function with return type {:?} passing \
                                    return place of type {:?}",
                            callee_fn_abi.ret.layout.ty,
                            caller_fn_abi.ret.layout.ty,
                        )
                    }
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
            ty::InstanceDef::Virtual(_, idx) => {
                let mut args = args.to_vec();
                // We have to implement all "object safe receivers".  Currently we
                // support built-in pointers `(&, &mut, Box)` as well as unsized-self.  We do
                // not yet support custom self types.
                // Also see `compiler/rustc_codegen_llvm/src/abi.rs` and `compiler/rustc_codegen_ssa/src/mir/block.rs`.
                let receiver_place = match args[0].layout.ty.builtin_deref(true) {
                    Some(_) => {
                        // Built-in pointer.
                        self.deref_operand(&args[0])?
                    }
                    None => {
                        // Unsized self.
                        args[0].assert_mem_place()
                    }
                };
                // Find and consult vtable
                let vtable = self.scalar_to_ptr(receiver_place.vtable());
                let fn_val = self.get_vtable_slot(vtable, u64::try_from(idx).unwrap())?;

                // `*mut receiver_place.layout.ty` is almost the layout that we
                // want for args[0]: We have to project to field 0 because we want
                // a thin pointer.
                assert!(receiver_place.layout.is_unsized());
                let receiver_ptr_ty = self.tcx.mk_mut_ptr(receiver_place.layout.ty);
                let this_receiver_ptr = self.layout_of(receiver_ptr_ty)?.field(self, 0);
                // Adjust receiver argument.
                args[0] = OpTy::from(ImmTy::from_immediate(
                    Scalar::from_maybe_pointer(receiver_place.ptr, self).into(),
                    this_receiver_ptr,
                ));
                trace!("Patched self operand to {:#?}", args[0]);
                // recurse with concrete function
                self.eval_fn_call(
                    fn_val,
                    (caller_abi, caller_fn_abi),
                    &args,
                    with_caller_location,
                    ret,
                    unwind,
                )
            }
        }
    }

    fn drop_in_place(
        &mut self,
        place: &PlaceTy<'tcx, M::PointerTag>,
        instance: ty::Instance<'tcx>,
        target: mir::BasicBlock,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        trace!("drop_in_place: {:?},\n  {:?}, {:?}", *place, place.layout.ty, instance);
        // We take the address of the object.  This may well be unaligned, which is fine
        // for us here.  However, unaligned accesses will probably make the actual drop
        // implementation fail -- a problem shared by rustc.
        let place = self.force_allocation(place)?;

        let (instance, place) = match place.layout.ty.kind() {
            ty::Dynamic(..) => {
                // Dropping a trait object.
                self.unpack_dyn_trait(&place)?
            }
            _ => (instance, place),
        };
        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty())?;

        let arg = ImmTy::from_immediate(
            place.to_ref(self),
            self.layout_of(self.tcx.mk_mut_ptr(place.layout.ty))?,
        );

        let ty = self.tcx.mk_unit(); // return type is ()
        let dest = MPlaceTy::dangling(self.layout_of(ty)?);

        self.eval_fn_call(
            FnVal::Instance(instance),
            (Abi::Rust, fn_abi),
            &[arg.into()],
            false,
            Some((&dest.into(), target)),
            match unwind {
                Some(cleanup) => StackPopUnwind::Cleanup(cleanup),
                None => StackPopUnwind::Skip,
            },
        )
    }
}
