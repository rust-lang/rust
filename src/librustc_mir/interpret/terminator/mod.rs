use rustc::mir;
use rustc::ty::{self, Ty};
use rustc::ty::layout::LayoutOf;
use syntax::codemap::Span;
use rustc_target::spec::abi::Abi;

use rustc::mir::interpret::EvalResult;
use super::{EvalContext, Place, Machine, ValTy};

use rustc_data_structures::indexed_vec::Idx;
use interpret::memory::HasMemory;

mod drop;

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub fn goto_block(&mut self, target: mir::BasicBlock) {
        self.frame_mut().block = target;
        self.frame_mut().stmt = 0;
    }

    pub(super) fn eval_terminator(
        &mut self,
        terminator: &mir::Terminator<'tcx>,
    ) -> EvalResult<'tcx> {
        use rustc::mir::TerminatorKind::*;
        match terminator.kind {
            Return => {
                self.dump_local(self.frame().return_place);
                self.pop_stack_frame()?
            }

            Goto { target } => self.goto_block(target),

            SwitchInt {
                ref discr,
                ref values,
                ref targets,
                ..
            } => {
                let discr_val = self.eval_operand(discr)?;
                let discr_prim = self.value_to_scalar(discr_val)?;
                let discr_layout = self.layout_of(discr_val.ty).unwrap();
                trace!("SwitchInt({:?}, {:#?})", discr_prim, discr_layout);
                let discr_prim = discr_prim.to_bits(discr_layout.size)?;

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets[targets.len() - 1];

                for (index, &const_int) in values.iter().enumerate() {
                    if discr_prim == const_int {
                        target_block = targets[index];
                        break;
                    }
                }

                self.goto_block(target_block);
            }

            Call {
                ref func,
                ref args,
                ref destination,
                ..
            } => {
                let destination = match *destination {
                    Some((ref lv, target)) => Some((self.eval_place(lv)?, target)),
                    None => None,
                };

                let func = self.eval_operand(func)?;
                let (fn_def, sig) = match func.ty.sty {
                    ty::TyFnPtr(sig) => {
                        let fn_ptr = self.value_to_scalar(func)?.to_ptr()?;
                        let instance = self.memory.get_fn(fn_ptr)?;
                        let instance_ty = instance.ty(*self.tcx);
                        match instance_ty.sty {
                            ty::TyFnDef(..) => {
                                let real_sig = instance_ty.fn_sig(*self.tcx);
                                let sig = self.tcx.normalize_erasing_late_bound_regions(
                                    ty::ParamEnv::reveal_all(),
                                    &sig,
                                );
                                let real_sig = self.tcx.normalize_erasing_late_bound_regions(
                                    ty::ParamEnv::reveal_all(),
                                    &real_sig,
                                );
                                if !self.check_sig_compat(sig, real_sig)? {
                                    return err!(FunctionPointerTyMismatch(real_sig, sig));
                                }
                            }
                            ref other => bug!("instance def ty: {:?}", other),
                        }
                        (instance, sig)
                    }
                    ty::TyFnDef(def_id, substs) => (
                        self.resolve(def_id, substs)?,
                        func.ty.fn_sig(*self.tcx),
                    ),
                    _ => {
                        let msg = format!("can't handle callee of type {:?}", func.ty);
                        return err!(Unimplemented(msg));
                    }
                };
                let args = self.operands_to_args(args)?;
                let sig = self.tcx.normalize_erasing_late_bound_regions(
                    ty::ParamEnv::reveal_all(),
                    &sig,
                );
                self.eval_fn_call(
                    fn_def,
                    destination,
                    &args,
                    terminator.source_info.span,
                    sig,
                )?;
            }

            Drop {
                ref location,
                target,
                ..
            } => {
                // FIXME(CTFE): forbid drop in const eval
                let place = self.eval_place(location)?;
                let ty = self.place_ty(location);
                let ty = self.tcx.subst_and_normalize_erasing_regions(
                    self.substs(),
                    ty::ParamEnv::reveal_all(),
                    &ty,
                );
                trace!("TerminatorKind::drop: {:?}, type {}", location, ty);

                let instance = ::monomorphize::resolve_drop_in_place(*self.tcx, ty);
                self.drop_place(
                    place,
                    instance,
                    ty,
                    terminator.source_info.span,
                    target,
                )?;
            }

            Assert {
                ref cond,
                expected,
                ref msg,
                target,
                ..
            } => {
                let cond_val = self.eval_operand_to_scalar(cond)?.to_bool()?;
                if expected == cond_val {
                    self.goto_block(target);
                } else {
                    use rustc::mir::interpret::EvalErrorKind::*;
                    return match *msg {
                        BoundsCheck { ref len, ref index } => {
                            let len = self.eval_operand_to_scalar(len)
                                .expect("can't eval len")
                                .to_bits(self.memory().pointer_size())? as u64;
                            let index = self.eval_operand_to_scalar(index)
                                .expect("can't eval index")
                                .to_bits(self.memory().pointer_size())? as u64;
                            err!(BoundsCheck { len, index })
                        }
                        Overflow(op) => Err(Overflow(op).into()),
                        OverflowNeg => Err(OverflowNeg.into()),
                        DivisionByZero => Err(DivisionByZero.into()),
                        RemainderByZero => Err(RemainderByZero.into()),
                        GeneratorResumedAfterReturn |
                        GeneratorResumedAfterPanic => unimplemented!(),
                        _ => bug!(),
                    };
                }
            }

            Yield { .. } => unimplemented!("{:#?}", terminator.kind),
            GeneratorDrop => unimplemented!(),
            DropAndReplace { .. } => unimplemented!(),
            Resume => unimplemented!(),
            Abort => unimplemented!(),
            FalseEdges { .. } => bug!("should have been eliminated by `simplify_branches` mir pass"),
            FalseUnwind { .. } => bug!("should have been eliminated by `simplify_branches` mir pass"),
            Unreachable => return err!(Unreachable),
        }

        Ok(())
    }

    /// Decides whether it is okay to call the method with signature `real_sig` using signature `sig`.
    /// FIXME: This should take into account the platform-dependent ABI description.
    fn check_sig_compat(
        &mut self,
        sig: ty::FnSig<'tcx>,
        real_sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        fn check_ty_compat<'tcx>(ty: Ty<'tcx>, real_ty: Ty<'tcx>) -> bool {
            if ty == real_ty {
                return true;
            } // This is actually a fast pointer comparison
            return match (&ty.sty, &real_ty.sty) {
                // Permit changing the pointer type of raw pointers and references as well as
                // mutability of raw pointers.
                // TODO: Should not be allowed when fat pointers are involved.
                (&ty::TyRawPtr(_), &ty::TyRawPtr(_)) => true,
                (&ty::TyRef(_, _, _), &ty::TyRef(_, _, _)) => {
                    ty.is_mutable_pointer() == real_ty.is_mutable_pointer()
                }
                // rule out everything else
                _ => false,
            };
        }

        if sig.abi == real_sig.abi && sig.variadic == real_sig.variadic &&
            sig.inputs_and_output.len() == real_sig.inputs_and_output.len() &&
            sig.inputs_and_output
                .iter()
                .zip(real_sig.inputs_and_output)
                .all(|(ty, real_ty)| check_ty_compat(ty, real_ty))
        {
            // Definitely good.
            return Ok(true);
        }

        if sig.variadic || real_sig.variadic {
            // We're not touching this
            return Ok(false);
        }

        // We need to allow what comes up when a non-capturing closure is cast to a fn().
        match (sig.abi, real_sig.abi) {
            (Abi::Rust, Abi::RustCall) // check the ABIs.  This makes the test here non-symmetric.
                if check_ty_compat(sig.output(), real_sig.output()) && real_sig.inputs_and_output.len() == 3 => {
                // First argument of real_sig must be a ZST
                let fst_ty = real_sig.inputs_and_output[0];
                if self.layout_of(fst_ty)?.is_zst() {
                    // Second argument must be a tuple matching the argument list of sig
                    let snd_ty = real_sig.inputs_and_output[1];
                    match snd_ty.sty {
                        ty::TyTuple(tys) if sig.inputs().len() == tys.len() =>
                            if sig.inputs().iter().zip(tys).all(|(ty, real_ty)| check_ty_compat(ty, real_ty)) {
                                return Ok(true)
                            },
                        _ => {}
                    }
                }
            }
            _ => {}
        };

        // Nope, this doesn't work.
        return Ok(false);
    }

    fn eval_fn_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx> {
        trace!("eval_fn_call: {:#?}", instance);
        match instance.def {
            ty::InstanceDef::Intrinsic(..) => {
                let (ret, target) = match destination {
                    Some(dest) => dest,
                    _ => return err!(Unreachable),
                };
                let ty = sig.output();
                let layout = self.layout_of(ty)?;
                M::call_intrinsic(self, instance, args, ret, layout, target)?;
                self.dump_local(ret);
                Ok(())
            }
            // FIXME: figure out why we can't just go through the shim
            ty::InstanceDef::ClosureOnceShim { .. } => {
                if M::eval_fn_call(self, instance, destination, args, span, sig)? {
                    return Ok(());
                }
                let mut arg_locals = self.frame().mir.args_iter();
                match sig.abi {
                    // closure as closure once
                    Abi::RustCall => {
                        for (arg_local, &valty) in arg_locals.zip(args) {
                            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                            self.write_value(valty, dest)?;
                        }
                    }
                    // non capture closure as fn ptr
                    // need to inject zst ptr for closure object (aka do nothing)
                    // and need to pack arguments
                    Abi::Rust => {
                        trace!(
                            "arg_locals: {:#?}",
                            self.frame().mir.args_iter().collect::<Vec<_>>()
                        );
                        trace!("args: {:#?}", args);
                        let local = arg_locals.nth(1).unwrap();
                        for (i, &valty) in args.into_iter().enumerate() {
                            let dest = self.eval_place(&mir::Place::Local(local).field(
                                mir::Field::new(i),
                                valty.ty,
                            ))?;
                            self.write_value(valty, dest)?;
                        }
                    }
                    _ => bug!("bad ABI for ClosureOnceShim: {:?}", sig.abi),
                }
                Ok(())
            }
            ty::InstanceDef::FnPtrShim(..) |
            ty::InstanceDef::DropGlue(..) |
            ty::InstanceDef::CloneShim(..) |
            ty::InstanceDef::Item(_) => {
                // Push the stack frame, and potentially be entirely done if the call got hooked
                if M::eval_fn_call(self, instance, destination, args, span, sig)? {
                    return Ok(());
                }

                // Pass the arguments
                let mut arg_locals = self.frame().mir.args_iter();
                trace!("ABI: {:?}", sig.abi);
                trace!(
                    "arg_locals: {:#?}",
                    self.frame().mir.args_iter().collect::<Vec<_>>()
                );
                trace!("args: {:#?}", args);
                match sig.abi {
                    Abi::RustCall => {
                        assert_eq!(args.len(), 2);

                        {
                            // write first argument
                            let first_local = arg_locals.next().unwrap();
                            let dest = self.eval_place(&mir::Place::Local(first_local))?;
                            self.write_value(args[0], dest)?;
                        }

                        // unpack and write all other args
                        let layout = self.layout_of(args[1].ty)?;
                        if let ty::TyTuple(_) = args[1].ty.sty {
                            if layout.is_zst() {
                                // Nothing to do, no need to unpack zsts
                                return Ok(());
                            }
                            if self.frame().mir.args_iter().count() == layout.fields.count() + 1 {
                                for (i, arg_local) in arg_locals.enumerate() {
                                    let field = mir::Field::new(i);
                                    let valty = self.read_field(args[1].value, None, field, args[1].ty)?;
                                    let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                                    self.write_value(valty, dest)?;
                                }
                            } else {
                                trace!("manual impl of rust-call ABI");
                                // called a manual impl of a rust-call function
                                let dest = self.eval_place(
                                    &mir::Place::Local(arg_locals.next().unwrap()),
                                )?;
                                self.write_value(args[1], dest)?;
                            }
                        } else {
                            bug!(
                                "rust-call ABI tuple argument was {:#?}, {:#?}",
                                args[1].ty,
                                layout
                            );
                        }
                    }
                    _ => {
                        for (arg_local, &valty) in arg_locals.zip(args) {
                            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                            self.write_value(valty, dest)?;
                        }
                    }
                }
                Ok(())
            }
            // cannot use the shim here, because that will only result in infinite recursion
            ty::InstanceDef::Virtual(_, idx) => {
                let ptr_size = self.memory.pointer_size();
                let ptr_align = self.tcx.data_layout.pointer_align;
                let (ptr, vtable) = self.into_ptr_vtable_pair(args[0].value)?;
                let fn_ptr = self.memory.read_ptr_sized(
                    vtable.offset(ptr_size * (idx as u64 + 3), &self)?,
                    ptr_align
                )?.to_ptr()?;
                let instance = self.memory.get_fn(fn_ptr)?;
                let mut args = args.to_vec();
                let ty = self.layout_of(args[0].ty)?.field(&self, 0)?.ty;
                args[0].ty = ty;
                args[0].value = ptr.to_value();
                // recurse with concrete function
                self.eval_fn_call(instance, destination, &args, span, sig)
            }
        }
    }
}
