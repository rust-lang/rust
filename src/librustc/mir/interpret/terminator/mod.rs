use rustc::mir;
use rustc::ty::{self, TypeVariants};
use rustc::ty::layout::Layout;
use syntax::codemap::Span;
use syntax::abi::Abi;

use super::{EvalResult, EvalContext, eval_context,
            PtrAndAlign, Lvalue, PrimVal, Value, Machine, ValTy};

use rustc_data_structures::indexed_vec::Idx;

mod drop;

impl<'a, 'tcx, M: Machine<'tcx>> EvalContext<'a, 'tcx, M> {
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
                self.dump_local(self.frame().return_lvalue);
                self.pop_stack_frame()?
            }

            Goto { target } => self.goto_block(target),

            SwitchInt {
                ref discr,
                ref values,
                ref targets,
                ..
            } => {
                // FIXME(CTFE): forbid branching
                let discr_val = self.eval_operand(discr)?;
                let discr_prim = self.value_to_primval(discr_val)?;

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets[targets.len() - 1];

                for (index, const_int) in values.iter().enumerate() {
                    let prim = PrimVal::Bytes(const_int.to_u128_unchecked());
                    if discr_prim.to_bytes()? == prim.to_bytes()? {
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
                    Some((ref lv, target)) => Some((self.eval_lvalue(lv)?, target)),
                    None => None,
                };

                let func_ty = self.operand_ty(func);
                let (fn_def, sig) = match func_ty.sty {
                    ty::TyFnPtr(sig) => {
                        let fn_ptr = self.eval_operand_to_primval(func)?.to_ptr()?;
                        let instance = self.memory.get_fn(fn_ptr)?;
                        let instance_ty = instance.def.def_ty(self.tcx);
                        let instance_ty = self.monomorphize(instance_ty, instance.substs);
                        match instance_ty.sty {
                            ty::TyFnDef(..) => {
                                let real_sig = instance_ty.fn_sig(self.tcx);
                                let sig = self.tcx.erase_late_bound_regions_and_normalize(&sig);
                                let real_sig = self.tcx.erase_late_bound_regions_and_normalize(&real_sig);
                                if !self.check_sig_compat(sig, real_sig)? {
                                    return err!(FunctionPointerTyMismatch(real_sig, sig));
                                }
                            }
                            ref other => bug!("instance def ty: {:?}", other),
                        }
                        (instance, sig)
                    }
                    ty::TyFnDef(def_id, substs) => (
                        eval_context::resolve(self.tcx, def_id, substs),
                        func_ty.fn_sig(self.tcx),
                    ),
                    _ => {
                        let msg = format!("can't handle callee of type {:?}", func_ty);
                        return err!(Unimplemented(msg));
                    }
                };
                let args = self.operands_to_args(args)?;
                let sig = self.tcx.erase_late_bound_regions_and_normalize(&sig);
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
                let lval = self.eval_lvalue(location)?;
                let ty = self.lvalue_ty(location);
                let ty = eval_context::apply_param_substs(self.tcx, self.substs(), &ty);
                trace!("TerminatorKind::drop: {:?}, type {}", location, ty);

                let instance = eval_context::resolve_drop_in_place(self.tcx, ty);
                self.drop_lvalue(
                    lval,
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
                let cond_val = self.eval_operand_to_primval(cond)?.to_bool()?;
                if expected == cond_val {
                    self.goto_block(target);
                } else {
                    use rustc::mir::AssertMessage::*;
                    return match *msg {
                        BoundsCheck { ref len, ref index } => {
                            let span = terminator.source_info.span;
                            let len = self.eval_operand_to_primval(len)
                                .expect("can't eval len")
                                .to_u64()?;
                            let index = self.eval_operand_to_primval(index)
                                .expect("can't eval index")
                                .to_u64()?;
                            err!(ArrayIndexOutOfBounds(span, len, index))
                        }
                        Math(ref err) => {
                            err!(Math(terminator.source_info.span, err.clone()))
                        }
                        GeneratorResumedAfterReturn |
                        GeneratorResumedAfterPanic => unimplemented!(),
                    };
                }
            }

            Yield { .. } => unimplemented!("{:#?}", terminator.kind),
            GeneratorDrop => unimplemented!(),
            DropAndReplace { .. } => unimplemented!(),
            Resume => unimplemented!(),
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
        fn check_ty_compat<'tcx>(ty: ty::Ty<'tcx>, real_ty: ty::Ty<'tcx>) -> bool {
            if ty == real_ty {
                return true;
            } // This is actually a fast pointer comparison
            return match (&ty.sty, &real_ty.sty) {
                // Permit changing the pointer type of raw pointers and references as well as
                // mutability of raw pointers.
                // TODO: Should not be allowed when fat pointers are involved.
                (&TypeVariants::TyRawPtr(_), &TypeVariants::TyRawPtr(_)) => true,
                (&TypeVariants::TyRef(_, _), &TypeVariants::TyRef(_, _)) => {
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
                let layout = self.type_layout(fst_ty)?;
                let size = layout.size(&self.tcx.data_layout).bytes();
                if size == 0 {
                    // Second argument must be a tuple matching the argument list of sig
                    let snd_ty = real_sig.inputs_and_output[1];
                    match snd_ty.sty {
                        TypeVariants::TyTuple(tys, _) if sig.inputs().len() == tys.len() =>
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
        destination: Option<(Lvalue, mir::BasicBlock)>,
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
                let layout = self.type_layout(ty)?;
                M::call_intrinsic(self, instance, args, ret, ty, layout, target)?;
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
                            let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                            self.write_value(valty, dest)?;
                        }
                    }
                    // non capture closure as fn ptr
                    // need to inject zst ptr for closure object (aka do nothing)
                    // and need to pack arguments
                    Abi::Rust => {
                        trace!(
                            "arg_locals: {:?}",
                            self.frame().mir.args_iter().collect::<Vec<_>>()
                        );
                        trace!("args: {:?}", args);
                        let local = arg_locals.nth(1).unwrap();
                        for (i, &valty) in args.into_iter().enumerate() {
                            let dest = self.eval_lvalue(&mir::Lvalue::Local(local).field(
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
                    "arg_locals: {:?}",
                    self.frame().mir.args_iter().collect::<Vec<_>>()
                );
                trace!("args: {:?}", args);
                match sig.abi {
                    Abi::RustCall => {
                        assert_eq!(args.len(), 2);

                        {
                            // write first argument
                            let first_local = arg_locals.next().unwrap();
                            let dest = self.eval_lvalue(&mir::Lvalue::Local(first_local))?;
                            self.write_value(args[0], dest)?;
                        }

                        // unpack and write all other args
                        let layout = self.type_layout(args[1].ty)?;
                        if let (&ty::TyTuple(fields, _),
                                &Layout::Univariant { ref variant, .. }) = (&args[1].ty.sty, layout)
                        {
                            trace!("fields: {:?}", fields);
                            if self.frame().mir.args_iter().count() == fields.len() + 1 {
                                let offsets = variant.offsets.iter().map(|s| s.bytes());
                                match args[1].value {
                                    Value::ByRef(PtrAndAlign { ptr, aligned }) => {
                                        assert!(
                                            aligned,
                                            "Unaligned ByRef-values cannot occur as function arguments"
                                        );
                                        for ((offset, ty), arg_local) in
                                            offsets.zip(fields).zip(arg_locals)
                                        {
                                            let arg = Value::by_ref(ptr.offset(offset, &self)?);
                                            let dest =
                                                self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                                            trace!(
                                                "writing arg {:?} to {:?} (type: {})",
                                                arg,
                                                dest,
                                                ty
                                            );
                                            let valty = ValTy {
                                                value: arg,
                                                ty,
                                            };
                                            self.write_value(valty, dest)?;
                                        }
                                    }
                                    Value::ByVal(PrimVal::Undef) => {}
                                    other => {
                                        assert_eq!(fields.len(), 1);
                                        let dest = self.eval_lvalue(&mir::Lvalue::Local(
                                            arg_locals.next().unwrap(),
                                        ))?;
                                        let valty = ValTy {
                                            value: other,
                                            ty: fields[0],
                                        };
                                        self.write_value(valty, dest)?;
                                    }
                                }
                            } else {
                                trace!("manual impl of rust-call ABI");
                                // called a manual impl of a rust-call function
                                let dest = self.eval_lvalue(
                                    &mir::Lvalue::Local(arg_locals.next().unwrap()),
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
                            let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                            self.write_value(valty, dest)?;
                        }
                    }
                }
                Ok(())
            }
            // cannot use the shim here, because that will only result in infinite recursion
            ty::InstanceDef::Virtual(_, idx) => {
                let ptr_size = self.memory.pointer_size();
                let (ptr, vtable) = args[0].into_ptr_vtable_pair(&self.memory)?;
                let fn_ptr = self.memory.read_ptr_sized_unsigned(
                    vtable.offset(ptr_size * (idx as u64 + 3), &self)?
                )?.to_ptr()?;
                let instance = self.memory.get_fn(fn_ptr)?;
                let mut args = args.to_vec();
                let ty = self.get_field_ty(args[0].ty, 0)?.ty; // TODO: packed flag is ignored
                args[0].ty = ty;
                args[0].value = ptr.to_value();
                // recurse with concrete function
                self.eval_fn_call(instance, destination, &args, span, sig)
            }
        }
    }
}
