// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir;
use rustc::ty::{self, Ty};
use rustc::ty::layout::LayoutOf;
use syntax::source_map::Span;
use rustc_target::spec::abi::Abi;

use rustc::mir::interpret::{EvalResult, Scalar};
use super::{EvalContext, Machine, Value, OpTy, PlaceTy, ValTy, Operand};

use rustc_data_structures::indexed_vec::Idx;

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
                self.dump_place(self.frame().return_place);
                self.pop_stack_frame()?
            }

            Goto { target } => self.goto_block(target),

            SwitchInt {
                ref discr,
                ref values,
                ref targets,
                ..
            } => {
                let discr_val = self.eval_operand(discr, None)?;
                let discr = self.read_value(discr_val)?;
                trace!("SwitchInt({:?})", *discr);

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets[targets.len() - 1];

                for (index, &const_int) in values.iter().enumerate() {
                    // Compare using binary_op
                    let const_int = Scalar::Bits {
                        bits: const_int,
                        size: discr.layout.size.bytes() as u8
                    };
                    let (res, _) = self.binary_op(mir::BinOp::Eq,
                        discr,
                        ValTy { value: Value::Scalar(const_int.into()), layout: discr.layout }
                    )?;
                    if res.to_bool()? {
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

                let func = self.eval_operand(func, None)?;
                let (fn_def, sig) = match func.layout.ty.sty {
                    ty::FnPtr(sig) => {
                        let fn_ptr = self.read_scalar(func)?.to_ptr()?;
                        let instance = self.memory.get_fn(fn_ptr)?;
                        let instance_ty = instance.ty(*self.tcx);
                        match instance_ty.sty {
                            ty::FnDef(..) => {
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
                    ty::FnDef(def_id, substs) => (
                        self.resolve(def_id, substs)?,
                        func.layout.ty.fn_sig(*self.tcx),
                    ),
                    _ => {
                        let msg = format!("can't handle callee of type {:?}", func.layout.ty);
                        return err!(Unimplemented(msg));
                    }
                };
                let args = self.eval_operands(args)?;
                let sig = self.tcx.normalize_erasing_late_bound_regions(
                    ty::ParamEnv::reveal_all(),
                    &sig,
                );
                self.eval_fn_call(
                    fn_def,
                    destination,
                    &args[..],
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
                let ty = place.layout.ty;
                trace!("TerminatorKind::drop: {:?}, type {}", location, ty);

                let instance = ::monomorphize::resolve_drop_in_place(*self.tcx, ty);
                self.drop_in_place(
                    place,
                    instance,
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
                let cond_val = self.eval_operand_and_read_value(cond, None)?
                    .to_scalar()?
                    .to_bool()?;
                if expected == cond_val {
                    self.goto_block(target);
                } else {
                    use rustc::mir::interpret::EvalErrorKind::*;
                    return match *msg {
                        BoundsCheck { ref len, ref index } => {
                            let len = self.eval_operand_and_read_value(len, None)
                                .expect("can't eval len").to_scalar()?
                                .to_bits(self.memory().pointer_size())? as u64;
                            let index = self.eval_operand_and_read_value(index, None)
                                .expect("can't eval index").to_scalar()?
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
            FalseEdges { .. } => bug!("should have been eliminated by\
                                      `simplify_branches` mir pass"),
            FalseUnwind { .. } => bug!("should have been eliminated by\
                                       `simplify_branches` mir pass"),
            Unreachable => return err!(Unreachable),
        }

        Ok(())
    }

    /// Decides whether it is okay to call the method with signature `real_sig`
    /// using signature `sig`.
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
                // FIXME: Should not be allowed when fat pointers are involved.
                (&ty::RawPtr(_), &ty::RawPtr(_)) => true,
                (&ty::Ref(_, _, _), &ty::Ref(_, _, _)) => {
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
                if check_ty_compat(sig.output(), real_sig.output())
                    && real_sig.inputs_and_output.len() == 3 => {
                // First argument of real_sig must be a ZST
                let fst_ty = real_sig.inputs_and_output[0];
                if self.layout_of(fst_ty)?.is_zst() {
                    // Second argument must be a tuple matching the argument list of sig
                    let snd_ty = real_sig.inputs_and_output[1];
                    match snd_ty.sty {
                        ty::Tuple(tys) if sig.inputs().len() == tys.len() =>
                            if sig.inputs()
                                .iter()
                                .zip(tys)
                                .all(|(ty, real_ty)| check_ty_compat(ty, real_ty)) {
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
        destination: Option<(PlaceTy<'tcx>, mir::BasicBlock)>,
        args: &[OpTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx> {
        trace!("eval_fn_call: {:#?}", instance);
        if let Some((place, _)) = destination {
            assert_eq!(place.layout.ty, sig.output());
        }
        match instance.def {
            ty::InstanceDef::Intrinsic(..) => {
                let (ret, target) = match destination {
                    Some(dest) => dest,
                    _ => return err!(Unreachable),
                };
                M::call_intrinsic(self, instance, args, ret, target)?;
                self.dump_place(*ret);
                Ok(())
            }
            // FIXME: figure out why we can't just go through the shim
            ty::InstanceDef::ClosureOnceShim { .. } => {
                if M::eval_fn_call(self, instance, destination, args, span)? {
                    return Ok(());
                }
                let mut arg_locals = self.frame().mir.args_iter();
                match sig.abi {
                    // closure as closure once
                    Abi::RustCall => {
                        for (arg_local, &op) in arg_locals.zip(args) {
                            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                            self.copy_op(op, dest)?;
                        }
                    }
                    // non capture closure as fn ptr
                    // need to inject zst ptr for closure object (aka do nothing)
                    // and need to pack arguments
                    Abi::Rust => {
                        trace!(
                            "args: {:#?}",
                            self.frame().mir.args_iter().zip(args.iter())
                                .map(|(local, arg)| (local, **arg, arg.layout.ty))
                                .collect::<Vec<_>>()
                        );
                        let local = arg_locals.nth(1).unwrap();
                        for (i, &op) in args.into_iter().enumerate() {
                            let dest = self.eval_place(&mir::Place::Local(local).field(
                                mir::Field::new(i),
                                op.layout.ty,
                            ))?;
                            self.copy_op(op, dest)?;
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
                if M::eval_fn_call(self, instance, destination, args, span)? {
                    // FIXME: Can we make it return the frame to push, instead
                    // of the hook doing half of the work and us doing the argument
                    // initialization?
                    return Ok(());
                }

                // Pass the arguments
                let mut arg_locals = self.frame().mir.args_iter();
                trace!("ABI: {:?}", sig.abi);
                trace!(
                    "args: {:#?}",
                    self.frame().mir.args_iter().zip(args.iter())
                        .map(|(local, arg)| (local, **arg, arg.layout.ty)).collect::<Vec<_>>()
                );
                match sig.abi {
                    Abi::RustCall => {
                        assert_eq!(args.len(), 2);

                        {
                            // write first argument
                            let first_local = arg_locals.next().unwrap();
                            let dest = self.eval_place(&mir::Place::Local(first_local))?;
                            self.copy_op(args[0], dest)?;
                        }

                        // unpack and write all other args
                        let layout = args[1].layout;
                        if let ty::Tuple(_) = layout.ty.sty {
                            if layout.is_zst() {
                                // Nothing to do, no need to unpack zsts
                                return Ok(());
                            }
                            if self.frame().mir.args_iter().count() == layout.fields.count() + 1 {
                                for (i, arg_local) in arg_locals.enumerate() {
                                    let arg = self.operand_field(args[1], i as u64)?;
                                    let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                                    self.copy_op(arg, dest)?;
                                }
                            } else {
                                trace!("manual impl of rust-call ABI");
                                // called a manual impl of a rust-call function
                                let dest = self.eval_place(
                                    &mir::Place::Local(arg_locals.next().unwrap()),
                                )?;
                                self.copy_op(args[1], dest)?;
                            }
                        } else {
                            bug!(
                                "rust-call ABI tuple argument was {:#?}",
                                layout
                            );
                        }
                    }
                    _ => {
                        for (arg_local, &op) in arg_locals.zip(args) {
                            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                            self.copy_op(op, dest)?;
                        }
                    }
                }
                Ok(())
            }
            // cannot use the shim here, because that will only result in infinite recursion
            ty::InstanceDef::Virtual(_, idx) => {
                let ptr_size = self.memory.pointer_size();
                let ptr_align = self.tcx.data_layout.pointer_align;
                let (ptr, vtable) = self.read_value(args[0])?.to_scalar_dyn_trait()?;
                let fn_ptr = self.memory.read_ptr_sized(
                    vtable.offset(ptr_size * (idx as u64 + 3), &self)?,
                    ptr_align
                )?.to_ptr()?;
                let instance = self.memory.get_fn(fn_ptr)?;

                // We have to patch the self argument, in particular get the layout
                // expected by the actual function. Cannot just use "field 0" due to
                // Box<self>.
                let mut args = args.to_vec();
                let pointee = args[0].layout.ty.builtin_deref(true).unwrap().ty;
                let fake_fat_ptr_ty = self.tcx.mk_mut_ptr(pointee);
                args[0].layout = self.layout_of(fake_fat_ptr_ty)?.field(&self, 0)?;
                args[0].op = Operand::Immediate(Value::Scalar(ptr.into())); // strip vtable
                trace!("Patched self operand to {:#?}", args[0]);
                // recurse with concrete function
                self.eval_fn_call(instance, destination, &args, span, sig)
            }
        }
    }
}
