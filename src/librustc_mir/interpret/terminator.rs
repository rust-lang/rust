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
use super::{
    EvalContext, Machine, Value, OpTy, Place, PlaceTy, PlaceExtra, ValTy, Operand, StackPopCleanup
};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    #[inline]
    pub fn goto_block(&mut self, target: Option<mir::BasicBlock>) -> EvalResult<'tcx> {
        if let Some(target) = target {
            self.frame_mut().block = target;
            self.frame_mut().stmt = 0;
            Ok(())
        } else {
            err!(Unreachable)
        }
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

            Goto { target } => self.goto_block(Some(target))?,

            SwitchInt {
                ref discr,
                ref values,
                ref targets,
                ..
            } => {
                let discr = self.read_value(self.eval_operand(discr, None)?)?;
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

                self.goto_block(Some(target_block))?;
            }

            Call {
                ref func,
                ref args,
                ref destination,
                ..
            } => {
                let (dest, ret) = match *destination {
                    Some((ref lv, target)) => (Some(self.eval_place(lv)?), Some(target)),
                    None => (None, None),
                };

                let func = self.eval_operand(func, None)?;
                let (fn_def, sig) = match func.layout.ty.sty {
                    ty::FnPtr(sig) => {
                        let fn_ptr = self.read_scalar(func)?.to_ptr()?;
                        let instance = self.memory.get_fn(fn_ptr)?;
                        let instance_ty = instance.ty(*self.tcx);
                        match instance_ty.sty {
                            ty::FnDef(..) => {
                                let sig = self.tcx.normalize_erasing_late_bound_regions(
                                    self.param_env,
                                    &sig,
                                );
                                let real_sig = instance_ty.fn_sig(*self.tcx);
                                let real_sig = self.tcx.normalize_erasing_late_bound_regions(
                                    self.param_env,
                                    &real_sig,
                                );
                                if !self.check_sig_compat(sig, real_sig)? {
                                    return err!(FunctionPointerTyMismatch(real_sig, sig));
                                }
                                (instance, sig)
                            }
                            ref other => bug!("instance def ty: {:?}", other),
                        }
                    }
                    ty::FnDef(def_id, substs) => {
                        let sig = func.layout.ty.fn_sig(*self.tcx);
                        let sig = self.tcx.normalize_erasing_late_bound_regions(
                            self.param_env,
                            &sig,
                        );
                        (self.resolve(def_id, substs)?, sig)
                    },
                    _ => {
                        let msg = format!("can't handle callee of type {:?}", func.layout.ty);
                        return err!(Unimplemented(msg));
                    }
                };
                let args = self.eval_operands(args)?;
                self.eval_fn_call(
                    fn_def,
                    &args[..],
                    dest,
                    ret,
                    terminator.source_info.span,
                    Some(sig),
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
                let cond_val = self.read_value(self.eval_operand(cond, None)?)?
                    .to_scalar()?.to_bool()?;
                if expected == cond_val {
                    self.goto_block(Some(target))?;
                } else {
                    use rustc::mir::interpret::EvalErrorKind::*;
                    return match *msg {
                        BoundsCheck { ref len, ref index } => {
                            let len = self.read_value(self.eval_operand(len, None)?)
                                .expect("can't eval len").to_scalar()?
                                .to_bits(self.memory().pointer_size())? as u64;
                            let index = self.read_value(self.eval_operand(index, None)?)
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

    /// Call this function -- pushing the stack frame and initializing the arguments.
    /// `sig` is ptional in case of FnPtr/FnDef -- but mandatory for closures!
    fn eval_fn_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: Option<PlaceTy<'tcx>>,
        ret: Option<mir::BasicBlock>,
        span: Span,
        sig: Option<ty::FnSig<'tcx>>,
    ) -> EvalResult<'tcx> {
        trace!("eval_fn_call: {:#?}", instance);

        match instance.def {
            ty::InstanceDef::Intrinsic(..) => {
                // The intrinsic itself cannot diverge, so if we got here without a return
                // place... (can happen e.g. for transmute returning `!`)
                let dest = match dest {
                    Some(dest) => dest,
                    None => return err!(Unreachable)
                };
                M::call_intrinsic(self, instance, args, dest)?;
                // No stack frame gets pushed, the main loop will just act as if the
                // call completed.
                self.goto_block(ret)?;
                self.dump_place(*dest);
                Ok(())
            }
            ty::InstanceDef::ClosureOnceShim { .. } |
            ty::InstanceDef::FnPtrShim(..) |
            ty::InstanceDef::DropGlue(..) |
            ty::InstanceDef::CloneShim(..) |
            ty::InstanceDef::Item(_) => {
                let mir = match M::find_fn(self, instance, args, dest, ret)? {
                    Some(mir) => mir,
                    None => return Ok(()),
                };

                let return_place = match dest {
                    Some(place) => *place,
                    None => Place::null(&self),
                };
                self.push_stack_frame(
                    instance,
                    span,
                    mir,
                    return_place,
                    StackPopCleanup::Goto(ret),
                )?;

                // If we didn't get a signture, ask `fn_sig`
                let sig = sig.unwrap_or_else(|| {
                    let fn_sig = instance.ty(*self.tcx).fn_sig(*self.tcx);
                    self.tcx.normalize_erasing_late_bound_regions(self.param_env, &fn_sig)
                });
                assert_eq!(sig.inputs().len(), args.len());
                // We can't test the types, as it is fine if the types are ABI-compatible but
                // not equal.

                // Figure out how to pass which arguments.
                // FIXME: Somehow this is horribly full of special cases here, and codegen has
                // none of that.  What is going on?
                trace!("ABI: {:?}", sig.abi);
                trace!(
                    "args: {:#?}",
                    args.iter()
                        .map(|arg| (arg.layout.ty, format!("{:?}", **arg)))
                        .collect::<Vec<_>>()
                );
                trace!(
                    "locals: {:#?}",
                    mir.args_iter()
                        .map(|local|
                            (local, self.layout_of_local(self.cur_frame(), local).unwrap().ty)
                        )
                        .collect::<Vec<_>>()
                );
                match instance.def {
                    ty::InstanceDef::ClosureOnceShim { .. } if sig.abi == Abi::Rust => {
                        // this has an entirely ridicolous calling convention where it uses the
                        // "Rust" ABI, but arguments come in untupled and are supposed to be tupled
                        // for the callee!  The function's first argument is a ZST, and then
                        // there comes a tuple for the rest.
                        let mut arg_locals = mir.args_iter();

                        {   // the ZST. nothing to write.
                            let arg_local = arg_locals.next().unwrap();
                            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                            assert!(dest.layout.is_zst());
                        }

                        {   // the tuple argument.
                            let arg_local = arg_locals.next().unwrap();
                            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                            assert_eq!(dest.layout.fields.count(), args.len());
                            for (i, &op) in args.iter().enumerate() {
                                let dest_field = self.place_field(dest, i as u64)?;
                                self.copy_op(op, dest_field)?;
                            }
                        }

                        // that should be it
                        assert!(arg_locals.next().is_none());
                    }
                    _ => {
                        // overloaded-calls-simple.rs in miri's test suite demomstrates that there is
                        // no way to predict, from the ABI and instance.def, whether the function
                        // wants arguments passed with untupling or not.  So we just make it
                        // depend on the number of arguments...
                        let untuple =
                            sig.abi == Abi::RustCall && !args.is_empty() && args.len() != mir.arg_count;
                        let (normal_args, untuple_arg) = if untuple {
                            let (tup, args) = args.split_last().unwrap();
                            trace!("eval_fn_call: Will pass last argument by untupling");
                            (args, Some(tup))
                        } else {
                            (&args[..], None)
                        };

                        // Pass the arguments.
                        let mut arg_locals = mir.args_iter();
                        // First the normal ones.
                        for &op in normal_args {
                            let arg_local = arg_locals.next().unwrap();
                            let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                            self.copy_op(op, dest)?;
                        }
                        // The the ones to untuple.
                        if let Some(&untuple_arg) = untuple_arg {
                            for i in 0..untuple_arg.layout.fields.count() {
                                let arg_local = arg_locals.next().unwrap();
                                let dest = self.eval_place(&mir::Place::Local(arg_local))?;
                                let op = self.operand_field(untuple_arg, i as u64)?;
                                self.copy_op(op, dest)?;
                            }
                        }
                        // That should be it.
                        assert!(arg_locals.next().is_none());
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
                self.eval_fn_call(instance, &args, dest, ret, span, sig)
            }
        }
    }

    fn drop_in_place(
        &mut self,
        place: PlaceTy<'tcx>,
        instance: ty::Instance<'tcx>,
        span: Span,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        trace!("drop_in_place: {:?},\n  {:?}, {:?}", *place, place.layout.ty, instance);
        // We take the address of the object.  This may well be unaligned, which is fine
        // for us here.  However, unaligned accesses will probably make the actual drop
        // implementation fail -- a problem shared by rustc.
        let place = self.force_allocation(place)?;

        let (instance, place) = match place.layout.ty.sty {
            ty::Dynamic(..) => {
                // Dropping a trait object.
                let vtable = match place.extra {
                    PlaceExtra::Vtable(vtable) => vtable,
                    _ => bug!("Expected vtable when dropping {:#?}", place),
                };
                let place = self.unpack_unsized_mplace(place)?;
                let instance = self.read_drop_type_from_vtable(vtable)?;
                (instance, place)
            }
            _ => (instance, place),
        };

        let arg = OpTy {
            op: Operand::Immediate(place.to_ref(&self)),
            layout: self.layout_of(self.tcx.mk_mut_ptr(place.layout.ty))?,
        };

        let ty = self.tcx.mk_tup((&[] as &[ty::Ty<'tcx>]).iter()); // return type is ()
        let dest = PlaceTy::null(&self, self.layout_of(ty)?);

        self.eval_fn_call(
            instance,
            &[arg],
            Some(dest),
            Some(target),
            span,
            None,
        )
    }
}
