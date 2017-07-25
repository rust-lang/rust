use rustc::mir;
use rustc::ty::{self, TypeVariants, Ty};
use rustc::ty::layout::Layout;
use syntax::codemap::Span;
use syntax::abi::Abi;

use super::{
    EvalError, EvalResult,
    EvalContext, StackPopCleanup, eval_context, TyAndPacked,
    Lvalue,
    MemoryPointer,
    PrimVal, Value,
    Machine,
    HasMemory,
};
use super::eval_context::IntegerExt;

use rustc_data_structures::indexed_vec::Idx;

mod drop;
mod intrinsic;

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

            SwitchInt { ref discr, ref values, ref targets, .. } => {
                // FIXME(CTFE): forbid branching
                let discr_val = self.eval_operand(discr)?;
                let discr_ty = self.operand_ty(discr);
                let discr_prim = self.value_to_primval(discr_val, discr_ty)?;

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

            Call { ref func, ref args, ref destination, .. } => {
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
                                let sig = self.erase_lifetimes(&sig);
                                let real_sig = self.erase_lifetimes(&real_sig);
                                let real_sig = self.tcx.normalize_associated_type(&real_sig);
                                if !self.check_sig_compat(sig, real_sig)? {
                                    return Err(EvalError::FunctionPointerTyMismatch(real_sig, sig));
                                }
                            },
                            ref other => bug!("instance def ty: {:?}", other),
                        }
                        (instance, sig)
                    },
                    ty::TyFnDef(def_id, substs) => (eval_context::resolve(self.tcx, def_id, substs), func_ty.fn_sig(self.tcx)),
                    _ => {
                        let msg = format!("can't handle callee of type {:?}", func_ty);
                        return Err(EvalError::Unimplemented(msg));
                    }
                };
                let sig = self.erase_lifetimes(&sig);
                self.eval_fn_call(fn_def, destination, args, terminator.source_info.span, sig)?;
            }

            Drop { ref location, target, .. } => {
                trace!("TerminatorKind::drop: {:?}, {:?}", location, self.substs());
                // FIXME(CTFE): forbid drop in const eval
                let lval = self.eval_lvalue(location)?;
                let ty = self.lvalue_ty(location);
                self.goto_block(target);
                let ty = eval_context::apply_param_substs(self.tcx, self.substs(), &ty);

                let instance = eval_context::resolve_drop_in_place(self.tcx, ty);
                self.drop_lvalue(lval, instance, ty, terminator.source_info.span)?;
            }

            Assert { ref cond, expected, ref msg, target, .. } => {
                let cond_val = self.eval_operand_to_primval(cond)?.to_bool()?;
                if expected == cond_val {
                    self.goto_block(target);
                } else {
                    return match *msg {
                        mir::AssertMessage::BoundsCheck { ref len, ref index } => {
                            let span = terminator.source_info.span;
                            let len = self.eval_operand_to_primval(len)
                                .expect("can't eval len")
                                .to_u64()?;
                            let index = self.eval_operand_to_primval(index)
                                .expect("can't eval index")
                                .to_u64()?;
                            Err(EvalError::ArrayIndexOutOfBounds(span, len, index))
                        },
                        mir::AssertMessage::Math(ref err) =>
                            Err(EvalError::Math(terminator.source_info.span, err.clone())),
                    }
                }
            },

            DropAndReplace { .. } => unimplemented!(),
            Resume => unimplemented!(),
            Unreachable => return Err(EvalError::Unreachable),
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
        fn check_ty_compat<'tcx>(
            ty: ty::Ty<'tcx>,
            real_ty: ty::Ty<'tcx>,
        ) -> bool {
            if ty == real_ty { return true; } // This is actually a fast pointer comparison
            return match (&ty.sty, &real_ty.sty) {
                // Permit changing the pointer type of raw pointers and references as well as
                // mutability of raw pointers.
                // TODO: Should not be allowed when fat pointers are involved.
                (&TypeVariants::TyRawPtr(_), &TypeVariants::TyRawPtr(_)) => true,
                (&TypeVariants::TyRef(_, _), &TypeVariants::TyRef(_, _)) =>
                    ty.is_mutable_pointer() == real_ty.is_mutable_pointer(),
                // rule out everything else
                _ => false
            }
        }

        if sig.abi == real_sig.abi &&
            sig.variadic == real_sig.variadic &&
            sig.inputs_and_output.len() == real_sig.inputs_and_output.len() &&
            sig.inputs_and_output.iter().zip(real_sig.inputs_and_output).all(|(ty, real_ty)| check_ty_compat(ty, real_ty)) {
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
        destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        arg_operands: &[mir::Operand<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx> {
        trace!("eval_fn_call: {:#?}", instance);
        match instance.def {
            ty::InstanceDef::Intrinsic(..) => {
                let (ret, target) = match destination {
                    Some(dest) => dest,
                    _ => return Err(EvalError::Unreachable),
                };
                let ty = sig.output();
                if !eval_context::is_inhabited(self.tcx, ty) {
                    return Err(EvalError::Unreachable);
                }
                let layout = self.type_layout(ty)?;
                self.call_intrinsic(instance, arg_operands, ret, ty, layout, target)?;
                self.dump_local(ret);
                Ok(())
            },
            ty::InstanceDef::ClosureOnceShim{..} => {
                let mut args = Vec::new();
                for arg in arg_operands {
                    let arg_val = self.eval_operand(arg)?;
                    let arg_ty = self.operand_ty(arg);
                    args.push((arg_val, arg_ty));
                }
                if self.eval_fn_call_inner(
                    instance,
                    destination,
                    arg_operands,
                    span,
                    sig,
                )? {
                    return Ok(());
                }
                let mut arg_locals = self.frame().mir.args_iter();
                match sig.abi {
                    // closure as closure once
                    Abi::RustCall => {
                        for (arg_local, (arg_val, arg_ty)) in arg_locals.zip(args) {
                            let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                            self.write_value(arg_val, dest, arg_ty)?;
                        }
                    },
                    // non capture closure as fn ptr
                    // need to inject zst ptr for closure object (aka do nothing)
                    // and need to pack arguments
                    Abi::Rust => {
                        trace!("arg_locals: {:?}", self.frame().mir.args_iter().collect::<Vec<_>>());
                        trace!("arg_operands: {:?}", arg_operands);
                        let local = arg_locals.nth(1).unwrap();
                        for (i, (arg_val, arg_ty)) in args.into_iter().enumerate() {
                            let dest = self.eval_lvalue(&mir::Lvalue::Local(local).field(mir::Field::new(i), arg_ty))?;
                            self.write_value(arg_val, dest, arg_ty)?;
                        }
                    },
                    _ => bug!("bad ABI for ClosureOnceShim: {:?}", sig.abi),
                }
                Ok(())
            }
            ty::InstanceDef::Item(_) => {
                let mut args = Vec::new();
                for arg in arg_operands {
                    let arg_val = self.eval_operand(arg)?;
                    let arg_ty = self.operand_ty(arg);
                    args.push((arg_val, arg_ty));
                }

                // Push the stack frame, and potentially be entirely done if the call got hooked
                if self.eval_fn_call_inner(
                    instance,
                    destination,
                    arg_operands,
                    span,
                    sig,
                )? {
                    return Ok(());
                }

                // Pass the arguments
                let mut arg_locals = self.frame().mir.args_iter();
                trace!("ABI: {:?}", sig.abi);
                trace!("arg_locals: {:?}", self.frame().mir.args_iter().collect::<Vec<_>>());
                trace!("arg_operands: {:?}", arg_operands);
                match sig.abi {
                    Abi::RustCall => {
                        assert_eq!(args.len(), 2);

                        {   // write first argument
                            let first_local = arg_locals.next().unwrap();
                            let dest = self.eval_lvalue(&mir::Lvalue::Local(first_local))?;
                            let (arg_val, arg_ty) = args.remove(0);
                            self.write_value(arg_val, dest, arg_ty)?;
                        }

                        // unpack and write all other args
                        let (arg_val, arg_ty) = args.remove(0);
                        let layout = self.type_layout(arg_ty)?;
                        if let (&ty::TyTuple(fields, _), &Layout::Univariant { ref variant, .. }) = (&arg_ty.sty, layout) {
                            trace!("fields: {:?}", fields);
                            if self.frame().mir.args_iter().count() == fields.len() + 1 {
                                let offsets = variant.offsets.iter().map(|s| s.bytes());
                                match arg_val {
                                    Value::ByRef { ptr, aligned } => {
                                        assert!(aligned, "Unaligned ByRef-values cannot occur as function arguments");
                                        for ((offset, ty), arg_local) in offsets.zip(fields).zip(arg_locals) {
                                            let arg = Value::ByRef { ptr: ptr.offset(offset, &self)?, aligned: true};
                                            let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                                            trace!("writing arg {:?} to {:?} (type: {})", arg, dest, ty);
                                            self.write_value(arg, dest, ty)?;
                                        }
                                    },
                                    Value::ByVal(PrimVal::Undef) => {},
                                    other => {
                                        assert_eq!(fields.len(), 1);
                                        let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_locals.next().unwrap()))?;
                                        self.write_value(other, dest, fields[0])?;
                                    }
                                }
                            } else {
                                trace!("manual impl of rust-call ABI");
                                // called a manual impl of a rust-call function
                                let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_locals.next().unwrap()))?;
                                self.write_value(arg_val, dest, arg_ty)?;
                            }
                        } else {
                            bug!("rust-call ABI tuple argument was {:?}, {:?}", arg_ty, layout);
                        }
                    },
                    _ => {
                        for (arg_local, (arg_val, arg_ty)) in arg_locals.zip(args) {
                            let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                            self.write_value(arg_val, dest, arg_ty)?;
                        }
                    }
                }
                Ok(())
            },
            ty::InstanceDef::DropGlue(..) => {
                assert_eq!(arg_operands.len(), 1);
                assert_eq!(sig.abi, Abi::Rust);
                let val = self.eval_operand(&arg_operands[0])?;
                let ty = self.operand_ty(&arg_operands[0]);
                let (_, target) = destination.expect("diverging drop glue");
                self.goto_block(target);
                // FIXME: deduplicate these matches
                let pointee_type = match ty.sty {
                    ty::TyRawPtr(ref tam) |
                    ty::TyRef(_, ref tam) => tam.ty,
                    ty::TyAdt(def, _) if def.is_box() => ty.boxed_ty(),
                    _ => bug!("can only deref pointer types"),
                };
                self.drop(val, instance, pointee_type, span)
            },
            ty::InstanceDef::FnPtrShim(..) => {
                trace!("ABI: {}", sig.abi);
                let mut args = Vec::new();
                for arg in arg_operands {
                    let arg_val = self.eval_operand(arg)?;
                    let arg_ty = self.operand_ty(arg);
                    args.push((arg_val, arg_ty));
                }
                if self.eval_fn_call_inner(
                    instance,
                    destination,
                    arg_operands,
                    span,
                    sig,
                )? {
                    return Ok(());
                }
                let arg_locals = self.frame().mir.args_iter();
                match sig.abi {
                    Abi::Rust => {
                        args.remove(0);
                    },
                    Abi::RustCall => {},
                    _ => unimplemented!(),
                };
                for (arg_local, (arg_val, arg_ty)) in arg_locals.zip(args) {
                    let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                    self.write_value(arg_val, dest, arg_ty)?;
                }
                Ok(())
            },
            ty::InstanceDef::Virtual(_, idx) => {
                let ptr_size = self.memory.pointer_size();
                let (_, vtable) = self.eval_operand(&arg_operands[0])?.into_ptr_vtable_pair(&self.memory)?;
                let fn_ptr = self.memory.read_ptr(vtable.offset(ptr_size * (idx as u64 + 3), &self)?)?;
                let instance = self.memory.get_fn(fn_ptr.to_ptr()?)?;
                let mut arg_operands = arg_operands.to_vec();
                let ty = self.operand_ty(&arg_operands[0]);
                let ty = self.get_field_ty(ty, 0)?.ty; // TODO: packed flag is ignored
                match arg_operands[0] {
                    mir::Operand::Consume(ref mut lval) => *lval = lval.clone().field(mir::Field::new(0), ty),
                    _ => bug!("virtual call first arg cannot be a constant"),
                }
                // recurse with concrete function
                self.eval_fn_call(
                    instance,
                    destination,
                    &arg_operands,
                    span,
                    sig,
                )
            },
        }
    }

    /// Returns Ok(true) when the function was handled completely due to mir not being available
    fn eval_fn_call_inner(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        arg_operands: &[mir::Operand<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        trace!("eval_fn_call_inner: {:#?}, {:#?}", instance, destination);

        // Only trait methods can have a Self parameter.

        let mir = match self.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(EvalError::NoMirFor(path)) => {
                M::call_missing_fn(self, instance, destination, arg_operands, sig, path)?;
                return Ok(true);
            },
            Err(other) => return Err(other),
        };

        if !self.tcx.is_const_fn(instance.def_id()) {
            M::check_non_const_fn_call(instance)?;
        }
        
        let (return_lvalue, return_to_block) = match destination {
            Some((lvalue, block)) => (lvalue, StackPopCleanup::Goto(block)),
            None => (Lvalue::undef(), StackPopCleanup::None),
        };

        self.push_stack_frame(
            instance,
            span,
            mir,
            return_lvalue,
            return_to_block,
        )?;

        Ok(false)
    }

    pub fn read_discriminant_value(&self, adt_ptr: MemoryPointer, adt_ty: Ty<'tcx>) -> EvalResult<'tcx, u128> {
        use rustc::ty::layout::Layout::*;
        let adt_layout = self.type_layout(adt_ty)?;
        //trace!("read_discriminant_value {:#?}", adt_layout);

        let discr_val = match *adt_layout {
            General { discr, .. } | CEnum { discr, signed: false, .. } => {
                let discr_size = discr.size().bytes();
                self.memory.read_uint(adt_ptr, discr_size)?
            }

            CEnum { discr, signed: true, .. } => {
                let discr_size = discr.size().bytes();
                self.memory.read_int(adt_ptr, discr_size)? as u128
            }

            RawNullablePointer { nndiscr, value } => {
                let discr_size = value.size(&self.tcx.data_layout).bytes();
                trace!("rawnullablepointer with size {}", discr_size);
                self.read_nonnull_discriminant_value(adt_ptr, nndiscr as u128, discr_size)?
            }

            StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
                let (offset, TyAndPacked { ty, packed }) = self.nonnull_offset_and_ty(adt_ty, nndiscr, discrfield)?;
                let nonnull = adt_ptr.offset(offset.bytes(), &*self)?;
                trace!("struct wrapped nullable pointer type: {}", ty);
                // only the pointer part of a fat pointer is used for this space optimization
                let discr_size = self.type_size(ty)?.expect("bad StructWrappedNullablePointer discrfield");
                self.read_maybe_aligned(!packed,
                    |ectx| ectx.read_nonnull_discriminant_value(nonnull, nndiscr as u128, discr_size))?
            }

            // The discriminant_value intrinsic returns 0 for non-sum types.
            Array { .. } | FatPointer { .. } | Scalar { .. } | Univariant { .. } |
            Vector { .. } | UntaggedUnion { .. } => 0,
        };

        Ok(discr_val)
    }

    fn read_nonnull_discriminant_value(&self, ptr: MemoryPointer, nndiscr: u128, discr_size: u64) -> EvalResult<'tcx, u128> {
        trace!("read_nonnull_discriminant_value: {:?}, {}, {}", ptr, nndiscr, discr_size);
        let not_null = match self.memory.read_uint(ptr, discr_size) {
            Ok(0) => false,
            Ok(_) | Err(EvalError::ReadPointerAsBytes) => true,
            Err(e) => return Err(e),
        };
        assert!(nndiscr == 0 || nndiscr == 1);
        Ok(if not_null { nndiscr } else { 1 - nndiscr })
    }
}
