use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc::ty::{self, TypeVariants, Ty};
use rustc::ty::layout::Layout;
use syntax::codemap::Span;
use syntax::attr;
use syntax::abi::Abi;

use error::{EvalError, EvalResult};
use eval_context::{EvalContext, IntegerExt, StackPopCleanup, is_inhabited};
use lvalue::Lvalue;
use memory::{MemoryPointer, TlsKey};
use value::PrimVal;
use value::Value;
use rustc_data_structures::indexed_vec::Idx;

mod drop;
mod intrinsic;

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(super) fn goto_block(&mut self, target: mir::BasicBlock) {
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
                if self.const_env() {
                    return Err(EvalError::NeedsRfc("branching (if, match, loop, ...)".to_string()));
                }
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
                    ty::TyFnDef(def_id, substs) => (::eval_context::resolve(self.tcx, def_id, substs), func_ty.fn_sig(self.tcx)),
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
                if self.const_env() {
                    return Err(EvalError::NeedsRfc("invoking `Drop::drop`".to_string()));
                }
                let lval = self.eval_lvalue(location)?;
                let ty = self.lvalue_ty(location);
                self.goto_block(target);
                let ty = ::eval_context::apply_param_substs(self.tcx, self.substs(), &ty);

                let instance = ::eval_context::resolve_drop_in_place(self.tcx, ty);
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
                if !is_inhabited(self.tcx, ty) {
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
                                    Value::ByRef(ptr) => {
                                        for ((offset, ty), arg_local) in offsets.zip(fields).zip(arg_locals) {
                                            let arg = Value::ByRef(ptr.offset(offset, self.memory.layout)?);
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
                let (_, vtable) = self.eval_operand(&arg_operands[0])?.expect_ptr_vtable_pair(&self.memory)?;
                let fn_ptr = self.memory.read_ptr(vtable.offset(ptr_size * (idx as u64 + 3), self.memory.layout)?)?;
                let instance = self.memory.get_fn(fn_ptr.to_ptr()?)?;
                let mut arg_operands = arg_operands.to_vec();
                let ty = self.operand_ty(&arg_operands[0]);
                let ty = self.get_field_ty(ty, 0)?;
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
                if self.const_env() {
                    return Err(EvalError::NeedsRfc(format!("calling extern function `{}`", path)));
                }
                self.call_missing_fn(instance, destination, arg_operands, sig, path)?;
                return Ok(true);
            },
            Err(other) => return Err(other),
        };

        if self.const_env() && !self.tcx.is_const_fn(instance.def_id()) {
            return Err(EvalError::NotConst(format!("calling non-const fn `{}`", instance)));
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
        trace!("read_discriminant_value {:#?}", adt_layout);

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
                let (offset, ty) = self.nonnull_offset_and_ty(adt_ty, nndiscr, discrfield)?;
                let nonnull = adt_ptr.offset(offset.bytes(), self.memory.layout)?;
                trace!("struct wrapped nullable pointer type: {}", ty);
                // only the pointer part of a fat pointer is used for this space optimization
                let discr_size = self.type_size(ty)?.expect("bad StructWrappedNullablePointer discrfield");
                self.read_nonnull_discriminant_value(nonnull, nndiscr as u128, discr_size)?
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
    
    /// Returns Ok() when the function was handled, fail otherwise
    fn call_missing_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        arg_operands: &[mir::Operand<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx> {
        // In some cases in non-MIR libstd-mode, not having a destination is legit.  Handle these early.
        match &path[..] {
            "std::panicking::rust_panic_with_hook" |
            "std::rt::begin_panic_fmt" => return Err(EvalError::Panic),
            _ => {},
        }

        let dest_ty = sig.output();
        let (dest, dest_block) = destination.ok_or_else(|| EvalError::NoMirFor(path.clone()))?;

        if sig.abi == Abi::C {
            // An external C function
            // TODO: That functions actually has a similar preamble to what follows here.  May make sense to
            // unify these two mechanisms for "hooking into missing functions".
            self.call_c_abi(instance.def_id(), arg_operands, dest, dest_ty, dest_block)?;
            return Ok(());
        }

        let args_res: EvalResult<Vec<Value>> = arg_operands.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args = args_res?;

        let usize = self.tcx.types.usize;
    
        match &path[..] {
            // Allocators are magic.  They have no MIR, even when the rest of libstd does.
            "alloc::heap::::__rust_alloc" => {
                let size = self.value_to_primval(args[0], usize)?.to_u64()?;
                let align = self.value_to_primval(args[1], usize)?.to_u64()?;
                if size == 0 {
                    return Err(EvalError::HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(size, align)?;
                self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
            }
            "alloc::heap::::__rust_alloc_zeroed" => {
                let size = self.value_to_primval(args[0], usize)?.to_u64()?;
                let align = self.value_to_primval(args[1], usize)?.to_u64()?;
                if size == 0 {
                    return Err(EvalError::HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(size, align)?;
                self.memory.write_repeat(PrimVal::Ptr(ptr), 0, size)?;
                self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
            }
            "alloc::heap::::__rust_dealloc" => {
                let ptr = args[0].read_ptr(&self.memory)?.to_ptr()?;
                let old_size = self.value_to_primval(args[1], usize)?.to_u64()?;
                let align = self.value_to_primval(args[2], usize)?.to_u64()?;
                if old_size == 0 {
                    return Err(EvalError::HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(align));
                }
                self.memory.deallocate(ptr, Some((old_size, align)))?;
            }
            "alloc::heap::::__rust_realloc" => {
                let ptr = args[0].read_ptr(&self.memory)?.to_ptr()?;
                let old_size = self.value_to_primval(args[1], usize)?.to_u64()?;
                let old_align = self.value_to_primval(args[2], usize)?.to_u64()?;
                let new_size = self.value_to_primval(args[3], usize)?.to_u64()?;
                let new_align = self.value_to_primval(args[4], usize)?.to_u64()?;
                if old_size == 0 || new_size == 0 {
                    return Err(EvalError::HeapAllocZeroBytes);
                }
                if !old_align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(old_align));
                }
                if !new_align.is_power_of_two() {
                    return Err(EvalError::HeapAllocNonPowerOfTwoAlignment(new_align));
                }
                let new_ptr = self.memory.reallocate(ptr, old_size, old_align, new_size, new_align)?;
                self.write_primval(dest, PrimVal::Ptr(new_ptr), dest_ty)?;
            }

            // A Rust function is missing, which means we are running with MIR missing for libstd (or other dependencies).
            // Still, we can make many things mostly work by "emulating" or ignoring some functions.
            "std::io::_print" => {
                trace!("Ignoring output.  To run programs that print, make sure you have a libstd with full MIR.");
            }
            "std::thread::Builder::new" => return Err(EvalError::Unimplemented("miri does not support threading".to_owned())),
            "std::env::args" => return Err(EvalError::Unimplemented("miri does not support program arguments".to_owned())),
            "std::panicking::panicking" |
            "std::rt::panicking" => {
                // we abort on panic -> `std::rt::panicking` always returns false
                let bool = self.tcx.types.bool;
                self.write_primval(dest, PrimVal::from_bool(false), bool)?;
            }
            _ => return Err(EvalError::NoMirFor(path)),
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        self.dump_local(dest);
        self.goto_block(dest_block);
        return Ok(());
    }

    fn call_c_abi(
        &mut self,
        def_id: DefId,
        arg_operands: &[mir::Operand<'tcx>],
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
        dest_block: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let name = self.tcx.item_name(def_id);
        let attrs = self.tcx.get_attrs(def_id);
        let link_name = attr::first_attr_value_str_by_name(&attrs, "link_name")
            .unwrap_or(name)
            .as_str();

        let args_res: EvalResult<Vec<Value>> = arg_operands.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args = args_res?;

        let usize = self.tcx.types.usize;

        match &link_name[..] {
            "malloc" => {
                let size = self.value_to_primval(args[0], usize)?.to_u64()?;
                if size == 0 {
                    self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
                } else {
                    let align = self.memory.pointer_size();
                    let ptr = self.memory.allocate(size, align)?;
                    self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
                }
            }

            "free" => {
                let ptr = args[0].read_ptr(&self.memory)?;
                if !ptr.is_null()? {
                    self.memory.deallocate(ptr.to_ptr()?, None)?;
                }
            }

            "syscall" => {
                match self.value_to_primval(args[0], usize)?.to_u64()? {
                    511 => return Err(EvalError::Unimplemented("miri does not support random number generators".to_owned())),
                    id => return Err(EvalError::Unimplemented(format!("miri does not support syscall id {}", id))),
                }
            }

            "dlsym" => {
                let _handle = args[0].read_ptr(&self.memory)?;
                let symbol = args[1].read_ptr(&self.memory)?.to_ptr()?;
                let symbol_name = self.memory.read_c_str(symbol)?;
                let err = format!("bad c unicode symbol: {:?}", symbol_name);
                let symbol_name = ::std::str::from_utf8(symbol_name).unwrap_or(&err);
                return Err(EvalError::Unimplemented(format!("miri does not support dynamically loading libraries (requested symbol: {})", symbol_name)));
            }

            "__rust_maybe_catch_panic" => {
                // fn __rust_maybe_catch_panic(f: fn(*mut u8), data: *mut u8, data_ptr: *mut usize, vtable_ptr: *mut usize) -> u32
                // We abort on panic, so not much is going on here, but we still have to call the closure
                let u8_ptr_ty = self.tcx.mk_mut_ptr(self.tcx.types.u8);
                let f = args[0].read_ptr(&self.memory)?.to_ptr()?;
                let data = args[1].read_ptr(&self.memory)?;
                let f_instance = self.memory.get_fn(f)?;
                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;

                // Now we make a function call.  TODO: Consider making this re-usable?  EvalContext::step does sth. similar for the TLS dtors,
                // and of course eval_main.
                let mir = self.load_mir(f_instance.def)?;
                self.push_stack_frame(
                    f_instance,
                    mir.span,
                    mir,
                    Lvalue::undef(),
                    StackPopCleanup::Goto(dest_block),
                )?;

                let arg_local = self.frame().mir.args_iter().next().ok_or(EvalError::AbiViolation("Argument to __rust_maybe_catch_panic does not take enough arguments.".to_owned()))?;
                let arg_dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                self.write_primval(arg_dest, data, u8_ptr_ty)?;

                // We ourselves return 0
                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;

                // Don't fall through
                return Ok(());
            }

            "__rust_start_panic" => {
                return Err(EvalError::Panic);
            }

            "memcmp" => {
                let left = args[0].read_ptr(&self.memory)?;
                let right = args[1].read_ptr(&self.memory)?;
                let n = self.value_to_primval(args[2], usize)?.to_u64()?;

                let result = {
                    let left_bytes = self.memory.read_bytes(left, n)?;
                    let right_bytes = self.memory.read_bytes(right, n)?;

                    use std::cmp::Ordering::*;
                    match left_bytes.cmp(right_bytes) {
                        Less => -1i8,
                        Equal => 0,
                        Greater => 1,
                    }
                };

                self.write_primval(dest, PrimVal::Bytes(result as u128), dest_ty)?;
            }

            "memrchr" => {
                let ptr = args[0].read_ptr(&self.memory)?;
                let val = self.value_to_primval(args[1], usize)?.to_u64()? as u8;
                let num = self.value_to_primval(args[2], usize)?.to_u64()?;
                if let Some(idx) = self.memory.read_bytes(ptr, num)?.iter().rev().position(|&c| c == val) {
                    let new_ptr = ptr.offset(num - idx as u64 - 1, self.memory.layout)?;
                    self.write_primval(dest, new_ptr, dest_ty)?;
                } else {
                    self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
                }
            }

            "memchr" => {
                let ptr = args[0].read_ptr(&self.memory)?;
                let val = self.value_to_primval(args[1], usize)?.to_u64()? as u8;
                let num = self.value_to_primval(args[2], usize)?.to_u64()?;
                if let Some(idx) = self.memory.read_bytes(ptr, num)?.iter().position(|&c| c == val) {
                    let new_ptr = ptr.offset(idx as u64, self.memory.layout)?;
                    self.write_primval(dest, new_ptr, dest_ty)?;
                } else {
                    self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
                }
            }

            "getenv" => {
                let result = {
                    let name_ptr = args[0].read_ptr(&self.memory)?.to_ptr()?;
                    let name = self.memory.read_c_str(name_ptr)?;
                    match self.env_vars.get(name) {
                        Some(&var) => PrimVal::Ptr(var),
                        None => PrimVal::Bytes(0),
                    }
                };
                self.write_primval(dest, result, dest_ty)?;
            }

            "unsetenv" => {
                let mut success = None;
                {
                    let name_ptr = args[0].read_ptr(&self.memory)?;
                    if !name_ptr.is_null()? {
                        let name = self.memory.read_c_str(name_ptr.to_ptr()?)?;
                        if !name.is_empty() && !name.contains(&b'=') {
                            success = Some(self.env_vars.remove(name));
                        }
                    }
                }
                if let Some(old) = success {
                    if let Some(var) = old {
                        self.memory.deallocate(var, None)?;
                    }
                    self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
                } else {
                    self.write_primval(dest, PrimVal::from_i128(-1), dest_ty)?;
                }
            }

            "setenv" => {
                let mut new = None;
                {
                    let name_ptr = args[0].read_ptr(&self.memory)?;
                    let value_ptr = args[1].read_ptr(&self.memory)?.to_ptr()?;
                    let value = self.memory.read_c_str(value_ptr)?;
                    if !name_ptr.is_null()? {
                        let name = self.memory.read_c_str(name_ptr.to_ptr()?)?;
                        if !name.is_empty() && !name.contains(&b'=') {
                            new = Some((name.to_owned(), value.to_owned()));
                        }
                    }
                }
                if let Some((name, value)) = new {
                    // +1 for the null terminator
                    let value_copy = self.memory.allocate((value.len() + 1) as u64, 1)?;
                    self.memory.write_bytes(PrimVal::Ptr(value_copy), &value)?;
                    self.memory.write_bytes(PrimVal::Ptr(value_copy.offset(value.len() as u64, self.memory.layout)?), &[0])?;
                    if let Some(var) = self.env_vars.insert(name.to_owned(), value_copy) {
                        self.memory.deallocate(var, None)?;
                    }
                    self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
                } else {
                    self.write_primval(dest, PrimVal::from_i128(-1), dest_ty)?;
                }
            }

            "write" => {
                let fd = self.value_to_primval(args[0], usize)?.to_u64()?;
                let buf = args[1].read_ptr(&self.memory)?;
                let n = self.value_to_primval(args[2], usize)?.to_u64()?;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, n);
                let result = if fd == 1 || fd == 2 { // stdout/stderr
                    use std::io::{self, Write};
                
                    let buf_cont = self.memory.read_bytes(buf, n)?;
                    let res = if fd == 1 { io::stdout().write(buf_cont) } else { io::stderr().write(buf_cont) };
                    match res { Ok(n) => n as isize, Err(_) => -1 }
                } else {
                    info!("Ignored output to FD {}", fd);
                    n as isize // pretend it all went well
                }; // now result is the value we return back to the program
                self.write_primval(dest, PrimVal::Bytes(result as u128), dest_ty)?;
            }

            "strlen" => {
                let ptr = args[0].read_ptr(&self.memory)?.to_ptr()?;
                let n = self.memory.read_c_str(ptr)?.len();
                self.write_primval(dest, PrimVal::Bytes(n as u128), dest_ty)?;
            }

            // Some things needed for sys::thread initialization to go through
            "signal" | "sigaction" | "sigaltstack" => {
                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
            }

            "sysconf" => {
                let name = self.value_to_primval(args[0], usize)?.to_u64()?;
                trace!("sysconf() called with name {}", name);
                let result = match name {
                    30 => PrimVal::Bytes(4096), // _SC_PAGESIZE
                    70 => PrimVal::from_i128(-1), // _SC_GETPW_R_SIZE_MAX
                    _ => return Err(EvalError::Unimplemented(format!("Unimplemented sysconf name: {}", name)))
                };
                self.write_primval(dest, result, dest_ty)?;
            }

            "mmap" => {
                // This is a horrible hack, but well... the guard page mechanism calls mmap and expects a particular return value, so we give it that value
                let addr = args[0].read_ptr(&self.memory)?;
                self.write_primval(dest, addr, dest_ty)?;
            }

            // Hook pthread calls that go to the thread-local storage memory subsystem
            "pthread_key_create" => {
                let key_ptr = args[0].read_ptr(&self.memory)?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves...)
                let dtor = match args[1].read_ptr(&self.memory)? {
                    PrimVal::Ptr(dtor_ptr) => Some(self.memory.get_fn(dtor_ptr)?),
                    PrimVal::Bytes(0) => None,
                    PrimVal::Bytes(_) => return Err(EvalError::ReadBytesAsPointer),
                    PrimVal::Undef => return Err(EvalError::ReadUndefBytes),
                };

                // Figure out how large a pthread TLS key actually is. This is libc::pthread_key_t.
                let key_type = self.operand_ty(&arg_operands[0]).builtin_deref(true, ty::LvaluePreference::NoPreference)
                                   .ok_or(EvalError::AbiViolation("Wrong signature used for pthread_key_create: First argument must be a raw pointer.".to_owned()))?.ty;
                let key_size = {
                    let layout = self.type_layout(key_type)?;
                    layout.size(&self.tcx.data_layout)
                };

                // Create key and write it into the memory where key_ptr wants it
                let key = self.memory.create_tls_key(dtor) as u128;
                if key_size.bits() < 128 && key >= (1u128 << key_size.bits() as u128) {
                    return Err(EvalError::OutOfTls);
                }
                // TODO: Does this need checking for alignment?
                self.memory.write_uint(key_ptr.to_ptr()?, key, key_size.bytes())?;

                // Return success (0)
                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
            }
            "pthread_key_delete" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0], usize)?.to_u64()? as TlsKey;
                self.memory.delete_tls_key(key)?;
                // Return success (0)
                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
            }
            "pthread_getspecific" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0], usize)?.to_u64()? as TlsKey;
                let ptr = self.memory.load_tls(key)?;
                self.write_primval(dest, ptr, dest_ty)?;
            }
            "pthread_setspecific" => {
                // The conversion into TlsKey here is a little fishy, but should work as long as usize >= libc::pthread_key_t
                let key = self.value_to_primval(args[0], usize)?.to_u64()? as TlsKey;
                let new_ptr = args[1].read_ptr(&self.memory)?;
                self.memory.store_tls(key, new_ptr)?;
                
                // Return success (0)
                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
            }

            // Stub out all the other pthread calls to just return 0
            link_name if link_name.starts_with("pthread_") => {
                warn!("ignoring C ABI call: {}", link_name);
                self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
            },

            _ => {
                return Err(EvalError::Unimplemented(format!("can't call C ABI function: {}", link_name)));
            }
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        self.dump_local(dest);
        self.goto_block(dest_block);
        Ok(())
    }
}
