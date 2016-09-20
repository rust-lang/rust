use rustc::hir::def_id::DefId;
use rustc::middle::const_val::ConstVal;
use rustc::mir::repr as mir;
use rustc::traits::{self, Reveal};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::layout::Layout;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt, BareFnTy};
use std::iter;
use std::rc::Rc;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::{ast, attr};

use error::{EvalError, EvalResult};
use memory::Pointer;
use super::{EvalContext, IntegerExt, StackPopCleanup};

impl<'a, 'tcx> EvalContext<'a, 'tcx> {

    pub(super) fn goto_block(&mut self, target: mir::BasicBlock) {
        self.frame_mut().block = target;
        self.frame_mut().stmt = 0;
    }

    pub(super) fn eval_terminator(
        &mut self,
        terminator: &mir::Terminator<'tcx>,
    ) -> EvalResult<'tcx, ()> {
        use rustc::mir::repr::TerminatorKind::*;
        match terminator.kind {
            Return => self.pop_stack_frame()?,

            Goto { target } => self.goto_block(target),

            If { ref cond, targets: (then_target, else_target) } => {
                let cond_val = self.eval_operand_to_primval(cond)?
                    .expect_bool("TerminatorKind::If condition constant was not a bool");
                self.goto_block(if cond_val { then_target } else { else_target });
            }

            SwitchInt { ref discr, ref values, ref targets, .. } => {
                let discr_ptr = self.eval_lvalue(discr)?.to_ptr();
                let discr_ty = self.lvalue_ty(discr);
                let discr_size = self
                    .type_layout(discr_ty)
                    .size(&self.tcx.data_layout)
                    .bytes() as usize;
                let discr_val = self.memory.read_uint(discr_ptr, discr_size)?;
                if let ty::TyChar = discr_ty.sty {
                    if ::std::char::from_u32(discr_val as u32).is_none() {
                        return Err(EvalError::InvalidChar(discr_val as u64));
                    }
                }

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets[targets.len() - 1];

                for (index, const_val) in values.iter().enumerate() {
                    let val = match const_val {
                        &ConstVal::Integral(i) => i.to_u64_unchecked(),
                        _ => bug!("TerminatorKind::SwitchInt branch constant was not an integer"),
                    };
                    if discr_val == val {
                        target_block = targets[index];
                        break;
                    }
                }

                self.goto_block(target_block);
            }

            Switch { ref discr, ref targets, adt_def } => {
                let adt_ptr = self.eval_lvalue(discr)?.to_ptr();
                let adt_ty = self.lvalue_ty(discr);
                let discr_val = self.read_discriminant_value(adt_ptr, adt_ty)?;
                let matching = adt_def.variants.iter()
                    .position(|v| discr_val == v.disr_val.to_u64_unchecked());

                match matching {
                    Some(i) => self.goto_block(targets[i]),
                    None => return Err(EvalError::InvalidDiscriminant),
                }
            }

            Call { ref func, ref args, ref destination, .. } => {
                let destination = match *destination {
                    Some((ref lv, target)) => Some((self.eval_lvalue(lv)?.to_ptr(), target)),
                    None => None,
                };

                let func_ty = self.operand_ty(func);
                match func_ty.sty {
                    ty::TyFnPtr(bare_fn_ty) => {
                        let fn_ptr = self.eval_operand_to_primval(func)?
                            .expect_fn_ptr("TyFnPtr callee did not evaluate to PrimVal::FnPtr");
                        let (def_id, substs, fn_ty) = self.memory.get_fn(fn_ptr.alloc_id)?;
                        if fn_ty != bare_fn_ty {
                            return Err(EvalError::FunctionPointerTyMismatch(fn_ty, bare_fn_ty));
                        }
                        self.eval_fn_call(def_id, substs, bare_fn_ty, destination, args,
                                          terminator.source_info.span)?
                    },
                    ty::TyFnDef(def_id, substs, fn_ty) => {
                        self.eval_fn_call(def_id, substs, fn_ty, destination, args,
                                          terminator.source_info.span)?
                    }

                    _ => return Err(EvalError::Unimplemented(format!("can't handle callee of type {:?}", func_ty))),
                }
            }

            Drop { ref location, target, .. } => {
                let ptr = self.eval_lvalue(location)?.to_ptr();
                let ty = self.lvalue_ty(location);
                self.drop(ptr, ty)?;
                self.goto_block(target);
            }

            Assert { ref cond, expected, ref msg, target, .. } => {
                let cond_val = self.eval_operand_to_primval(cond)?
                    .expect_bool("TerminatorKind::Assert condition constant was not a bool");
                if expected == cond_val {
                    self.goto_block(target);
                } else {
                    return match *msg {
                        mir::AssertMessage::BoundsCheck { ref len, ref index } => {
                            let span = terminator.source_info.span;
                            let len = self.eval_operand_to_primval(len).expect("can't eval len")
                                .expect_uint("BoundsCheck len wasn't a uint");
                            let index = self.eval_operand_to_primval(index)
                                .expect("can't eval index")
                                .expect_uint("BoundsCheck index wasn't a uint");
                            Err(EvalError::ArrayIndexOutOfBounds(span, len, index))
                        },
                        mir::AssertMessage::Math(ref err) =>
                            Err(EvalError::Math(terminator.source_info.span, err.clone())),
                    }
                }
            },

            DropAndReplace { .. } => unimplemented!(),
            Resume => unimplemented!(),
            Unreachable => unimplemented!(),
        }

        Ok(())
    }

    fn eval_fn_call(
        &mut self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
        fn_ty: &'tcx BareFnTy,
        destination: Option<(Pointer, mir::BasicBlock)>,
        args: &[mir::Operand<'tcx>],
        span: Span,
    ) -> EvalResult<'tcx, ()> {
        use syntax::abi::Abi;
        match fn_ty.abi {
            Abi::RustIntrinsic => {
                let ty = fn_ty.sig.0.output;
                let layout = self.type_layout(ty);
                let (ret, target) = destination.unwrap();
                self.call_intrinsic(def_id, substs, args, ret, layout)?;
                self.goto_block(target);
                Ok(())
            }

            Abi::C => {
                let ty = fn_ty.sig.0.output;
                let size = self.type_size(ty);
                let (ret, target) = destination.unwrap();
                self.call_c_abi(def_id, args, ret, size)?;
                self.goto_block(target);
                Ok(())
            }

            Abi::Rust | Abi::RustCall => {
                // TODO(solson): Adjust the first argument when calling a Fn or
                // FnMut closure via FnOnce::call_once.

                let mut arg_srcs = Vec::new();
                for arg in args {
                    let src = self.eval_operand_to_ptr(arg)?;
                    let src_ty = self.operand_ty(arg);
                    arg_srcs.push((src, src_ty));
                }

                // Only trait methods can have a Self parameter.
                let (resolved_def_id, resolved_substs) =
                    if let Some(trait_id) = self.tcx.trait_of_item(def_id) {
                        self.trait_method(trait_id, def_id, substs, &mut arg_srcs)?
                    } else {
                        (def_id, substs)
                    };

                if fn_ty.abi == Abi::RustCall {
                    if let Some((last, last_ty)) = arg_srcs.pop() {
                        let last_layout = self.type_layout(last_ty);
                        match (&last_ty.sty, last_layout) {
                            (&ty::TyTuple(fields),
                             &Layout::Univariant { ref variant, .. }) => {
                                let offsets = iter::once(0)
                                    .chain(variant.offset_after_field.iter()
                                        .map(|s| s.bytes()));
                                for (offset, ty) in offsets.zip(fields) {
                                    let src = last.offset(offset as isize);
                                    arg_srcs.push((src, ty));
                                }
                            }
                            ty => bug!("expected tuple as last argument in function with 'rust-call' ABI, got {:?}", ty),
                        }
                    }
                }

                let mir = self.load_mir(resolved_def_id);
                let (return_ptr, return_to_block) = match destination {
                    Some((ptr, block)) => (Some(ptr), StackPopCleanup::Goto(block)),
                    None => (None, StackPopCleanup::None),
                };
                self.push_stack_frame(resolved_def_id, span, mir, resolved_substs, return_ptr, return_to_block)?;

                for (i, (src, src_ty)) in arg_srcs.into_iter().enumerate() {
                    let dest = self.frame().locals[i];
                    self.move_(src, dest, src_ty)?;
                }

                Ok(())
            }

            abi => Err(EvalError::Unimplemented(format!("can't handle function with {:?} ABI", abi))),
        }
    }

    fn read_discriminant_value(&self, adt_ptr: Pointer, adt_ty: Ty<'tcx>) -> EvalResult<'tcx, u64> {
        use rustc::ty::layout::Layout::*;
        let adt_layout = self.type_layout(adt_ty);

        let discr_val = match *adt_layout {
            General { discr, .. } | CEnum { discr, .. } => {
                let discr_size = discr.size().bytes();
                self.memory.read_uint(adt_ptr, discr_size as usize)?
            }

            RawNullablePointer { nndiscr, .. } => {
                self.read_nonnull_discriminant_value(adt_ptr, nndiscr)?
            }

            StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
                let offset = self.nonnull_offset(adt_ty, nndiscr, discrfield)?;
                let nonnull = adt_ptr.offset(offset.bytes() as isize);
                self.read_nonnull_discriminant_value(nonnull, nndiscr)?
            }

            // The discriminant_value intrinsic returns 0 for non-sum types.
            Array { .. } | FatPointer { .. } | Scalar { .. } | Univariant { .. } |
            Vector { .. } | UntaggedUnion { .. } => 0,
        };

        Ok(discr_val)
    }

    fn read_nonnull_discriminant_value(&self, ptr: Pointer, nndiscr: u64) -> EvalResult<'tcx, u64> {
        let not_null = match self.memory.read_usize(ptr) {
            Ok(0) => false,
            Ok(_) | Err(EvalError::ReadPointerAsBytes) => true,
            Err(e) => return Err(e),
        };
        assert!(nndiscr == 0 || nndiscr == 1);
        Ok(if not_null { nndiscr } else { 1 - nndiscr })
    }

    fn call_intrinsic(
        &mut self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
        args: &[mir::Operand<'tcx>],
        dest: Pointer,
        dest_layout: &'tcx Layout,
    ) -> EvalResult<'tcx, ()> {
        // TODO(solson): We can probably remove this _to_ptr easily.
        let args_res: EvalResult<Vec<Pointer>> = args.iter()
            .map(|arg| self.eval_operand_to_ptr(arg))
            .collect();
        let args_ptrs = args_res?;
        let pointer_size = self.memory.pointer_size();

        match &self.tcx.item_name(def_id).as_str()[..] {
            "add_with_overflow" => self.intrinsic_with_overflow(mir::BinOp::Add, &args[0], &args[1], dest, dest_layout)?,
            "sub_with_overflow" => self.intrinsic_with_overflow(mir::BinOp::Sub, &args[0], &args[1], dest, dest_layout)?,
            "mul_with_overflow" => self.intrinsic_with_overflow(mir::BinOp::Mul, &args[0], &args[1], dest, dest_layout)?,

            "assume" => {
                if !self.memory.read_bool(args_ptrs[0])? {
                    return Err(EvalError::AssumptionNotHeld);
                }
            }

            "copy_nonoverlapping" => {
                let elem_ty = substs.type_at(0);
                let elem_size = self.type_size(elem_ty);
                let elem_align = self.type_align(elem_ty);
                let src = self.memory.read_ptr(args_ptrs[0])?;
                let dest = self.memory.read_ptr(args_ptrs[1])?;
                let count = self.memory.read_isize(args_ptrs[2])?;
                self.memory.copy(src, dest, count as usize * elem_size, elem_align)?;
            }

            "ctpop" => {
                let elem_ty = substs.type_at(0);
                let elem_size = self.type_size(elem_ty);
                let num = self.memory.read_uint(args_ptrs[0], elem_size)?.count_ones();
                self.memory.write_uint(dest, num.into(), elem_size)?;
            }

            "ctlz" => {
                let elem_ty = substs.type_at(0);
                let elem_size = self.type_size(elem_ty);
                let num = self.memory.read_uint(args_ptrs[0], elem_size)?.leading_zeros();
                self.memory.write_uint(dest, num.into(), elem_size)?;
            }

            "discriminant_value" => {
                let ty = substs.type_at(0);
                let adt_ptr = self.memory.read_ptr(args_ptrs[0])?;
                let discr_val = self.read_discriminant_value(adt_ptr, ty)?;
                self.memory.write_uint(dest, discr_val, 8)?;
            }

            "forget" => {}

            "init" => self.memory.write_repeat(dest, 0, dest_layout.size(&self.tcx.data_layout).bytes() as usize)?,

            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = self.type_align(elem_ty);
                self.memory.write_uint(dest, elem_align as u64, pointer_size)?;
            }

            "move_val_init" => {
                let ty = substs.type_at(0);
                let ptr = self.memory.read_ptr(args_ptrs[0])?;
                self.move_(args_ptrs[1], ptr, ty)?;
            }

            "offset" => {
                let pointee_ty = substs.type_at(0);
                let pointee_size = self.type_size(pointee_ty) as isize;
                let ptr_arg = args_ptrs[0];
                let offset = self.memory.read_isize(args_ptrs[1])?;

                match self.memory.read_ptr(ptr_arg) {
                    Ok(ptr) => {
                        let result_ptr = ptr.offset(offset as isize * pointee_size);
                        self.memory.write_ptr(dest, result_ptr)?;
                    }
                    Err(EvalError::ReadBytesAsPointer) => {
                        let addr = self.memory.read_isize(ptr_arg)?;
                        let result_addr = addr + offset * pointee_size as i64;
                        self.memory.write_isize(dest, result_addr)?;
                    }
                    Err(e) => return Err(e),
                }
            }

            "overflowing_sub" => {
                self.intrinsic_overflowing(mir::BinOp::Sub, &args[0], &args[1], dest)?;
            }

            "overflowing_mul" => {
                self.intrinsic_overflowing(mir::BinOp::Mul, &args[0], &args[1], dest)?;
            }

            "overflowing_add" => {
                self.intrinsic_overflowing(mir::BinOp::Add, &args[0], &args[1], dest)?;
            }

            "size_of" => {
                let ty = substs.type_at(0);
                let size = self.type_size(ty) as u64;
                self.memory.write_uint(dest, size, pointer_size)?;
            }

            "size_of_val" => {
                let ty = substs.type_at(0);
                if self.type_is_sized(ty) {
                    let size = self.type_size(ty) as u64;
                    self.memory.write_uint(dest, size, pointer_size)?;
                } else {
                    match ty.sty {
                        ty::TySlice(_) | ty::TyStr => {
                            let elem_ty = ty.sequence_element_type(self.tcx);
                            let elem_size = self.type_size(elem_ty) as u64;
                            let ptr_size = self.memory.pointer_size() as isize;
                            let n = self.memory.read_usize(args_ptrs[0].offset(ptr_size))?;
                            self.memory.write_uint(dest, n * elem_size, pointer_size)?;
                        }

                        _ => return Err(EvalError::Unimplemented(format!("unimplemented: size_of_val::<{:?}>", ty))),
                    }
                }
            }

            "transmute" => {
                let ty = substs.type_at(0);
                self.move_(args_ptrs[0], dest, ty)?;
            }
            "uninit" => self.memory.mark_definedness(dest, dest_layout.size(&self.tcx.data_layout).bytes() as usize, false)?,

            name => return Err(EvalError::Unimplemented(format!("unimplemented intrinsic: {}", name))),
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }

    fn call_c_abi(
        &mut self,
        def_id: DefId,
        args: &[mir::Operand<'tcx>],
        dest: Pointer,
        dest_size: usize,
    ) -> EvalResult<'tcx, ()> {
        let name = self.tcx.item_name(def_id);
        let attrs = self.tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, "link_name") {
            Some(ln) => ln.clone(),
            None => name.as_str(),
        };

        // TODO(solson): We can probably remove this _to_ptr easily.
        let args_res: EvalResult<Vec<Pointer>> = args.iter()
            .map(|arg| self.eval_operand_to_ptr(arg))
            .collect();
        let args = args_res?;

        if link_name.starts_with("pthread_") {
            warn!("ignoring C ABI call: {}", link_name);
            return Ok(());
        }

        match &link_name[..] {
            "__rust_allocate" => {
                let size = self.memory.read_usize(args[0])?;
                let align = self.memory.read_usize(args[1])?;
                let ptr = self.memory.allocate(size as usize, align as usize)?;
                self.memory.write_ptr(dest, ptr)?;
            }

            "__rust_reallocate" => {
                let ptr = self.memory.read_ptr(args[0])?;
                let size = self.memory.read_usize(args[2])?;
                let align = self.memory.read_usize(args[3])?;
                let new_ptr = self.memory.reallocate(ptr, size as usize, align as usize)?;
                self.memory.write_ptr(dest, new_ptr)?;
            }

            "memcmp" => {
                let left = self.memory.read_ptr(args[0])?;
                let right = self.memory.read_ptr(args[1])?;
                let n = self.memory.read_usize(args[2])? as usize;

                let result = {
                    let left_bytes = self.memory.read_bytes(left, n)?;
                    let right_bytes = self.memory.read_bytes(right, n)?;

                    use std::cmp::Ordering::*;
                    match left_bytes.cmp(right_bytes) {
                        Less => -1,
                        Equal => 0,
                        Greater => 1,
                    }
                };

                self.memory.write_int(dest, result, dest_size)?;
            }

            _ => {
                return Err(EvalError::Unimplemented(format!("can't call C ABI function: {}", link_name)));
            }
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }

    pub(super) fn fulfill_obligation(&self, trait_ref: ty::PolyTraitRef<'tcx>) -> traits::Vtable<'tcx, ()> {
        // Do the initial selection for the obligation. This yields the shallow result we are
        // looking for -- that is, what specific impl.
        self.tcx.infer_ctxt(None, None, Reveal::All).enter(|infcx| {
            let mut selcx = traits::SelectionContext::new(&infcx);

            let obligation = traits::Obligation::new(
                traits::ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID),
                trait_ref.to_poly_trait_predicate(),
            );
            let selection = selcx.select(&obligation).unwrap().unwrap();

            // Currently, we use a fulfillment context to completely resolve all nested obligations.
            // This is because they can inform the inference of the impl's type parameters.
            let mut fulfill_cx = traits::FulfillmentContext::new();
            let vtable = selection.map(|predicate| {
                fulfill_cx.register_predicate_obligation(&infcx, predicate);
            });
            infcx.drain_fulfillment_cx_or_panic(DUMMY_SP, &mut fulfill_cx, &vtable)
        })
    }

    /// Trait method, which has to be resolved to an impl method.
    fn trait_method(
        &self,
        trait_id: DefId,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
        args: &mut Vec<(Pointer, Ty<'tcx>)>,
    ) -> EvalResult<'tcx, (DefId, &'tcx Substs<'tcx>)> {
        let trait_ref = ty::TraitRef::from_method(self.tcx, trait_id, substs);
        let trait_ref = self.tcx.normalize_associated_type(&ty::Binder(trait_ref));

        match self.fulfill_obligation(trait_ref) {
            traits::VtableImpl(vtable_impl) => {
                let impl_did = vtable_impl.impl_def_id;
                let mname = self.tcx.item_name(def_id);
                // Create a concatenated set of substitutions which includes those from the impl
                // and those from the method:
                let mth = get_impl_method(self.tcx, substs, impl_did, vtable_impl.substs, mname);

                Ok((mth.method.def_id, mth.substs))
            }

            traits::VtableClosure(vtable_closure) =>
                Ok((vtable_closure.closure_def_id, vtable_closure.substs.func_substs)),

            traits::VtableFnPointer(vtable_fn_ptr) => {
                if let ty::TyFnDef(did, ref substs, _) = vtable_fn_ptr.fn_ty.sty {
                    args.remove(0);
                    Ok((did, substs))
                } else {
                    bug!("VtableFnPointer did not contain a concrete function: {:?}", vtable_fn_ptr)
                }
            }

            traits::VtableObject(ref data) => {
                let idx = self.tcx.get_vtable_index_of_object_method(data, def_id);
                if let Some(&mut(first_arg, ref mut first_ty)) = args.get_mut(0) {
                    let (_, vtable) = self.get_fat_ptr(first_arg);
                    let vtable = self.memory.read_ptr(vtable)?;
                    let idx = idx + 3;
                    let offset = idx * self.memory.pointer_size();
                    let fn_ptr = self.memory.read_ptr(vtable.offset(offset as isize))?;
                    let (def_id, substs, ty) = self.memory.get_fn(fn_ptr.alloc_id)?;
                    // FIXME: skip_binder is wrong for HKL
                    *first_ty = ty.sig.skip_binder().inputs[0];
                    Ok((def_id, substs))
                } else {
                    Err(EvalError::VtableForArgumentlessMethod)
                }
            },
            vtable => bug!("resolved vtable bad vtable {:?} in trans", vtable),
        }
    }

    pub(super) fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        self.tcx.type_needs_drop_given_env(ty, &self.tcx.empty_parameter_environment())
    }

    fn drop(&mut self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, ()> {
        if !self.type_needs_drop(ty) {
            debug!("no need to drop {:?}", ty);
            return Ok(());
        }
        trace!("-need to drop {:?}", ty);

        // TODO(solson): Call user-defined Drop::drop impls.

        match ty.sty {
            ty::TyBox(_contents_ty) => {
                let contents_ptr = self.memory.read_ptr(ptr)?;
                // self.drop(contents_ptr, contents_ty)?;
                trace!("-deallocating box");
                self.memory.deallocate(contents_ptr)?;
            }

            // TODO(solson): Implement drop for other relevant types (e.g. aggregates).
            _ => {}
        }

        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct ImplMethod<'tcx> {
    pub(super) method: Rc<ty::Method<'tcx>>,
    pub(super) substs: &'tcx Substs<'tcx>,
    pub(super) is_provided: bool,
}

/// Locates the applicable definition of a method, given its name.
pub(super) fn get_impl_method<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    substs: &'tcx Substs<'tcx>,
    impl_def_id: DefId,
    impl_substs: &'tcx Substs<'tcx>,
    name: ast::Name,
) -> ImplMethod<'tcx> {
    assert!(!substs.needs_infer());

    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();
    let trait_def = tcx.lookup_trait_def(trait_def_id);

    match trait_def.ancestors(impl_def_id).fn_defs(tcx, name).next() {
        Some(node_item) => {
            let substs = tcx.infer_ctxt(None, None, Reveal::All).enter(|infcx| {
                let substs = substs.rebase_onto(tcx, trait_def_id, impl_substs);
                let substs = traits::translate_substs(&infcx, impl_def_id,
                                                      substs, node_item.node);
                tcx.lift(&substs).unwrap_or_else(|| {
                    bug!("trans::meth::get_impl_method: translate_substs \
                          returned {:?} which contains inference types/regions",
                         substs);
                })
            });
            ImplMethod {
                method: node_item.item,
                substs: substs,
                is_provided: node_item.node.is_from_trait(),
            }
        }
        None => {
            bug!("method {:?} not found in {:?}", name, impl_def_id)
        }
    }
}
