use rustc::hir::def_id::DefId;
use rustc::mir::repr as mir;
use rustc::traits::{self, ProjectionMode};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::layout::Layout;
use rustc::ty::subst::{self, Substs};
use rustc::ty::{self, Ty, TyCtxt, BareFnTy};
use std::rc::Rc;
use std::iter;
use syntax::{ast, attr};
use syntax::codemap::{DUMMY_SP, Span};

use super::{EvalContext, IntegerExt};
use error::{EvalError, EvalResult};
use memory::{Pointer, FunctionDefinition};

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(super) fn eval_terminator(
        &mut self,
        terminator: &mir::Terminator<'tcx>,
    ) -> EvalResult<'tcx, ()> {
        use rustc::mir::repr::TerminatorKind::*;
        match terminator.kind {
            Return => self.pop_stack_frame(),

            Goto { target } => {
                self.frame_mut().block = target;
            },

            If { ref cond, targets: (then_target, else_target) } => {
                let cond_ptr = self.eval_operand(cond)?;
                let cond_val = self.memory.read_bool(cond_ptr)?;
                self.frame_mut().block = if cond_val { then_target } else { else_target };
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
                        return Err(EvalError::InvalidChar(discr_val as u32));
                    }
                }

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets[targets.len() - 1];

                for (index, val_const) in values.iter().enumerate() {
                    let ptr = self.const_to_ptr(val_const)?;
                    let val = self.memory.read_uint(ptr, discr_size)?;
                    if discr_val == val {
                        target_block = targets[index];
                        break;
                    }
                }

                self.frame_mut().block = target_block;
            }

            Switch { ref discr, ref targets, adt_def } => {
                let adt_ptr = self.eval_lvalue(discr)?.to_ptr();
                let adt_ty = self.lvalue_ty(discr);
                let discr_val = self.read_discriminant_value(adt_ptr, adt_ty)?;
                let matching = adt_def.variants.iter()
                    .position(|v| discr_val == v.disr_val.to_u64_unchecked());

                match matching {
                    Some(i) => {
                        self.frame_mut().block = targets[i];
                    },
                    None => return Err(EvalError::InvalidDiscriminant),
                }
            }

            Call { ref func, ref args, ref destination, .. } => {
                let mut return_ptr = None;
                if let Some((ref lv, target)) = *destination {
                    self.frame_mut().block = target;
                    return_ptr = Some(self.eval_lvalue(lv)?.to_ptr());
                }

                let func_ty = self.operand_ty(func);
                match func_ty.sty {
                    ty::TyFnPtr(bare_fn_ty) => {
                        let ptr = self.eval_operand(func)?;
                        assert_eq!(ptr.offset, 0);
                        let fn_ptr = self.memory.read_ptr(ptr)?;
                        let FunctionDefinition { def_id, substs, fn_ty } = self.memory.get_fn(fn_ptr.alloc_id)?;
                        if fn_ty != bare_fn_ty {
                            return Err(EvalError::FunctionPointerTyMismatch(fn_ty, bare_fn_ty));
                        }
                        self.eval_fn_call(def_id, substs, bare_fn_ty, return_ptr, args,
                                          terminator.source_info.span)?
                    },
                    ty::TyFnDef(def_id, substs, fn_ty) => {
                        self.eval_fn_call(def_id, substs, fn_ty, return_ptr, args,
                                          terminator.source_info.span)?
                    }

                    _ => return Err(EvalError::Unimplemented(format!("can't handle callee of type {:?}", func_ty))),
                }
            }

            Drop { ref location, target, .. } => {
                let ptr = self.eval_lvalue(location)?.to_ptr();
                let ty = self.lvalue_ty(location);
                self.drop(ptr, ty)?;
                self.frame_mut().block = target;
            }

            Assert { ref cond, expected, ref msg, target, .. } => {
                let cond_ptr = self.eval_operand(cond)?;
                if expected == self.memory.read_bool(cond_ptr)? {
                    self.frame_mut().block = target;
                } else {
                    return match *msg {
                        mir::AssertMessage::BoundsCheck { ref len, ref index } => {
                            let len = self.eval_operand(len).expect("can't eval len");
                            let len = self.memory.read_usize(len).expect("can't read len");
                            let index = self.eval_operand(index).expect("can't eval index");
                            let index = self.memory.read_usize(index).expect("can't read index");
                            Err(EvalError::ArrayIndexOutOfBounds(terminator.source_info.span, len, index))
                        },
                        mir::AssertMessage::Math(ref err) => Err(EvalError::Math(terminator.source_info.span, err.clone())),
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
        return_ptr: Option<Pointer>,
        args: &[mir::Operand<'tcx>],
        span: Span,
    ) -> EvalResult<'tcx, ()> {
        use syntax::abi::Abi;
        match fn_ty.abi {
            Abi::RustIntrinsic => {
                let name = self.tcx.item_name(def_id).as_str();
                match fn_ty.sig.0.output {
                    ty::FnConverging(ty) => {
                        let layout = self.type_layout(ty);
                        let ret = return_ptr.unwrap();
                        self.call_intrinsic(&name, substs, args, ret, layout)
                    }
                    ty::FnDiverging => unimplemented!(),
                }
            }

            Abi::C => {
                match fn_ty.sig.0.output {
                    ty::FnConverging(ty) => {
                        let size = self.type_size(ty);
                        self.call_c_abi(def_id, args, return_ptr.unwrap(), size)
                    }
                    ty::FnDiverging => unimplemented!(),
                }
            }

            Abi::Rust | Abi::RustCall => {
                // TODO(solson): Adjust the first argument when calling a Fn or
                // FnMut closure via FnOnce::call_once.

                // Only trait methods can have a Self parameter.
                let (resolved_def_id, resolved_substs) = if substs.self_ty().is_some() {
                    self.trait_method(def_id, substs)
                } else {
                    (def_id, substs)
                };

                let mut arg_srcs = Vec::new();
                for arg in args {
                    let src = self.eval_operand(arg)?;
                    let src_ty = self.operand_ty(arg);
                    arg_srcs.push((src, src_ty));
                }

                if fn_ty.abi == Abi::RustCall && !args.is_empty() {
                    arg_srcs.pop();
                    let last_arg = args.last().unwrap();
                    let last = self.eval_operand(last_arg)?;
                    let last_ty = self.operand_ty(last_arg);
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
                        ty => panic!("expected tuple as last argument in function with 'rust-call' ABI, got {:?}", ty),
                    }
                }

                let mir = self.load_mir(resolved_def_id);
                self.push_stack_frame(def_id, span, mir, resolved_substs, return_ptr);

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
            Vector { .. } => 0,
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
        name: &str,
        substs: &'tcx Substs<'tcx>,
        args: &[mir::Operand<'tcx>],
        dest: Pointer,
        dest_layout: &'tcx Layout,
    ) -> EvalResult<'tcx, ()> {
        let args_res: EvalResult<Vec<Pointer>> = args.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args_ptrs = args_res?;

        let pointer_size = self.memory.pointer_size();

        match name {
            "add_with_overflow" => self.intrinsic_with_overflow(mir::BinOp::Add, &args[0], &args[1], dest, dest_layout)?,
            "sub_with_overflow" => self.intrinsic_with_overflow(mir::BinOp::Sub, &args[0], &args[1], dest, dest_layout)?,
            "mul_with_overflow" => self.intrinsic_with_overflow(mir::BinOp::Mul, &args[0], &args[1], dest, dest_layout)?,

            // FIXME: turn into an assertion to catch wrong `assume` that would cause UB in llvm
            "assume" => {}

            "copy_nonoverlapping" => {
                let elem_ty = *substs.types.get(subst::FnSpace, 0);
                let elem_size = self.type_size(elem_ty);
                let src = self.memory.read_ptr(args_ptrs[0])?;
                let dest = self.memory.read_ptr(args_ptrs[1])?;
                let count = self.memory.read_isize(args_ptrs[2])?;
                self.memory.copy(src, dest, count as usize * elem_size)?;
            }

            "discriminant_value" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                let adt_ptr = self.memory.read_ptr(args_ptrs[0])?;
                let discr_val = self.read_discriminant_value(adt_ptr, ty)?;
                self.memory.write_uint(dest, discr_val, 8)?;
            }

            "forget" => {}

            "init" => self.memory.write_repeat(dest, 0, dest_layout.size(&self.tcx.data_layout).bytes() as usize)?,

            "min_align_of" => {
                // FIXME: use correct value
                self.memory.write_int(dest, 1, pointer_size)?;
            }

            "move_val_init" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                let ptr = self.memory.read_ptr(args_ptrs[0])?;
                self.move_(args_ptrs[1], ptr, ty)?;
            }

            "offset" => {
                let pointee_ty = *substs.types.get(subst::FnSpace, 0);
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
                let ty = *substs.types.get(subst::FnSpace, 0);
                let size = self.type_size(ty) as u64;
                self.memory.write_uint(dest, size, pointer_size)?;
            }

            "size_of_val" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
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
                let ty = *substs.types.get(subst::FnSpace, 0);
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

        let args_res: EvalResult<Vec<Pointer>> = args.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args = args_res?;

        if link_name.starts_with("pthread_") {
            warn!("ignoring C ABI call: {}", link_name);
            return Ok(());
        }

        match &link_name[..] {
            "__rust_allocate" => {
                let size = self.memory.read_usize(args[0])?;
                let ptr = self.memory.allocate(size as usize);
                self.memory.write_ptr(dest, ptr)?;
            }

            "__rust_reallocate" => {
                let ptr = self.memory.read_ptr(args[0])?;
                let size = self.memory.read_usize(args[2])?;
                let new_ptr = self.memory.reallocate(ptr, size as usize)?;
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

    fn fulfill_obligation(&self, trait_ref: ty::PolyTraitRef<'tcx>) -> traits::Vtable<'tcx, ()> {
        // Do the initial selection for the obligation. This yields the shallow result we are
        // looking for -- that is, what specific impl.
        self.tcx.normalizing_infer_ctxt(ProjectionMode::Any).enter(|infcx| {
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
        def_id: DefId,
        substs: &'tcx Substs<'tcx>
    ) -> (DefId, &'tcx Substs<'tcx>) {
        let method_item = self.tcx.impl_or_trait_item(def_id);
        let trait_id = method_item.container().id();
        let trait_ref = ty::Binder(substs.to_trait_ref(self.tcx, trait_id));
        match self.fulfill_obligation(trait_ref) {
            traits::VtableImpl(vtable_impl) => {
                let impl_did = vtable_impl.impl_def_id;
                let mname = self.tcx.item_name(def_id);
                // Create a concatenated set of substitutions which includes those from the impl
                // and those from the method:
                let impl_substs = vtable_impl.substs.with_method_from(substs);
                let substs = self.tcx.mk_substs(impl_substs);
                let mth = get_impl_method(self.tcx, impl_did, substs, mname);

                (mth.method.def_id, mth.substs)
            }

            traits::VtableClosure(vtable_closure) =>
                (vtable_closure.closure_def_id, vtable_closure.substs.func_substs),

            traits::VtableFnPointer(_fn_ty) => {
                let _trait_closure_kind = self.tcx.lang_items.fn_trait_kind(trait_id).unwrap();
                unimplemented!()
                // let llfn = trans_fn_pointer_shim(ccx, trait_closure_kind, fn_ty);

                // let method_ty = def_ty(tcx, def_id, substs);
                // let fn_ptr_ty = match method_ty.sty {
                //     ty::TyFnDef(_, _, fty) => tcx.mk_ty(ty::TyFnPtr(fty)),
                //     _ => unreachable!("expected fn item type, found {}",
                //                       method_ty)
                // };
                // Callee::ptr(immediate_rvalue(llfn, fn_ptr_ty))
            }

            traits::VtableObject(ref _data) => {
                unimplemented!()
                // Callee {
                //     data: Virtual(traits::get_vtable_index_of_object_method(
                //                   tcx, data, def_id)),
                //                   ty: def_ty(tcx, def_id, substs)
                // }
            }
            vtable => unreachable!("resolved vtable bad vtable {:?} in trans", vtable),
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
struct ImplMethod<'tcx> {
    method: Rc<ty::Method<'tcx>>,
    substs: &'tcx Substs<'tcx>,
    is_provided: bool,
}

/// Locates the applicable definition of a method, given its name.
fn get_impl_method<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    impl_def_id: DefId,
    substs: &'tcx Substs<'tcx>,
    name: ast::Name,
) -> ImplMethod<'tcx> {
    assert!(!substs.types.needs_infer());

    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();
    let trait_def = tcx.lookup_trait_def(trait_def_id);

    match trait_def.ancestors(impl_def_id).fn_defs(tcx, name).next() {
        Some(node_item) => {
            let substs = tcx.normalizing_infer_ctxt(ProjectionMode::Any).enter(|infcx| {
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
