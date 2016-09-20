use rustc::hir::def_id::DefId;
use rustc::mir::repr as mir;
use rustc::ty::layout::Layout;
use rustc::ty::subst::Substs;
use rustc::ty;

use error::{EvalError, EvalResult};
use memory::Pointer;
use interpreter::EvalContext;
use primval;

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(super) fn call_intrinsic(
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

            "arith_offset" => {
                let ptr = self.memory.read_ptr(args_ptrs[0])?;
                let offset = self.memory.read_int(args_ptrs[1], pointer_size)?;
                let new_ptr = ptr.offset(offset as isize);
                self.memory.write_ptr(dest, new_ptr)?;
            }

            "assume" => {
                if !self.memory.read_bool(args_ptrs[0])? {
                    return Err(EvalError::AssumptionNotHeld);
                }
            }

            "breakpoint" => unimplemented!(), // halt miri

            "copy" |
            "copy_nonoverlapping" => {
                // FIXME: check whether overlapping occurs
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

            "fabsf32" => {
                let f = self.memory.read_f32(args_ptrs[0])?;
                self.memory.write_f32(dest, f.abs())?;
            }

            "fabsf64" => {
                let f = self.memory.read_f64(args_ptrs[0])?;
                self.memory.write_f64(dest, f.abs())?;
            }

            "fadd_fast" => {
                let ty = substs.type_at(0);
                let a = self.read_primval(args_ptrs[0], ty)?;
                let b = self.read_primval(args_ptrs[0], ty)?;
                let result = primval::binary_op(mir::BinOp::Add, a, b)?;
                self.memory.write_primval(dest, result.0)?;
            }

            "likely" |
            "unlikely" |
            "forget" => {}

            "init" => self.memory.write_repeat(dest, 0, dest_layout.size(&self.tcx.data_layout).bytes() as usize)?,

            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = self.type_align(elem_ty);
                self.memory.write_uint(dest, elem_align as u64, pointer_size)?;
            }

            "pref_align_of" => {
                let ty = substs.type_at(0);
                let layout = self.type_layout(ty);
                let align = layout.align(&self.tcx.data_layout).pref();
                self.memory.write_uint(dest, align, pointer_size)?;
            }

            "move_val_init" => {
                let ty = substs.type_at(0);
                let ptr = self.memory.read_ptr(args_ptrs[0])?;
                self.move_(args_ptrs[1], ptr, ty)?;
            }

            "needs_drop" => {
                let ty = substs.type_at(0);
                self.memory.write_bool(dest, self.tcx.type_needs_drop_given_env(ty, &self.tcx.empty_parameter_environment()))?;
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

            "powif32" => {
                let f = self.memory.read_f32(args_ptrs[0])?;
                let i = self.memory.read_int(args_ptrs[1], 4)?;
                self.memory.write_f32(dest, f.powi(i as i32))?;
            }

            "powif64" => {
                let f = self.memory.read_f32(args_ptrs[0])?;
                let i = self.memory.read_int(args_ptrs[1], 4)?;
                self.memory.write_f32(dest, f.powi(i as i32))?;
            }

            "sqrtf32" => {
                let f = self.memory.read_f32(args_ptrs[0])?;
                self.memory.write_f32(dest, f.sqrt())?;
            }

            "sqrtf64" => {
                let f = self.memory.read_f64(args_ptrs[0])?;
                self.memory.write_f64(dest, f.sqrt())?;
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
            // FIXME: wait for eval_operand_to_ptr to be gone
            /*
            "type_name" => {
                let ty = substs.type_at(0);
                let ty_name = ty.to_string();
                let s = self.str_to_value(&ty_name)?;
                self.memory.write_ptr(dest, s)?;
            }*/
            "type_id" => {
                let ty = substs.type_at(0);
                let n = self.tcx.type_id_hash(ty);
                self.memory.write_uint(dest, n, 8)?;
            }

            "transmute" => {
                let ty = substs.type_at(0);
                self.move_(args_ptrs[0], dest, ty)?;
            }

            "try" => unimplemented!(),

            "uninit" => self.memory.mark_definedness(dest, dest_layout.size(&self.tcx.data_layout).bytes() as usize, false)?,

            "volatile_load" => {
                let ty = substs.type_at(0);
                let ptr = self.memory.read_ptr(args_ptrs[0])?;
                self.move_(ptr, dest, ty)?;
            }

            "volatile_store" => {
                let ty = substs.type_at(0);
                let dest = self.memory.read_ptr(args_ptrs[0])?;
                self.move_(args_ptrs[1], dest, ty)?;
            }

            name => return Err(EvalError::Unimplemented(format!("unimplemented intrinsic: {}", name))),
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }
}
