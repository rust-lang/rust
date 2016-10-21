use rustc::hir::def_id::DefId;
use rustc::mir::repr as mir;
use rustc::ty::layout::Layout;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty};

use error::{EvalError, EvalResult};
use interpreter::value::Value;
use interpreter::{EvalContext, Lvalue};
use primval::{self, PrimVal, PrimValKind};

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(super) fn call_intrinsic(
        &mut self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
        args: &[mir::Operand<'tcx>],
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
        dest_layout: &'tcx Layout,
    ) -> EvalResult<'tcx, ()> {
        let args_ptrs: EvalResult<Vec<Value>> = args.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args_ptrs = args_ptrs?;
        let i32 = self.tcx.types.i32;
        let isize = self.tcx.types.isize;
        let usize = self.tcx.types.usize;
        let f32 = self.tcx.types.f32;
        let f64 = self.tcx.types.f64;

        let intrinsic_name = &self.tcx.item_name(def_id).as_str()[..];
        match intrinsic_name {
            "add_with_overflow" =>
                self.intrinsic_with_overflow(mir::BinOp::Add, &args[0], &args[1], dest, dest_ty)?,

            "sub_with_overflow" =>
                self.intrinsic_with_overflow(mir::BinOp::Sub, &args[0], &args[1], dest, dest_ty)?,

            "mul_with_overflow" =>
                self.intrinsic_with_overflow(mir::BinOp::Mul, &args[0], &args[1], dest, dest_ty)?,


            "arith_offset" => {
                let ptr = args_ptrs[0].read_ptr(&self.memory)?;
                let offset = self.value_to_primval(args_ptrs[1], isize)?
                    .expect_int("arith_offset second arg not isize");
                let new_ptr = ptr.offset(offset as isize);
                self.write_primval(dest, PrimVal::from_ptr(new_ptr))?;
            }

            "assume" => {
                let bool = self.tcx.types.bool;
                let cond = self.value_to_primval(args_ptrs[0], bool)?
                    .expect_bool("assume arg not bool");
                if !cond { return Err(EvalError::AssumptionNotHeld); }
            }

            "atomic_load" |
            "volatile_load" => {
                let ty = substs.type_at(0);
                let ptr = args_ptrs[0].read_ptr(&self.memory)?;
                self.write_value(Value::ByRef(ptr), dest, ty)?;
            }

            "atomic_store" |
            "volatile_store" => {
                let ty = substs.type_at(0);
                let dest = args_ptrs[0].read_ptr(&self.memory)?;
                self.write_value_to_ptr(args_ptrs[1], dest, ty)?;
            }

            "breakpoint" => unimplemented!(), // halt miri

            "copy" |
            "copy_nonoverlapping" => {
                // FIXME: check whether overlapping occurs
                let elem_ty = substs.type_at(0);
                let elem_size = self.type_size(elem_ty);
                let elem_align = self.type_align(elem_ty);
                let src = args_ptrs[0].read_ptr(&self.memory)?;
                let dest = args_ptrs[1].read_ptr(&self.memory)?;
                let count = self.value_to_primval(args_ptrs[2], usize)?
                    .expect_uint("arith_offset second arg not isize");
                self.memory.copy(src, dest, count as usize * elem_size, elem_align)?;
            }

            "ctpop" |
            "cttz" |
            "ctlz" |
            "bswap" => {
                let elem_ty = substs.type_at(0);
                let num = self.value_to_primval(args_ptrs[0], elem_ty)?;
                let num = numeric_intrinsic(intrinsic_name, num);
                self.write_primval(dest, num)?;
            }

            "discriminant_value" => {
                let ty = substs.type_at(0);
                let adt_ptr = args_ptrs[0].read_ptr(&self.memory)?;
                let discr_val = self.read_discriminant_value(adt_ptr, ty)?;
                self.write_primval(dest, PrimVal::new(discr_val, PrimValKind::U64))?;
            }

            "fabsf32" => {
                let f = self.value_to_primval(args_ptrs[2], f32)?
                    .expect_f32("fabsf32 read non f32");
                self.write_primval(dest, PrimVal::from_f32(f.abs()))?;
            }

            "fabsf64" => {
                let f = self.value_to_primval(args_ptrs[2], f64)?
                    .expect_f64("fabsf64 read non f64");
                self.write_primval(dest, PrimVal::from_f64(f.abs()))?;
            }

            "fadd_fast" => {
                let ty = substs.type_at(0);
                let a = self.value_to_primval(args_ptrs[0], ty)?;
                let b = self.value_to_primval(args_ptrs[0], ty)?;
                let result = primval::binary_op(mir::BinOp::Add, a, b)?;
                self.write_primval(dest, result.0)?;
            }

            "likely" |
            "unlikely" |
            "forget" => {}

            "init" => {
                // FIXME(solson)
                let dest = self.force_allocation(dest)?.to_ptr();

                let size = dest_layout.size(&self.tcx.data_layout).bytes() as usize;
                self.memory.write_repeat(dest, 0, size)?;
            }

            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = self.type_align(elem_ty);
                let align_val = self.usize_primval(elem_align as u64);
                self.write_primval(dest, align_val)?;
            }

            "pref_align_of" => {
                let ty = substs.type_at(0);
                let layout = self.type_layout(ty);
                let align = layout.align(&self.tcx.data_layout).pref();
                let align_val = self.usize_primval(align);
                self.write_primval(dest, align_val)?;
            }

            "move_val_init" => {
                let ty = substs.type_at(0);
                let ptr = args_ptrs[0].read_ptr(&self.memory)?;
                self.write_value_to_ptr(args_ptrs[1], ptr, ty)?;
            }

            "needs_drop" => {
                let ty = substs.type_at(0);
                let env = self.tcx.empty_parameter_environment();
                let needs_drop = self.tcx.type_needs_drop_given_env(ty, &env);
                self.write_primval(dest, PrimVal::from_bool(needs_drop))?;
            }

            "offset" => {
                let pointee_ty = substs.type_at(0);
                let pointee_size = self.type_size(pointee_ty) as isize;
                let offset = self.value_to_primval(args_ptrs[1], isize)?
                    .expect_int("offset second arg not isize");

                let ptr = args_ptrs[0].read_ptr(&self.memory)?;
                let result_ptr = ptr.offset(offset as isize * pointee_size);
                self.write_primval(dest, PrimVal::from_ptr(result_ptr))?;
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
                let f = self.value_to_primval(args_ptrs[0], f32)?
                    .expect_f32("powif32 first arg not f32");
                let i = self.value_to_primval(args_ptrs[1], i32)?
                    .expect_int("powif32 second arg not i32");
                self.write_primval(dest, PrimVal::from_f32(f.powi(i as i32)))?;
            }

            "powif64" => {
                let f = self.value_to_primval(args_ptrs[0], f64)?
                    .expect_f64("powif64 first arg not f64");
                let i = self.value_to_primval(args_ptrs[1], i32)?
                    .expect_int("powif64 second arg not i32");
                self.write_primval(dest, PrimVal::from_f64(f.powi(i as i32)))?;
            }

            "sqrtf32" => {
                let f = self.value_to_primval(args_ptrs[0], f32)?
                    .expect_f32("sqrtf32 first arg not f32");
                self.write_primval(dest, PrimVal::from_f32(f.sqrt()))?;
            }

            "sqrtf64" => {
                let f = self.value_to_primval(args_ptrs[0], f64)?
                    .expect_f64("sqrtf64 first arg not f64");
                self.write_primval(dest, PrimVal::from_f64(f.sqrt()))?;
            }

            "size_of" => {
                let ty = substs.type_at(0);
                let size = self.type_size(ty) as u64;
                let size_val = self.usize_primval(size);
                self.write_primval(dest, size_val)?;
            }

            "size_of_val" => {
                let ty = substs.type_at(0);
                let (size, _) = self.size_and_align_of_dst(ty, args_ptrs[0])?;
                let size_val = self.usize_primval(size);
                self.write_primval(dest, size_val)?;
            }
            "type_name" => {
                let ty = substs.type_at(0);
                let ty_name = ty.to_string();
                let s = self.str_to_value(&ty_name)?;
                self.write_value(s, dest, dest_ty)?;
            }
            "type_id" => {
                let ty = substs.type_at(0);
                let n = self.tcx.type_id_hash(ty);
                self.write_primval(dest, PrimVal::new(n, PrimValKind::U64))?;
            }

            "transmute" => {
                let ty = substs.type_at(0);
                self.write_value(args_ptrs[0], dest, ty)?;
            }

            "uninit" => {
                // FIXME(solson): Attempt writing a None over the destination when it's an
                // Lvalue::Local (that is not ByRef). Otherwise do the mark_definedness as usual.
                let dest = self.force_allocation(dest)?.to_ptr();

                let size = dest_layout.size(&self.tcx.data_layout).bytes() as usize;
                self.memory.mark_definedness(dest, size, false)?;
            }

            name => return Err(EvalError::Unimplemented(format!("unimplemented intrinsic: {}", name))),
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }

    fn size_and_align_of_dst(
        &self,
        ty: ty::Ty<'tcx>,
        value: Value,
    ) -> EvalResult<'tcx, (u64, u64)> {
        let pointer_size = self.memory.pointer_size();
        if self.type_is_sized(ty) {
            Ok((self.type_size(ty) as u64, self.type_align(ty) as u64))
        } else {
            match ty.sty {
                ty::TyAdt(def, substs) => {
                    // First get the size of all statically known fields.
                    // Don't use type_of::sizing_type_of because that expects t to be sized,
                    // and it also rounds up to alignment, which we want to avoid,
                    // as the unsized field's alignment could be smaller.
                    assert!(!ty.is_simd());
                    let layout = self.type_layout(ty);
                    debug!("DST {} layout: {:?}", ty, layout);

                    let (sized_size, sized_align) = match *layout {
                        ty::layout::Layout::Univariant { ref variant, .. } => {
                            // The offset of the start of the last field gives the size of the
                            // sized part of the type.
                            let size = variant.offsets.last().map_or(0, |f| f.bytes());
                            (size, variant.align.abi())
                        }
                        _ => {
                            bug!("size_and_align_of_dst: expcted Univariant for `{}`, found {:#?}",
                                 ty, layout);
                        }
                    };
                    debug!("DST {} statically sized prefix size: {} align: {}",
                           ty, sized_size, sized_align);

                    // Recurse to get the size of the dynamically sized field (must be
                    // the last field).
                    let last_field = def.struct_variant().fields.last().unwrap();
                    let field_ty = self.field_ty(substs, last_field);
                    let (unsized_size, unsized_align) = self.size_and_align_of_dst(field_ty, value)?;

                    // FIXME (#26403, #27023): We should be adding padding
                    // to `sized_size` (to accommodate the `unsized_align`
                    // required of the unsized field that follows) before
                    // summing it with `sized_size`. (Note that since #26403
                    // is unfixed, we do not yet add the necessary padding
                    // here. But this is where the add would go.)

                    // Return the sum of sizes and max of aligns.
                    let size = sized_size + unsized_size;

                    // Choose max of two known alignments (combined value must
                    // be aligned according to more restrictive of the two).
                    let align = ::std::cmp::max(sized_align, unsized_align);

                    // Issue #27023: must add any necessary padding to `size`
                    // (to make it a multiple of `align`) before returning it.
                    //
                    // Namely, the returned size should be, in C notation:
                    //
                    //   `size + ((size & (align-1)) ? align : 0)`
                    //
                    // emulated via the semi-standard fast bit trick:
                    //
                    //   `(size + (align-1)) & -align`

                    if size & (align - 1) != 0 {
                        Ok((size + align, align))
                    } else {
                        Ok((size, align))
                    }
                }
                ty::TyTrait(..) => {
                    let (_, vtable) = value.expect_ptr_vtable_pair(&self.memory)?;
                    // the second entry in the vtable is the dynamic size of the object.
                    let size = self.memory.read_usize(vtable.offset(pointer_size as isize))?;
                    let align = self.memory.read_usize(vtable.offset(pointer_size as isize * 2))?;
                    Ok((size, align))
                }

                ty::TySlice(_) | ty::TyStr => {
                    let elem_ty = ty.sequence_element_type(self.tcx);
                    let elem_size = self.type_size(elem_ty) as u64;
                    let len = value.expect_slice_len(&self.memory)?;
                    let align = self.type_align(elem_ty);
                    Ok((len * elem_size, align as u64))
                }

                _ => bug!("size_of_val::<{:?}>", ty),
            }
        }
    }
    /// Returns the normalized type of a struct field
    fn field_ty(
        &self,
        param_substs: &Substs<'tcx>,
        f: ty::FieldDef<'tcx>,
    )-> ty::Ty<'tcx> {
        self.tcx.normalize_associated_type(&f.ty(self.tcx, param_substs))
    }
}

macro_rules! integer_intrinsic {
    ($name:expr, $val:expr, $method:ident) => ({
        let val = $val;

        use primval::PrimValKind::*;
        let bits = match val.kind {
            I8 => (val.bits as i8).$method() as u64,
            U8 => (val.bits as u8).$method() as u64,
            I16 => (val.bits as i16).$method() as u64,
            U16 => (val.bits as u16).$method() as u64,
            I32 => (val.bits as i32).$method() as u64,
            U32 => (val.bits as u32).$method() as u64,
            I64 => (val.bits as i64).$method() as u64,
            U64 => (val.bits as u64).$method() as u64,
            _ => bug!("invalid `{}` argument: {:?}", $name, val),
        };

        PrimVal::new(bits, val.kind)
    });
}

fn numeric_intrinsic(name: &str, val: PrimVal) -> PrimVal {
    match name {
        "bswap" => integer_intrinsic!("bswap", val, swap_bytes),
        "ctlz"  => integer_intrinsic!("ctlz", val, leading_zeros),
        "ctpop" => integer_intrinsic!("ctpop", val, count_ones),
        "cttz"  => integer_intrinsic!("cttz", val, trailing_zeros),
        _       => bug!("not a numeric intrinsic: {}", name),
    }
}
