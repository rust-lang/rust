// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{self, Ty};
use rustc::ty::cast::{CastTy, IntTy};
use rustc::ty::layout::{self, LayoutOf, HasTyCtxt, TyLayout};
use rustc::mir;
use rustc::middle::lang_items::ExchangeMallocFnLangItem;
use rustc_apfloat::{ieee, Float, Status, Round};
use std::{u128, i128};

use base;
use callee;
use common::{self, IntPredicate, RealPredicate};
use monomorphize;
use type_of::LayoutLlvmExt;

use interfaces::*;

use super::{FunctionCx, LocalRef};
use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;

impl<'a, 'f, 'll: 'a + 'f, 'tcx: 'll, Cx: 'a + CodegenMethods<'ll, 'tcx>>
    FunctionCx<'a, 'f, 'll, 'tcx, Cx> where
    &'a Cx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
{
    pub fn codegen_rvalue<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(
        &mut self,
        bx: Bx,
        dest: PlaceRef<'tcx, Cx::Value>,
        rvalue: &mir::Rvalue<'tcx>
    ) -> Bx {
        debug!("codegen_rvalue(dest.llval={:?}, rvalue={:?})",
               dest.llval, rvalue);

        match *rvalue {
           mir::Rvalue::Use(ref operand) => {
               let cg_operand = self.codegen_operand(&bx, operand);
               // FIXME: consider not copying constants through stack. (fixable by codegenning
               // constants into OperandValue::Ref, why don’t we do that yet if we don’t?)
               cg_operand.val.store(&bx, dest);
               bx
           }

            mir::Rvalue::Cast(mir::CastKind::Unsize, ref source, _) => {
                // The destination necessarily contains a fat pointer, so if
                // it's a scalar pair, it's a fat pointer or newtype thereof.
                if dest.layout.is_llvm_scalar_pair() {
                    // into-coerce of a thin pointer to a fat pointer - just
                    // use the operand path.
                    let (bx, temp) = self.codegen_rvalue_operand(bx, rvalue);
                    temp.val.store(&bx, dest);
                    return bx;
                }

                // Unsize of a nontrivial struct. I would prefer for
                // this to be eliminated by MIR building, but
                // `CoerceUnsized` can be passed by a where-clause,
                // so the (generic) MIR may not be able to expand it.
                let operand = self.codegen_operand(&bx, source);
                match operand.val {
                    OperandValue::Pair(..) |
                    OperandValue::Immediate(_) => {
                        // unsize from an immediate structure. We don't
                        // really need a temporary alloca here, but
                        // avoiding it would require us to have
                        // `coerce_unsized_into` use extractvalue to
                        // index into the struct, and this case isn't
                        // important enough for it.
                        debug!("codegen_rvalue: creating ugly alloca");
                        let scratch = PlaceRef::alloca(&bx, operand.layout, "__unsize_temp");
                        scratch.storage_live(&bx);
                        operand.val.store(&bx, scratch);
                        base::coerce_unsized_into(&bx, scratch, dest);
                        scratch.storage_dead(&bx);
                    }
                    OperandValue::Ref(llref, None, align) => {
                        let source = PlaceRef::new_sized(llref, operand.layout, align);
                        base::coerce_unsized_into(&bx, source, dest);
                    }
                    OperandValue::Ref(_, Some(_), _) => {
                        bug!("unsized coercion on an unsized rvalue")
                    }
                }
                bx
            }

            mir::Rvalue::Repeat(ref elem, count) => {
                let cg_elem = self.codegen_operand(&bx, elem);

                // Do not generate the loop for zero-sized elements or empty arrays.
                if dest.layout.is_zst() {
                    return bx;
                }

                let start = dest.project_index(&bx, bx.cx().const_usize(0)).llval;

                if let OperandValue::Immediate(v) = cg_elem.val {
                    let align = bx.cx().const_i32(dest.align.abi() as i32);
                    let size = bx.cx().const_usize(dest.layout.size.bytes());

                    // Use llvm.memset.p0i8.* to initialize all zero arrays
                    if bx.cx().is_const_integral(v) && bx.cx().const_to_uint(v) == 0 {
                        let fill = bx.cx().const_u8(0);
                        bx.call_memset(start, fill, size, align, false);
                        return bx;
                    }

                    // Use llvm.memset.p0i8.* to initialize byte arrays
                    let v = base::from_immediate(&bx, v);
                    if bx.cx().val_ty(v) == bx.cx().type_i8() {
                        bx.call_memset(start, v, size, align, false);
                        return bx;
                    }
                }

                let count = bx.cx().const_usize(count);
                let end = dest.project_index(&bx, count).llval;

                let header_bx = bx.build_sibling_block("repeat_loop_header");
                let body_bx = bx.build_sibling_block("repeat_loop_body");
                let next_bx = bx.build_sibling_block("repeat_loop_next");

                bx.br(header_bx.llbb());
                let current = header_bx.phi(bx.cx().val_ty(start), &[start], &[bx.llbb()]);

                let keep_going = header_bx.icmp(IntPredicate::IntNE, current, end);
                header_bx.cond_br(keep_going, body_bx.llbb(), next_bx.llbb());

                cg_elem.val.store(&body_bx,
                    PlaceRef::new_sized(current, cg_elem.layout, dest.align));

                let next = body_bx.inbounds_gep(current, &[bx.cx().const_usize(1)]);
                body_bx.br(header_bx.llbb());
                header_bx.add_incoming_to_phi(current, next, body_bx.llbb());

                next_bx
            }

            mir::Rvalue::Aggregate(ref kind, ref operands) => {
                let (dest, active_field_index) = match **kind {
                    mir::AggregateKind::Adt(adt_def, variant_index, _, _, active_field_index) => {
                        dest.codegen_set_discr(&bx, variant_index);
                        if adt_def.is_enum() {
                            (dest.project_downcast(&bx, variant_index), active_field_index)
                        } else {
                            (dest, active_field_index)
                        }
                    }
                    _ => (dest, None)
                };
                for (i, operand) in operands.iter().enumerate() {
                    let op = self.codegen_operand(&bx, operand);
                    // Do not generate stores and GEPis for zero-sized fields.
                    if !op.layout.is_zst() {
                        let field_index = active_field_index.unwrap_or(i);
                        op.val.store(&bx, dest.project_field(&bx, field_index));
                    }
                }
                bx
            }

            _ => {
                assert!(self.rvalue_creates_operand(rvalue));
                let (bx, temp) = self.codegen_rvalue_operand(bx, rvalue);
                temp.val.store(&bx, dest);
                bx
            }
        }
    }

    pub fn codegen_rvalue_unsized<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(&mut self,
                        bx: Bx,
                        indirect_dest: PlaceRef<'tcx, Cx::Value>,
                        rvalue: &mir::Rvalue<'tcx>)
                        -> Bx
    {
        debug!("codegen_rvalue_unsized(indirect_dest.llval={:?}, rvalue={:?})",
               indirect_dest.llval, rvalue);

        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                let cg_operand = self.codegen_operand(&bx, operand);
                cg_operand.val.store_unsized(&bx, indirect_dest);
                bx
            }

            _ => bug!("unsized assignment other than Rvalue::Use"),
        }
    }

    pub fn codegen_rvalue_operand<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(
        &mut self,
        bx: Bx,
        rvalue: &mir::Rvalue<'tcx>
    ) -> (Bx, OperandRef<'tcx, Cx::Value>) {
        assert!(self.rvalue_creates_operand(rvalue), "cannot codegen {:?} to operand", rvalue);

        match *rvalue {
            mir::Rvalue::Cast(ref kind, ref source, mir_cast_ty) => {
                let operand = self.codegen_operand(&bx, source);
                debug!("cast operand is {:?}", operand);
                let cast = bx.cx().layout_of(self.monomorphize(&mir_cast_ty));

                let val = match *kind {
                    mir::CastKind::ReifyFnPointer => {
                        match operand.layout.ty.sty {
                            ty::FnDef(def_id, substs) => {
                                if bx.cx().tcx().has_attr(def_id, "rustc_args_required_const") {
                                    bug!("reifying a fn ptr that requires \
                                          const arguments");
                                }
                                OperandValue::Immediate(
                                    callee::resolve_and_get_fn(bx.cx(), def_id, substs))
                            }
                            _ => {
                                bug!("{} cannot be reified to a fn ptr", operand.layout.ty)
                            }
                        }
                    }
                    mir::CastKind::ClosureFnPointer => {
                        match operand.layout.ty.sty {
                            ty::Closure(def_id, substs) => {
                                let instance = monomorphize::resolve_closure(
                                    *bx.cx().tcx(), def_id, substs, ty::ClosureKind::FnOnce);
                                OperandValue::Immediate(bx.cx().get_fn(instance))
                            }
                            _ => {
                                bug!("{} cannot be cast to a fn ptr", operand.layout.ty)
                            }
                        }
                    }
                    mir::CastKind::UnsafeFnPointer => {
                        // this is a no-op at the LLVM level
                        operand.val
                    }
                    mir::CastKind::Unsize => {
                        assert!(cast.is_llvm_scalar_pair());
                        match operand.val {
                            OperandValue::Pair(lldata, llextra) => {
                                // unsize from a fat pointer - this is a
                                // "trait-object-to-supertrait" coercion, for
                                // example,
                                //   &'a fmt::Debug+Send => &'a fmt::Debug,

                                // HACK(eddyb) have to bitcast pointers
                                // until LLVM removes pointee types.
                                let lldata = bx.pointercast(lldata,
                                    bx.cx().scalar_pair_element_backend_type(&cast, 0, true));
                                OperandValue::Pair(lldata, llextra)
                            }
                            OperandValue::Immediate(lldata) => {
                                // "standard" unsize
                                let (lldata, llextra) = base::unsize_thin_ptr(&bx, lldata,
                                    operand.layout.ty, cast.ty);
                                OperandValue::Pair(lldata, llextra)
                            }
                            OperandValue::Ref(..) => {
                                bug!("by-ref operand {:?} in codegen_rvalue_operand",
                                     operand);
                            }
                        }
                    }
                    mir::CastKind::Misc if operand.layout.is_llvm_scalar_pair() => {
                        if let OperandValue::Pair(data_ptr, meta) = operand.val {
                            if cast.is_llvm_scalar_pair() {
                                let data_cast = bx.pointercast(data_ptr,
                                    bx.cx().scalar_pair_element_backend_type(&cast, 0, true));
                                OperandValue::Pair(data_cast, meta)
                            } else { // cast to thin-ptr
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr and
                                // pointer-cast of that pointer to desired pointer type.
                                let llcast_ty = bx.cx().immediate_backend_type(&cast);
                                let llval = bx.pointercast(data_ptr, llcast_ty);
                                OperandValue::Immediate(llval)
                            }
                        } else {
                            bug!("Unexpected non-Pair operand")
                        }
                    }
                    mir::CastKind::Misc => {
                        assert!(cast.is_llvm_immediate());
                        let ll_t_out = bx.cx().immediate_backend_type(&cast);
                        if operand.layout.abi.is_uninhabited() {
                            let val = OperandValue::Immediate(bx.cx().const_undef(ll_t_out));
                            return (bx, OperandRef {
                                val,
                                layout: cast,
                            });
                        }
                        let r_t_in = CastTy::from_ty(operand.layout.ty)
                            .expect("bad input type for cast");
                        let r_t_out = CastTy::from_ty(cast.ty).expect("bad output type for cast");
                        let ll_t_in = bx.cx().immediate_backend_type(&operand.layout);
                        match operand.layout.variants {
                            layout::Variants::Single { index } => {
                                if let Some(def) = operand.layout.ty.ty_adt_def() {
                                    let discr_val = def
                                        .discriminant_for_variant(*bx.cx().tcx(), index)
                                        .val;
                                    let discr = bx.cx().const_uint_big(ll_t_out, discr_val);
                                    return (bx, OperandRef {
                                        val: OperandValue::Immediate(discr),
                                        layout: cast,
                                    });
                                }
                            }
                            layout::Variants::Tagged { .. } |
                            layout::Variants::NicheFilling { .. } => {},
                        }
                        let llval = operand.immediate();

                        let mut signed = false;
                        if let layout::Abi::Scalar(ref scalar) = operand.layout.abi {
                            if let layout::Int(_, s) = scalar.value {
                                // We use `i1` for bytes that are always `0` or `1`,
                                // e.g. `#[repr(i8)] enum E { A, B }`, but we can't
                                // let LLVM interpret the `i1` as signed, because
                                // then `i1 1` (i.e. E::B) is effectively `i8 -1`.
                                signed = !scalar.is_bool() && s;

                                let er = scalar.valid_range_exclusive(bx.cx());
                                if er.end != er.start &&
                                   scalar.valid_range.end() > scalar.valid_range.start() {
                                    // We want `table[e as usize]` to not
                                    // have bound checks, and this is the most
                                    // convenient place to put the `assume`.

                                    base::call_assume(&bx, bx.icmp(
                                        IntPredicate::IntULE,
                                        llval,
                                        bx.cx().const_uint_big(ll_t_in, *scalar.valid_range.end())
                                    ));
                                }
                            }
                        }

                        let newval = match (r_t_in, r_t_out) {
                            (CastTy::Int(_), CastTy::Int(_)) => {
                                bx.intcast(llval, ll_t_out, signed)
                            }
                            (CastTy::Float, CastTy::Float) => {
                                let srcsz = bx.cx().float_width(ll_t_in);
                                let dstsz = bx.cx().float_width(ll_t_out);
                                if dstsz > srcsz {
                                    bx.fpext(llval, ll_t_out)
                                } else if srcsz > dstsz {
                                    bx.fptrunc(llval, ll_t_out)
                                } else {
                                    llval
                                }
                            }
                            (CastTy::Ptr(_), CastTy::Ptr(_)) |
                            (CastTy::FnPtr, CastTy::Ptr(_)) |
                            (CastTy::RPtr(_), CastTy::Ptr(_)) =>
                                bx.pointercast(llval, ll_t_out),
                            (CastTy::Ptr(_), CastTy::Int(_)) |
                            (CastTy::FnPtr, CastTy::Int(_)) =>
                                bx.ptrtoint(llval, ll_t_out),
                            (CastTy::Int(_), CastTy::Ptr(_)) => {
                                let usize_llval = bx.intcast(llval, bx.cx().type_isize(), signed);
                                bx.inttoptr(usize_llval, ll_t_out)
                            }
                            (CastTy::Int(_), CastTy::Float) =>
                                cast_int_to_float(&bx, signed, llval, ll_t_in, ll_t_out),
                            (CastTy::Float, CastTy::Int(IntTy::I)) =>
                                cast_float_to_int(&bx, true, llval, ll_t_in, ll_t_out),
                            (CastTy::Float, CastTy::Int(_)) =>
                                cast_float_to_int(&bx, false, llval, ll_t_in, ll_t_out),
                            _ => bug!("unsupported cast: {:?} to {:?}", operand.layout.ty, cast.ty)
                        };
                        OperandValue::Immediate(newval)
                    }
                };
                (bx, OperandRef {
                    val,
                    layout: cast
                })
            }

            mir::Rvalue::Ref(_, bk, ref place) => {
                let cg_place = self.codegen_place(&bx, place);

                let ty = cg_place.layout.ty;

                // Note: places are indirect, so storing the `llval` into the
                // destination effectively creates a reference.
                let val = if !bx.cx().type_has_metadata(ty) {
                    OperandValue::Immediate(cg_place.llval)
                } else {
                    OperandValue::Pair(cg_place.llval, cg_place.llextra.unwrap())
                };
                (bx, OperandRef {
                    val,
                    layout: self.cx.layout_of(self.cx.tcx().mk_ref(
                        self.cx.tcx().types.re_erased,
                        ty::TypeAndMut { ty, mutbl: bk.to_mutbl_lossy() }
                    )),
                })
            }

            mir::Rvalue::Len(ref place) => {
                let size = self.evaluate_array_len(&bx, place);
                let operand = OperandRef {
                    val: OperandValue::Immediate(size),
                    layout: bx.cx().layout_of(bx.tcx().types.usize),
                };
                (bx, operand)
            }

            mir::Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.codegen_operand(&bx, lhs);
                let rhs = self.codegen_operand(&bx, rhs);
                let llresult = match (lhs.val, rhs.val) {
                    (OperandValue::Pair(lhs_addr, lhs_extra),
                     OperandValue::Pair(rhs_addr, rhs_extra)) => {
                        self.codegen_fat_ptr_binop(&bx, op,
                                                 lhs_addr, lhs_extra,
                                                 rhs_addr, rhs_extra,
                                                 lhs.layout.ty)
                    }

                    (OperandValue::Immediate(lhs_val),
                     OperandValue::Immediate(rhs_val)) => {
                        self.codegen_scalar_binop(&bx, op, lhs_val, rhs_val, lhs.layout.ty)
                    }

                    _ => bug!()
                };
                let operand = OperandRef {
                    val: OperandValue::Immediate(llresult),
                    layout: bx.cx().layout_of(
                        op.ty(bx.tcx(), lhs.layout.ty, rhs.layout.ty)),
                };
                (bx, operand)
            }
            mir::Rvalue::CheckedBinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.codegen_operand(&bx, lhs);
                let rhs = self.codegen_operand(&bx, rhs);
                let result = self.codegen_scalar_checked_binop(&bx, op,
                                                             lhs.immediate(), rhs.immediate(),
                                                             lhs.layout.ty);
                let val_ty = op.ty(bx.tcx(), lhs.layout.ty, rhs.layout.ty);
                let operand_ty = bx.tcx().intern_tup(&[val_ty, bx.tcx().types.bool]);
                let operand = OperandRef {
                    val: result,
                    layout: bx.cx().layout_of(operand_ty)
                };

                (bx, operand)
            }

            mir::Rvalue::UnaryOp(op, ref operand) => {
                let operand = self.codegen_operand(&bx, operand);
                let lloperand = operand.immediate();
                let is_float = operand.layout.ty.is_fp();
                let llval = match op {
                    mir::UnOp::Not => bx.not(lloperand),
                    mir::UnOp::Neg => if is_float {
                        bx.fneg(lloperand)
                    } else {
                        bx.neg(lloperand)
                    }
                };
                (bx, OperandRef {
                    val: OperandValue::Immediate(llval),
                    layout: operand.layout,
                })
            }

            mir::Rvalue::Discriminant(ref place) => {
                let discr_ty = rvalue.ty(&*self.mir, bx.tcx());
                let discr =  self.codegen_place(&bx, place)
                    .codegen_get_discr(&bx, discr_ty);
                (bx, OperandRef {
                    val: OperandValue::Immediate(discr),
                    layout: self.cx.layout_of(discr_ty)
                })
            }

            mir::Rvalue::NullaryOp(mir::NullOp::SizeOf, ty) => {
                assert!(bx.cx().type_is_sized(ty));
                let val = bx.cx().const_usize(bx.cx().layout_of(ty).size.bytes());
                let tcx = bx.tcx();
                (bx, OperandRef {
                    val: OperandValue::Immediate(val),
                    layout: self.cx.layout_of(tcx.types.usize),
                })
            }

            mir::Rvalue::NullaryOp(mir::NullOp::Box, content_ty) => {
                let content_ty: Ty<'tcx> = self.monomorphize(&content_ty);
                let (size, align) = bx.cx().layout_of(content_ty).size_and_align();
                let llsize = bx.cx().const_usize(size.bytes());
                let llalign = bx.cx().const_usize(align.abi());
                let box_layout = bx.cx().layout_of(bx.tcx().mk_box(content_ty));
                let llty_ptr = bx.cx().backend_type(&box_layout);

                // Allocate space:
                let def_id = match bx.tcx().lang_items().require(ExchangeMallocFnLangItem) {
                    Ok(id) => id,
                    Err(s) => {
                        bx.cx().sess().fatal(&format!("allocation of `{}` {}", box_layout.ty, s));
                    }
                };
                let instance = ty::Instance::mono(bx.tcx(), def_id);
                let r = bx.cx().get_fn(instance);
                let val = bx.pointercast(bx.call(r, &[llsize, llalign], None), llty_ptr);

                let operand = OperandRef {
                    val: OperandValue::Immediate(val),
                    layout: box_layout,
                };
                (bx, operand)
            }
            mir::Rvalue::Use(ref operand) => {
                let operand = self.codegen_operand(&bx, operand);
                (bx, operand)
            }
            mir::Rvalue::Repeat(..) |
            mir::Rvalue::Aggregate(..) => {
                // According to `rvalue_creates_operand`, only ZST
                // aggregate rvalues are allowed to be operands.
                let ty = rvalue.ty(self.mir, *self.cx.tcx());
                (bx, OperandRef::new_zst(self.cx,
                    self.cx.layout_of(self.monomorphize(&ty))))
            }
        }
    }

    fn evaluate_array_len<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(
        &mut self,
        bx: &Bx,
        place: &mir::Place<'tcx>,
    ) -> Cx::Value {
        // ZST are passed as operands and require special handling
        // because codegen_place() panics if Local is operand.
        if let mir::Place::Local(index) = *place {
            if let LocalRef::Operand(Some(op)) = self.locals[index] {
                if let ty::Array(_, n) = op.layout.ty.sty {
                    let n = n.unwrap_usize(*bx.cx().tcx());
                    return bx.cx().const_usize(n);
                }
            }
        }
        // use common size calculation for non zero-sized types
        let cg_value = self.codegen_place(bx, place);
        return cg_value.len(bx.cx());
    }

    pub fn codegen_scalar_binop<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(
        &mut self,
        bx: &Bx,
        op: mir::BinOp,
        lhs: Cx::Value,
        rhs: Cx::Value,
        input_ty: Ty<'tcx>,
    ) -> Cx::Value {
        let is_float = input_ty.is_fp();
        let is_signed = input_ty.is_signed();
        let is_unit = input_ty.is_unit();
        match op {
            mir::BinOp::Add => if is_float {
                bx.fadd(lhs, rhs)
            } else {
                bx.add(lhs, rhs)
            },
            mir::BinOp::Sub => if is_float {
                bx.fsub(lhs, rhs)
            } else {
                bx.sub(lhs, rhs)
            },
            mir::BinOp::Mul => if is_float {
                bx.fmul(lhs, rhs)
            } else {
                bx.mul(lhs, rhs)
            },
            mir::BinOp::Div => if is_float {
                bx.fdiv(lhs, rhs)
            } else if is_signed {
                bx.sdiv(lhs, rhs)
            } else {
                bx.udiv(lhs, rhs)
            },
            mir::BinOp::Rem => if is_float {
                bx.frem(lhs, rhs)
            } else if is_signed {
                bx.srem(lhs, rhs)
            } else {
                bx.urem(lhs, rhs)
            },
            mir::BinOp::BitOr => bx.or(lhs, rhs),
            mir::BinOp::BitAnd => bx.and(lhs, rhs),
            mir::BinOp::BitXor => bx.xor(lhs, rhs),
            mir::BinOp::Offset => bx.inbounds_gep(lhs, &[rhs]),
            mir::BinOp::Shl => common::build_unchecked_lshift(bx, lhs, rhs),
            mir::BinOp::Shr => common::build_unchecked_rshift(bx, input_ty, lhs, rhs),
            mir::BinOp::Ne | mir::BinOp::Lt | mir::BinOp::Gt |
            mir::BinOp::Eq | mir::BinOp::Le | mir::BinOp::Ge => if is_unit {
                bx.cx().const_bool(match op {
                    mir::BinOp::Ne | mir::BinOp::Lt | mir::BinOp::Gt => false,
                    mir::BinOp::Eq | mir::BinOp::Le | mir::BinOp::Ge => true,
                    _ => unreachable!()
                })
            } else if is_float {
                bx.fcmp(
                    base::bin_op_to_fcmp_predicate(op.to_hir_binop()),
                    lhs, rhs
                )
            } else {
                bx.icmp(
                    base::bin_op_to_icmp_predicate(op.to_hir_binop(), is_signed),
                    lhs, rhs
                )
            }
        }
    }

    pub fn codegen_fat_ptr_binop<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(
        &mut self,
        bx: &Bx,
        op: mir::BinOp,
        lhs_addr: Cx::Value,
        lhs_extra: Cx::Value,
        rhs_addr: Cx::Value,
        rhs_extra: Cx::Value,
        _input_ty: Ty<'tcx>,
    ) -> Cx::Value {
        match op {
            mir::BinOp::Eq => {
                bx.and(
                    bx.icmp(IntPredicate::IntEQ, lhs_addr, rhs_addr),
                    bx.icmp(IntPredicate::IntEQ, lhs_extra, rhs_extra)
                )
            }
            mir::BinOp::Ne => {
                bx.or(
                    bx.icmp(IntPredicate::IntNE, lhs_addr, rhs_addr),
                    bx.icmp(IntPredicate::IntNE, lhs_extra, rhs_extra)
                )
            }
            mir::BinOp::Le | mir::BinOp::Lt |
            mir::BinOp::Ge | mir::BinOp::Gt => {
                // a OP b ~ a.0 STRICT(OP) b.0 | (a.0 == b.0 && a.1 OP a.1)
                let (op, strict_op) = match op {
                    mir::BinOp::Lt => (IntPredicate::IntULT, IntPredicate::IntULT),
                    mir::BinOp::Le => (IntPredicate::IntULE, IntPredicate::IntULT),
                    mir::BinOp::Gt => (IntPredicate::IntUGT, IntPredicate::IntUGT),
                    mir::BinOp::Ge => (IntPredicate::IntUGE, IntPredicate::IntUGT),
                    _ => bug!(),
                };

                bx.or(
                    bx.icmp(strict_op, lhs_addr, rhs_addr),
                    bx.and(
                        bx.icmp(IntPredicate::IntEQ, lhs_addr, rhs_addr),
                        bx.icmp(op, lhs_extra, rhs_extra)
                    )
                )
            }
            _ => {
                bug!("unexpected fat ptr binop");
            }
        }
    }

    pub fn codegen_scalar_checked_binop<Bx: BuilderMethods<'a, 'll, 'tcx, CodegenCx=Cx>>(
        &mut self,
        bx: &Bx,
        op: mir::BinOp,
        lhs: Cx::Value,
        rhs: Cx::Value,
        input_ty: Ty<'tcx>
    ) -> OperandValue<Cx::Value> {
        // This case can currently arise only from functions marked
        // with #[rustc_inherit_overflow_checks] and inlined from
        // another crate (mostly core::num generic/#[inline] fns),
        // while the current crate doesn't use overflow checks.
        if !bx.cx().check_overflow() {
            let val = self.codegen_scalar_binop(bx, op, lhs, rhs, input_ty);
            return OperandValue::Pair(val, bx.cx().const_bool(false));
        }

        let (val, of) = match op {
            // These are checked using intrinsics
            mir::BinOp::Add | mir::BinOp::Sub | mir::BinOp::Mul => {
                let oop = match op {
                    mir::BinOp::Add => OverflowOp::Add,
                    mir::BinOp::Sub => OverflowOp::Sub,
                    mir::BinOp::Mul => OverflowOp::Mul,
                    _ => unreachable!()
                };
                let intrinsic = get_overflow_intrinsic(oop, bx, input_ty);
                let res = bx.call(intrinsic, &[lhs, rhs], None);

                (bx.extract_value(res, 0),
                 bx.extract_value(res, 1))
            }
            mir::BinOp::Shl | mir::BinOp::Shr => {
                let lhs_llty = bx.cx().val_ty(lhs);
                let rhs_llty = bx.cx().val_ty(rhs);
                let invert_mask = common::shift_mask_val(bx, lhs_llty, rhs_llty, true);
                let outer_bits = bx.and(rhs, invert_mask);

                let of = bx.icmp(IntPredicate::IntNE, outer_bits, bx.cx().const_null(rhs_llty));
                let val = self.codegen_scalar_binop(bx, op, lhs, rhs, input_ty);

                (val, of)
            }
            _ => {
                bug!("Operator `{:?}` is not a checkable operator", op)
            }
        };

        OperandValue::Pair(val, of)
    }
}

impl<'a, 'f, 'll: 'a + 'f, 'tcx: 'll, Cx: 'a + CodegenMethods<'ll, 'tcx>>
    FunctionCx<'a, 'f, 'll, 'tcx, Cx> where
    &'a Cx : LayoutOf<Ty=Ty<'tcx>, TyLayout=TyLayout<'tcx>> + HasTyCtxt<'tcx>
{
    pub fn rvalue_creates_operand(&self, rvalue: &mir::Rvalue<'tcx>) -> bool {
        match *rvalue {
            mir::Rvalue::Ref(..) |
            mir::Rvalue::Len(..) |
            mir::Rvalue::Cast(..) | // (*)
            mir::Rvalue::BinaryOp(..) |
            mir::Rvalue::CheckedBinaryOp(..) |
            mir::Rvalue::UnaryOp(..) |
            mir::Rvalue::Discriminant(..) |
            mir::Rvalue::NullaryOp(..) |
            mir::Rvalue::Use(..) => // (*)
                true,
            mir::Rvalue::Repeat(..) |
            mir::Rvalue::Aggregate(..) => {
                let ty = rvalue.ty(self.mir, *self.cx.tcx());
                let ty = self.monomorphize(&ty);
                self.cx.layout_of(ty).is_zst()
            }
        }

        // (*) this is only true if the type is suitable
    }
}

#[derive(Copy, Clone)]
enum OverflowOp {
    Add, Sub, Mul
}

fn get_overflow_intrinsic<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    oop: OverflowOp,
    bx: &Bx,
    ty: Ty
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    use syntax::ast::IntTy::*;
    use syntax::ast::UintTy::*;
    use rustc::ty::{Int, Uint};

    let tcx = bx.tcx();

    let new_sty = match ty.sty {
        Int(Isize) => Int(tcx.sess.target.isize_ty),
        Uint(Usize) => Uint(tcx.sess.target.usize_ty),
        ref t @ Uint(_) | ref t @ Int(_) => t.clone(),
        _ => panic!("tried to get overflow intrinsic for op applied to non-int type")
    };

    let name = match oop {
        OverflowOp::Add => match new_sty {
            Int(I8) => "llvm.sadd.with.overflow.i8",
            Int(I16) => "llvm.sadd.with.overflow.i16",
            Int(I32) => "llvm.sadd.with.overflow.i32",
            Int(I64) => "llvm.sadd.with.overflow.i64",
            Int(I128) => "llvm.sadd.with.overflow.i128",

            Uint(U8) => "llvm.uadd.with.overflow.i8",
            Uint(U16) => "llvm.uadd.with.overflow.i16",
            Uint(U32) => "llvm.uadd.with.overflow.i32",
            Uint(U64) => "llvm.uadd.with.overflow.i64",
            Uint(U128) => "llvm.uadd.with.overflow.i128",

            _ => unreachable!(),
        },
        OverflowOp::Sub => match new_sty {
            Int(I8) => "llvm.ssub.with.overflow.i8",
            Int(I16) => "llvm.ssub.with.overflow.i16",
            Int(I32) => "llvm.ssub.with.overflow.i32",
            Int(I64) => "llvm.ssub.with.overflow.i64",
            Int(I128) => "llvm.ssub.with.overflow.i128",

            Uint(U8) => "llvm.usub.with.overflow.i8",
            Uint(U16) => "llvm.usub.with.overflow.i16",
            Uint(U32) => "llvm.usub.with.overflow.i32",
            Uint(U64) => "llvm.usub.with.overflow.i64",
            Uint(U128) => "llvm.usub.with.overflow.i128",

            _ => unreachable!(),
        },
        OverflowOp::Mul => match new_sty {
            Int(I8) => "llvm.smul.with.overflow.i8",
            Int(I16) => "llvm.smul.with.overflow.i16",
            Int(I32) => "llvm.smul.with.overflow.i32",
            Int(I64) => "llvm.smul.with.overflow.i64",
            Int(I128) => "llvm.smul.with.overflow.i128",

            Uint(U8) => "llvm.umul.with.overflow.i8",
            Uint(U16) => "llvm.umul.with.overflow.i16",
            Uint(U32) => "llvm.umul.with.overflow.i32",
            Uint(U64) => "llvm.umul.with.overflow.i64",
            Uint(U128) => "llvm.umul.with.overflow.i128",

            _ => unreachable!(),
        },
    };

    bx.cx().get_intrinsic(&name)
}

fn cast_int_to_float<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    bx: &Bx,
    signed: bool,
    x: <Bx::CodegenCx as Backend<'ll>>::Value,
    int_ty: <Bx::CodegenCx as Backend<'ll>>::Type,
    float_ty: <Bx::CodegenCx as Backend<'ll>>::Type
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    // Most integer types, even i128, fit into [-f32::MAX, f32::MAX] after rounding.
    // It's only u128 -> f32 that can cause overflows (i.e., should yield infinity).
    // LLVM's uitofp produces undef in those cases, so we manually check for that case.
    let is_u128_to_f32 = !signed &&
        bx.cx().int_width(int_ty) == 128 &&
        bx.cx().float_width(float_ty) == 32;
    if is_u128_to_f32 {
        // All inputs greater or equal to (f32::MAX + 0.5 ULP) are rounded to infinity,
        // and for everything else LLVM's uitofp works just fine.
        use rustc_apfloat::ieee::Single;
        use rustc_apfloat::Float;
        const MAX_F32_PLUS_HALF_ULP: u128 = ((1 << (Single::PRECISION + 1)) - 1)
                                            << (Single::MAX_EXP - Single::PRECISION as i16);
        let max = bx.cx().const_uint_big(int_ty, MAX_F32_PLUS_HALF_ULP);
        let overflow = bx.icmp(IntPredicate::IntUGE, x, max);
        let infinity_bits = bx.cx().const_u32(ieee::Single::INFINITY.to_bits() as u32);
        let infinity = bx.bitcast(infinity_bits, float_ty);
        bx.select(overflow, infinity, bx.uitofp(x, float_ty))
    } else {
        if signed {
            bx.sitofp(x, float_ty)
        } else {
            bx.uitofp(x, float_ty)
        }
    }
}

fn cast_float_to_int<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    bx: &Bx,
    signed: bool,
    x: <Bx::CodegenCx as Backend<'ll>>::Value,
    float_ty: <Bx::CodegenCx as Backend<'ll>>::Type,
    int_ty: <Bx::CodegenCx as Backend<'ll>>::Type
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    let fptosui_result = if signed {
        bx.fptosi(x, int_ty)
    } else {
        bx.fptoui(x, int_ty)
    };

    if !bx.cx().sess().opts.debugging_opts.saturating_float_casts {
        return fptosui_result;
    }
    // LLVM's fpto[su]i returns undef when the input x is infinite, NaN, or does not fit into the
    // destination integer type after rounding towards zero. This `undef` value can cause UB in
    // safe code (see issue #10184), so we implement a saturating conversion on top of it:
    // Semantically, the mathematical value of the input is rounded towards zero to the next
    // mathematical integer, and then the result is clamped into the range of the destination
    // integer type. Positive and negative infinity are mapped to the maximum and minimum value of
    // the destination integer type. NaN is mapped to 0.
    //
    // Define f_min and f_max as the largest and smallest (finite) floats that are exactly equal to
    // a value representable in int_ty.
    // They are exactly equal to int_ty::{MIN,MAX} if float_ty has enough significand bits.
    // Otherwise, int_ty::MAX must be rounded towards zero, as it is one less than a power of two.
    // int_ty::MIN, however, is either zero or a negative power of two and is thus exactly
    // representable. Note that this only works if float_ty's exponent range is sufficiently large.
    // f16 or 256 bit integers would break this property. Right now the smallest float type is f32
    // with exponents ranging up to 127, which is barely enough for i128::MIN = -2^127.
    // On the other hand, f_max works even if int_ty::MAX is greater than float_ty::MAX. Because
    // we're rounding towards zero, we just get float_ty::MAX (which is always an integer).
    // This already happens today with u128::MAX = 2^128 - 1 > f32::MAX.
    let int_max = |signed: bool, int_ty: <Bx::CodegenCx as Backend<'ll>>::Type| -> u128 {
        let shift_amount = 128 - bx.cx().int_width(int_ty);
        if signed {
            i128::MAX as u128 >> shift_amount
        } else {
            u128::MAX >> shift_amount
        }
    };
    let int_min = |signed: bool, int_ty: <Bx::CodegenCx as Backend<'ll>>::Type| -> i128 {
        if signed {
            i128::MIN >> (128 - bx.cx().int_width(int_ty))
        } else {
            0
        }
    };

    let compute_clamp_bounds_single =
    |signed: bool, int_ty: <Bx::CodegenCx as Backend<'ll>>::Type| -> (u128, u128) {
        let rounded_min = ieee::Single::from_i128_r(int_min(signed, int_ty), Round::TowardZero);
        assert_eq!(rounded_min.status, Status::OK);
        let rounded_max = ieee::Single::from_u128_r(int_max(signed, int_ty), Round::TowardZero);
        assert!(rounded_max.value.is_finite());
        (rounded_min.value.to_bits(), rounded_max.value.to_bits())
    };
    let compute_clamp_bounds_double =
    |signed: bool, int_ty: <Bx::CodegenCx as Backend<'ll>>::Type| -> (u128, u128) {
        let rounded_min = ieee::Double::from_i128_r(int_min(signed, int_ty), Round::TowardZero);
        assert_eq!(rounded_min.status, Status::OK);
        let rounded_max = ieee::Double::from_u128_r(int_max(signed, int_ty), Round::TowardZero);
        assert!(rounded_max.value.is_finite());
        (rounded_min.value.to_bits(), rounded_max.value.to_bits())
    };

    let float_bits_to_llval = |bits| {
        let bits_llval = match bx.cx().float_width(float_ty) {
            32 => bx.cx().const_u32(bits as u32),
            64 => bx.cx().const_u64(bits as u64),
            n => bug!("unsupported float width {}", n),
        };
        bx.bitcast(bits_llval, float_ty)
    };
    let (f_min, f_max) = match bx.cx().float_width(float_ty) {
        32 => compute_clamp_bounds_single(signed, int_ty),
        64 => compute_clamp_bounds_double(signed, int_ty),
        n => bug!("unsupported float width {}", n),
    };
    let f_min = float_bits_to_llval(f_min);
    let f_max = float_bits_to_llval(f_max);
    // To implement saturation, we perform the following steps:
    //
    // 1. Cast x to an integer with fpto[su]i. This may result in undef.
    // 2. Compare x to f_min and f_max, and use the comparison results to select:
    //  a) int_ty::MIN if x < f_min or x is NaN
    //  b) int_ty::MAX if x > f_max
    //  c) the result of fpto[su]i otherwise
    // 3. If x is NaN, return 0.0, otherwise return the result of step 2.
    //
    // This avoids resulting undef because values in range [f_min, f_max] by definition fit into the
    // destination type. It creates an undef temporary, but *producing* undef is not UB. Our use of
    // undef does not introduce any non-determinism either.
    // More importantly, the above procedure correctly implements saturating conversion.
    // Proof (sketch):
    // If x is NaN, 0 is returned by definition.
    // Otherwise, x is finite or infinite and thus can be compared with f_min and f_max.
    // This yields three cases to consider:
    // (1) if x in [f_min, f_max], the result of fpto[su]i is returned, which agrees with
    //     saturating conversion for inputs in that range.
    // (2) if x > f_max, then x is larger than int_ty::MAX. This holds even if f_max is rounded
    //     (i.e., if f_max < int_ty::MAX) because in those cases, nextUp(f_max) is already larger
    //     than int_ty::MAX. Because x is larger than int_ty::MAX, the return value of int_ty::MAX
    //     is correct.
    // (3) if x < f_min, then x is smaller than int_ty::MIN. As shown earlier, f_min exactly equals
    //     int_ty::MIN and therefore the return value of int_ty::MIN is correct.
    // QED.

    // Step 1 was already performed above.

    // Step 2: We use two comparisons and two selects, with %s1 being the result:
    //     %less_or_nan = fcmp ult %x, %f_min
    //     %greater = fcmp olt %x, %f_max
    //     %s0 = select %less_or_nan, int_ty::MIN, %fptosi_result
    //     %s1 = select %greater, int_ty::MAX, %s0
    // Note that %less_or_nan uses an *unordered* comparison. This comparison is true if the
    // operands are not comparable (i.e., if x is NaN). The unordered comparison ensures that s1
    // becomes int_ty::MIN if x is NaN.
    // Performance note: Unordered comparison can be lowered to a "flipped" comparison and a
    // negation, and the negation can be merged into the select. Therefore, it not necessarily any
    // more expensive than a ordered ("normal") comparison. Whether these optimizations will be
    // performed is ultimately up to the backend, but at least x86 does perform them.
    let less_or_nan = bx.fcmp(RealPredicate::RealULT, x, f_min);
    let greater = bx.fcmp(RealPredicate::RealOGT, x, f_max);
    let int_max = bx.cx().const_uint_big(int_ty, int_max(signed, int_ty));
    let int_min = bx.cx().const_uint_big(int_ty, int_min(signed, int_ty) as u128);
    let s0 = bx.select(less_or_nan, int_min, fptosui_result);
    let s1 = bx.select(greater, int_max, s0);

    // Step 3: NaN replacement.
    // For unsigned types, the above step already yielded int_ty::MIN == 0 if x is NaN.
    // Therefore we only need to execute this step for signed integer types.
    if signed {
        // LLVM has no isNaN predicate, so we use (x == x) instead
        bx.select(bx.fcmp(RealPredicate::RealOEQ, x, x), s1, bx.cx().const_uint(int_ty, 0))
    } else {
        s1
    }
}
