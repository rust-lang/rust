// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, ValueRef};
use rustc::ty::{self, Ty};
use rustc::ty::cast::{CastTy, IntTy};
use rustc::ty::layout::{self, LayoutOf};
use rustc::mir;
use rustc::middle::lang_items::ExchangeMallocFnLangItem;
use rustc_apfloat::{ieee, Float, Status, Round};
use rustc_const_math::MAX_F32_PLUS_HALF_ULP;
use std::{u128, i128};

use base;
use builder::Builder;
use callee;
use common::{self, val_ty};
use common::{C_bool, C_u8, C_i32, C_u32, C_u64, C_null, C_usize, C_uint, C_uint_big};
use consts;
use monomorphize;
use type_::Type;
use type_of::LayoutLlvmExt;
use value::Value;

use super::{MirContext, LocalRef};
use super::constant::const_scalar_checked_binop;
use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_rvalue(&mut self,
                        bcx: Builder<'a, 'tcx>,
                        dest: PlaceRef<'tcx>,
                        rvalue: &mir::Rvalue<'tcx>)
                        -> Builder<'a, 'tcx>
    {
        debug!("trans_rvalue(dest.llval={:?}, rvalue={:?})",
               Value(dest.llval), rvalue);

        match *rvalue {
           mir::Rvalue::Use(ref operand) => {
               let tr_operand = self.trans_operand(&bcx, operand);
               // FIXME: consider not copying constants through stack. (fixable by translating
               // constants into OperandValue::Ref, why don’t we do that yet if we don’t?)
               tr_operand.val.store(&bcx, dest);
               bcx
           }

            mir::Rvalue::Cast(mir::CastKind::Unsize, ref source, _) => {
                // The destination necessarily contains a fat pointer, so if
                // it's a scalar pair, it's a fat pointer or newtype thereof.
                if dest.layout.is_llvm_scalar_pair() {
                    // into-coerce of a thin pointer to a fat pointer - just
                    // use the operand path.
                    let (bcx, temp) = self.trans_rvalue_operand(bcx, rvalue);
                    temp.val.store(&bcx, dest);
                    return bcx;
                }

                // Unsize of a nontrivial struct. I would prefer for
                // this to be eliminated by MIR translation, but
                // `CoerceUnsized` can be passed by a where-clause,
                // so the (generic) MIR may not be able to expand it.
                let operand = self.trans_operand(&bcx, source);
                match operand.val {
                    OperandValue::Pair(..) |
                    OperandValue::Immediate(_) => {
                        // unsize from an immediate structure. We don't
                        // really need a temporary alloca here, but
                        // avoiding it would require us to have
                        // `coerce_unsized_into` use extractvalue to
                        // index into the struct, and this case isn't
                        // important enough for it.
                        debug!("trans_rvalue: creating ugly alloca");
                        let scratch = PlaceRef::alloca(&bcx, operand.layout, "__unsize_temp");
                        scratch.storage_live(&bcx);
                        operand.val.store(&bcx, scratch);
                        base::coerce_unsized_into(&bcx, scratch, dest);
                        scratch.storage_dead(&bcx);
                    }
                    OperandValue::Ref(llref, align) => {
                        let source = PlaceRef::new_sized(llref, operand.layout, align);
                        base::coerce_unsized_into(&bcx, source, dest);
                    }
                }
                bcx
            }

            mir::Rvalue::Repeat(ref elem, count) => {
                let tr_elem = self.trans_operand(&bcx, elem);

                // Do not generate the loop for zero-sized elements or empty arrays.
                if dest.layout.is_zst() {
                    return bcx;
                }

                let start = dest.project_index(&bcx, C_usize(bcx.ccx, 0)).llval;

                if let OperandValue::Immediate(v) = tr_elem.val {
                    let align = C_i32(bcx.ccx, dest.align.abi() as i32);
                    let size = C_usize(bcx.ccx, dest.layout.size.bytes());

                    // Use llvm.memset.p0i8.* to initialize all zero arrays
                    if common::is_const_integral(v) && common::const_to_uint(v) == 0 {
                        let fill = C_u8(bcx.ccx, 0);
                        base::call_memset(&bcx, start, fill, size, align, false);
                        return bcx;
                    }

                    // Use llvm.memset.p0i8.* to initialize byte arrays
                    let v = base::from_immediate(&bcx, v);
                    if common::val_ty(v) == Type::i8(bcx.ccx) {
                        base::call_memset(&bcx, start, v, size, align, false);
                        return bcx;
                    }
                }

                let count = count.as_u64();
                let count = C_usize(bcx.ccx, count);
                let end = dest.project_index(&bcx, count).llval;

                let header_bcx = bcx.build_sibling_block("repeat_loop_header");
                let body_bcx = bcx.build_sibling_block("repeat_loop_body");
                let next_bcx = bcx.build_sibling_block("repeat_loop_next");

                bcx.br(header_bcx.llbb());
                let current = header_bcx.phi(common::val_ty(start), &[start], &[bcx.llbb()]);

                let keep_going = header_bcx.icmp(llvm::IntNE, current, end);
                header_bcx.cond_br(keep_going, body_bcx.llbb(), next_bcx.llbb());

                tr_elem.val.store(&body_bcx,
                    PlaceRef::new_sized(current, tr_elem.layout, dest.align));

                let next = body_bcx.inbounds_gep(current, &[C_usize(bcx.ccx, 1)]);
                body_bcx.br(header_bcx.llbb());
                header_bcx.add_incoming_to_phi(current, next, body_bcx.llbb());

                next_bcx
            }

            mir::Rvalue::Aggregate(ref kind, ref operands) => {
                let (dest, active_field_index) = match **kind {
                    mir::AggregateKind::Adt(adt_def, variant_index, _, active_field_index) => {
                        dest.trans_set_discr(&bcx, variant_index);
                        if adt_def.is_enum() {
                            (dest.project_downcast(&bcx, variant_index), active_field_index)
                        } else {
                            (dest, active_field_index)
                        }
                    }
                    _ => (dest, None)
                };
                for (i, operand) in operands.iter().enumerate() {
                    let op = self.trans_operand(&bcx, operand);
                    // Do not generate stores and GEPis for zero-sized fields.
                    if !op.layout.is_zst() {
                        let field_index = active_field_index.unwrap_or(i);
                        op.val.store(&bcx, dest.project_field(&bcx, field_index));
                    }
                }
                bcx
            }

            _ => {
                assert!(self.rvalue_creates_operand(rvalue));
                let (bcx, temp) = self.trans_rvalue_operand(bcx, rvalue);
                temp.val.store(&bcx, dest);
                bcx
            }
        }
    }

    pub fn trans_rvalue_operand(&mut self,
                                bcx: Builder<'a, 'tcx>,
                                rvalue: &mir::Rvalue<'tcx>)
                                -> (Builder<'a, 'tcx>, OperandRef<'tcx>)
    {
        assert!(self.rvalue_creates_operand(rvalue), "cannot trans {:?} to operand", rvalue);

        match *rvalue {
            mir::Rvalue::Cast(ref kind, ref source, mir_cast_ty) => {
                let operand = self.trans_operand(&bcx, source);
                debug!("cast operand is {:?}", operand);
                let cast = bcx.ccx.layout_of(self.monomorphize(&mir_cast_ty));

                let val = match *kind {
                    mir::CastKind::ReifyFnPointer => {
                        match operand.layout.ty.sty {
                            ty::TyFnDef(def_id, substs) => {
                                OperandValue::Immediate(
                                    callee::resolve_and_get_fn(bcx.ccx, def_id, substs))
                            }
                            _ => {
                                bug!("{} cannot be reified to a fn ptr", operand.layout.ty)
                            }
                        }
                    }
                    mir::CastKind::ClosureFnPointer => {
                        match operand.layout.ty.sty {
                            ty::TyClosure(def_id, substs) => {
                                let instance = monomorphize::resolve_closure(
                                    bcx.ccx.tcx(), def_id, substs, ty::ClosureKind::FnOnce);
                                OperandValue::Immediate(callee::get_fn(bcx.ccx, instance))
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
                                let lldata = bcx.pointercast(lldata,
                                    cast.scalar_pair_element_llvm_type(bcx.ccx, 0));
                                OperandValue::Pair(lldata, llextra)
                            }
                            OperandValue::Immediate(lldata) => {
                                // "standard" unsize
                                let (lldata, llextra) = base::unsize_thin_ptr(&bcx, lldata,
                                    operand.layout.ty, cast.ty);
                                OperandValue::Pair(lldata, llextra)
                            }
                            OperandValue::Ref(..) => {
                                bug!("by-ref operand {:?} in trans_rvalue_operand",
                                     operand);
                            }
                        }
                    }
                    mir::CastKind::Misc if operand.layout.is_llvm_scalar_pair() => {
                        if let OperandValue::Pair(data_ptr, meta) = operand.val {
                            if cast.is_llvm_scalar_pair() {
                                let data_cast = bcx.pointercast(data_ptr,
                                    cast.scalar_pair_element_llvm_type(bcx.ccx, 0));
                                OperandValue::Pair(data_cast, meta)
                            } else { // cast to thin-ptr
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr and
                                // pointer-cast of that pointer to desired pointer type.
                                let llcast_ty = cast.immediate_llvm_type(bcx.ccx);
                                let llval = bcx.pointercast(data_ptr, llcast_ty);
                                OperandValue::Immediate(llval)
                            }
                        } else {
                            bug!("Unexpected non-Pair operand")
                        }
                    }
                    mir::CastKind::Misc => {
                        assert!(cast.is_llvm_immediate());
                        let r_t_in = CastTy::from_ty(operand.layout.ty)
                            .expect("bad input type for cast");
                        let r_t_out = CastTy::from_ty(cast.ty).expect("bad output type for cast");
                        let ll_t_in = operand.layout.immediate_llvm_type(bcx.ccx);
                        let ll_t_out = cast.immediate_llvm_type(bcx.ccx);
                        let llval = operand.immediate();

                        let mut signed = false;
                        if let layout::Abi::Scalar(ref scalar) = operand.layout.abi {
                            if let layout::Int(_, s) = scalar.value {
                                signed = s;

                                if scalar.valid_range.end > scalar.valid_range.start {
                                    // We want `table[e as usize]` to not
                                    // have bound checks, and this is the most
                                    // convenient place to put the `assume`.

                                    base::call_assume(&bcx, bcx.icmp(
                                        llvm::IntULE,
                                        llval,
                                        C_uint_big(ll_t_in, scalar.valid_range.end)
                                    ));
                                }
                            }
                        }

                        let newval = match (r_t_in, r_t_out) {
                            (CastTy::Int(_), CastTy::Int(_)) => {
                                bcx.intcast(llval, ll_t_out, signed)
                            }
                            (CastTy::Float, CastTy::Float) => {
                                let srcsz = ll_t_in.float_width();
                                let dstsz = ll_t_out.float_width();
                                if dstsz > srcsz {
                                    bcx.fpext(llval, ll_t_out)
                                } else if srcsz > dstsz {
                                    bcx.fptrunc(llval, ll_t_out)
                                } else {
                                    llval
                                }
                            }
                            (CastTy::Ptr(_), CastTy::Ptr(_)) |
                            (CastTy::FnPtr, CastTy::Ptr(_)) |
                            (CastTy::RPtr(_), CastTy::Ptr(_)) =>
                                bcx.pointercast(llval, ll_t_out),
                            (CastTy::Ptr(_), CastTy::Int(_)) |
                            (CastTy::FnPtr, CastTy::Int(_)) =>
                                bcx.ptrtoint(llval, ll_t_out),
                            (CastTy::Int(_), CastTy::Ptr(_)) =>
                                bcx.inttoptr(llval, ll_t_out),
                            (CastTy::Int(_), CastTy::Float) =>
                                cast_int_to_float(&bcx, signed, llval, ll_t_in, ll_t_out),
                            (CastTy::Float, CastTy::Int(IntTy::I)) =>
                                cast_float_to_int(&bcx, true, llval, ll_t_in, ll_t_out),
                            (CastTy::Float, CastTy::Int(_)) =>
                                cast_float_to_int(&bcx, false, llval, ll_t_in, ll_t_out),
                            _ => bug!("unsupported cast: {:?} to {:?}", operand.layout.ty, cast.ty)
                        };
                        OperandValue::Immediate(newval)
                    }
                };
                (bcx, OperandRef {
                    val,
                    layout: cast
                })
            }

            mir::Rvalue::Ref(_, bk, ref place) => {
                let tr_place = self.trans_place(&bcx, place);

                let ty = tr_place.layout.ty;

                // Note: places are indirect, so storing the `llval` into the
                // destination effectively creates a reference.
                let val = if !bcx.ccx.shared().type_has_metadata(ty) {
                    OperandValue::Immediate(tr_place.llval)
                } else {
                    OperandValue::Pair(tr_place.llval, tr_place.llextra)
                };
                (bcx, OperandRef {
                    val,
                    layout: self.ccx.layout_of(self.ccx.tcx().mk_ref(
                        self.ccx.tcx().types.re_erased,
                        ty::TypeAndMut { ty, mutbl: bk.to_mutbl_lossy() }
                    )),
                })
            }

            mir::Rvalue::Len(ref place) => {
                let size = self.evaluate_array_len(&bcx, place);
                let operand = OperandRef {
                    val: OperandValue::Immediate(size),
                    layout: bcx.ccx.layout_of(bcx.tcx().types.usize),
                };
                (bcx, operand)
            }

            mir::Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.trans_operand(&bcx, lhs);
                let rhs = self.trans_operand(&bcx, rhs);
                let llresult = match (lhs.val, rhs.val) {
                    (OperandValue::Pair(lhs_addr, lhs_extra),
                     OperandValue::Pair(rhs_addr, rhs_extra)) => {
                        self.trans_fat_ptr_binop(&bcx, op,
                                                 lhs_addr, lhs_extra,
                                                 rhs_addr, rhs_extra,
                                                 lhs.layout.ty)
                    }

                    (OperandValue::Immediate(lhs_val),
                     OperandValue::Immediate(rhs_val)) => {
                        self.trans_scalar_binop(&bcx, op, lhs_val, rhs_val, lhs.layout.ty)
                    }

                    _ => bug!()
                };
                let operand = OperandRef {
                    val: OperandValue::Immediate(llresult),
                    layout: bcx.ccx.layout_of(
                        op.ty(bcx.tcx(), lhs.layout.ty, rhs.layout.ty)),
                };
                (bcx, operand)
            }
            mir::Rvalue::CheckedBinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.trans_operand(&bcx, lhs);
                let rhs = self.trans_operand(&bcx, rhs);
                let result = self.trans_scalar_checked_binop(&bcx, op,
                                                             lhs.immediate(), rhs.immediate(),
                                                             lhs.layout.ty);
                let val_ty = op.ty(bcx.tcx(), lhs.layout.ty, rhs.layout.ty);
                let operand_ty = bcx.tcx().intern_tup(&[val_ty, bcx.tcx().types.bool], false);
                let operand = OperandRef {
                    val: result,
                    layout: bcx.ccx.layout_of(operand_ty)
                };

                (bcx, operand)
            }

            mir::Rvalue::UnaryOp(op, ref operand) => {
                let operand = self.trans_operand(&bcx, operand);
                let lloperand = operand.immediate();
                let is_float = operand.layout.ty.is_fp();
                let llval = match op {
                    mir::UnOp::Not => bcx.not(lloperand),
                    mir::UnOp::Neg => if is_float {
                        bcx.fneg(lloperand)
                    } else {
                        bcx.neg(lloperand)
                    }
                };
                (bcx, OperandRef {
                    val: OperandValue::Immediate(llval),
                    layout: operand.layout,
                })
            }

            mir::Rvalue::Discriminant(ref place) => {
                let discr_ty = rvalue.ty(&*self.mir, bcx.tcx());
                let discr =  self.trans_place(&bcx, place)
                    .trans_get_discr(&bcx, discr_ty);
                (bcx, OperandRef {
                    val: OperandValue::Immediate(discr),
                    layout: self.ccx.layout_of(discr_ty)
                })
            }

            mir::Rvalue::NullaryOp(mir::NullOp::SizeOf, ty) => {
                assert!(bcx.ccx.shared().type_is_sized(ty));
                let val = C_usize(bcx.ccx, bcx.ccx.size_of(ty).bytes());
                let tcx = bcx.tcx();
                (bcx, OperandRef {
                    val: OperandValue::Immediate(val),
                    layout: self.ccx.layout_of(tcx.types.usize),
                })
            }

            mir::Rvalue::NullaryOp(mir::NullOp::Box, content_ty) => {
                let content_ty: Ty<'tcx> = self.monomorphize(&content_ty);
                let (size, align) = bcx.ccx.size_and_align_of(content_ty);
                let llsize = C_usize(bcx.ccx, size.bytes());
                let llalign = C_usize(bcx.ccx, align.abi());
                let box_layout = bcx.ccx.layout_of(bcx.tcx().mk_box(content_ty));
                let llty_ptr = box_layout.llvm_type(bcx.ccx);

                // Allocate space:
                let def_id = match bcx.tcx().lang_items().require(ExchangeMallocFnLangItem) {
                    Ok(id) => id,
                    Err(s) => {
                        bcx.sess().fatal(&format!("allocation of `{}` {}", box_layout.ty, s));
                    }
                };
                let instance = ty::Instance::mono(bcx.tcx(), def_id);
                let r = callee::get_fn(bcx.ccx, instance);
                let val = bcx.pointercast(bcx.call(r, &[llsize, llalign], None), llty_ptr);

                let operand = OperandRef {
                    val: OperandValue::Immediate(val),
                    layout: box_layout,
                };
                (bcx, operand)
            }
            mir::Rvalue::Use(ref operand) => {
                let operand = self.trans_operand(&bcx, operand);
                (bcx, operand)
            }
            mir::Rvalue::Repeat(..) |
            mir::Rvalue::Aggregate(..) => {
                // According to `rvalue_creates_operand`, only ZST
                // aggregate rvalues are allowed to be operands.
                let ty = rvalue.ty(self.mir, self.ccx.tcx());
                (bcx, OperandRef::new_zst(self.ccx,
                    self.ccx.layout_of(self.monomorphize(&ty))))
            }
        }
    }

    fn evaluate_array_len(&mut self,
                          bcx: &Builder<'a, 'tcx>,
                          place: &mir::Place<'tcx>) -> ValueRef
    {
        // ZST are passed as operands and require special handling
        // because trans_place() panics if Local is operand.
        if let mir::Place::Local(index) = *place {
            if let LocalRef::Operand(Some(op)) = self.locals[index] {
                if let ty::TyArray(_, n) = op.layout.ty.sty {
                    let n = n.val.to_const_int().unwrap().to_u64().unwrap();
                    return common::C_usize(bcx.ccx, n);
                }
            }
        }
        // use common size calculation for non zero-sized types
        let tr_value = self.trans_place(&bcx, place);
        return tr_value.len(bcx.ccx);
    }

    pub fn trans_scalar_binop(&mut self,
                              bcx: &Builder<'a, 'tcx>,
                              op: mir::BinOp,
                              lhs: ValueRef,
                              rhs: ValueRef,
                              input_ty: Ty<'tcx>) -> ValueRef {
        let is_float = input_ty.is_fp();
        let is_signed = input_ty.is_signed();
        let is_nil = input_ty.is_nil();
        let is_bool = input_ty.is_bool();
        match op {
            mir::BinOp::Add => if is_float {
                bcx.fadd(lhs, rhs)
            } else {
                bcx.add(lhs, rhs)
            },
            mir::BinOp::Sub => if is_float {
                bcx.fsub(lhs, rhs)
            } else {
                bcx.sub(lhs, rhs)
            },
            mir::BinOp::Mul => if is_float {
                bcx.fmul(lhs, rhs)
            } else {
                bcx.mul(lhs, rhs)
            },
            mir::BinOp::Div => if is_float {
                bcx.fdiv(lhs, rhs)
            } else if is_signed {
                bcx.sdiv(lhs, rhs)
            } else {
                bcx.udiv(lhs, rhs)
            },
            mir::BinOp::Rem => if is_float {
                bcx.frem(lhs, rhs)
            } else if is_signed {
                bcx.srem(lhs, rhs)
            } else {
                bcx.urem(lhs, rhs)
            },
            mir::BinOp::BitOr => bcx.or(lhs, rhs),
            mir::BinOp::BitAnd => bcx.and(lhs, rhs),
            mir::BinOp::BitXor => bcx.xor(lhs, rhs),
            mir::BinOp::Offset => bcx.inbounds_gep(lhs, &[rhs]),
            mir::BinOp::Shl => common::build_unchecked_lshift(bcx, lhs, rhs),
            mir::BinOp::Shr => common::build_unchecked_rshift(bcx, input_ty, lhs, rhs),
            mir::BinOp::Ne | mir::BinOp::Lt | mir::BinOp::Gt |
            mir::BinOp::Eq | mir::BinOp::Le | mir::BinOp::Ge => if is_nil {
                C_bool(bcx.ccx, match op {
                    mir::BinOp::Ne | mir::BinOp::Lt | mir::BinOp::Gt => false,
                    mir::BinOp::Eq | mir::BinOp::Le | mir::BinOp::Ge => true,
                    _ => unreachable!()
                })
            } else if is_float {
                bcx.fcmp(
                    base::bin_op_to_fcmp_predicate(op.to_hir_binop()),
                    lhs, rhs
                )
            } else {
                let (lhs, rhs) = if is_bool {
                    // FIXME(#36856) -- extend the bools into `i8` because
                    // LLVM's i1 comparisons are broken.
                    (bcx.zext(lhs, Type::i8(bcx.ccx)),
                     bcx.zext(rhs, Type::i8(bcx.ccx)))
                } else {
                    (lhs, rhs)
                };

                bcx.icmp(
                    base::bin_op_to_icmp_predicate(op.to_hir_binop(), is_signed),
                    lhs, rhs
                )
            }
        }
    }

    pub fn trans_fat_ptr_binop(&mut self,
                               bcx: &Builder<'a, 'tcx>,
                               op: mir::BinOp,
                               lhs_addr: ValueRef,
                               lhs_extra: ValueRef,
                               rhs_addr: ValueRef,
                               rhs_extra: ValueRef,
                               _input_ty: Ty<'tcx>)
                               -> ValueRef {
        match op {
            mir::BinOp::Eq => {
                bcx.and(
                    bcx.icmp(llvm::IntEQ, lhs_addr, rhs_addr),
                    bcx.icmp(llvm::IntEQ, lhs_extra, rhs_extra)
                )
            }
            mir::BinOp::Ne => {
                bcx.or(
                    bcx.icmp(llvm::IntNE, lhs_addr, rhs_addr),
                    bcx.icmp(llvm::IntNE, lhs_extra, rhs_extra)
                )
            }
            mir::BinOp::Le | mir::BinOp::Lt |
            mir::BinOp::Ge | mir::BinOp::Gt => {
                // a OP b ~ a.0 STRICT(OP) b.0 | (a.0 == b.0 && a.1 OP a.1)
                let (op, strict_op) = match op {
                    mir::BinOp::Lt => (llvm::IntULT, llvm::IntULT),
                    mir::BinOp::Le => (llvm::IntULE, llvm::IntULT),
                    mir::BinOp::Gt => (llvm::IntUGT, llvm::IntUGT),
                    mir::BinOp::Ge => (llvm::IntUGE, llvm::IntUGT),
                    _ => bug!(),
                };

                bcx.or(
                    bcx.icmp(strict_op, lhs_addr, rhs_addr),
                    bcx.and(
                        bcx.icmp(llvm::IntEQ, lhs_addr, rhs_addr),
                        bcx.icmp(op, lhs_extra, rhs_extra)
                    )
                )
            }
            _ => {
                bug!("unexpected fat ptr binop");
            }
        }
    }

    pub fn trans_scalar_checked_binop(&mut self,
                                      bcx: &Builder<'a, 'tcx>,
                                      op: mir::BinOp,
                                      lhs: ValueRef,
                                      rhs: ValueRef,
                                      input_ty: Ty<'tcx>) -> OperandValue {
        // This case can currently arise only from functions marked
        // with #[rustc_inherit_overflow_checks] and inlined from
        // another crate (mostly core::num generic/#[inline] fns),
        // while the current crate doesn't use overflow checks.
        if !bcx.ccx.check_overflow() {
            let val = self.trans_scalar_binop(bcx, op, lhs, rhs, input_ty);
            return OperandValue::Pair(val, C_bool(bcx.ccx, false));
        }

        // First try performing the operation on constants, which
        // will only succeed if both operands are constant.
        // This is necessary to determine when an overflow Assert
        // will always panic at runtime, and produce a warning.
        if let Some((val, of)) = const_scalar_checked_binop(bcx.tcx(), op, lhs, rhs, input_ty) {
            return OperandValue::Pair(val, C_bool(bcx.ccx, of));
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
                let intrinsic = get_overflow_intrinsic(oop, bcx, input_ty);
                let res = bcx.call(intrinsic, &[lhs, rhs], None);

                (bcx.extract_value(res, 0),
                 bcx.extract_value(res, 1))
            }
            mir::BinOp::Shl | mir::BinOp::Shr => {
                let lhs_llty = val_ty(lhs);
                let rhs_llty = val_ty(rhs);
                let invert_mask = common::shift_mask_val(&bcx, lhs_llty, rhs_llty, true);
                let outer_bits = bcx.and(rhs, invert_mask);

                let of = bcx.icmp(llvm::IntNE, outer_bits, C_null(rhs_llty));
                let val = self.trans_scalar_binop(bcx, op, lhs, rhs, input_ty);

                (val, of)
            }
            _ => {
                bug!("Operator `{:?}` is not a checkable operator", op)
            }
        };

        OperandValue::Pair(val, of)
    }

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
                let ty = rvalue.ty(self.mir, self.ccx.tcx());
                let ty = self.monomorphize(&ty);
                self.ccx.layout_of(ty).is_zst()
            }
        }

        // (*) this is only true if the type is suitable
    }
}

#[derive(Copy, Clone)]
enum OverflowOp {
    Add, Sub, Mul
}

fn get_overflow_intrinsic(oop: OverflowOp, bcx: &Builder, ty: Ty) -> ValueRef {
    use syntax::ast::IntTy::*;
    use syntax::ast::UintTy::*;
    use rustc::ty::{TyInt, TyUint};

    let tcx = bcx.tcx();

    let new_sty = match ty.sty {
        TyInt(Is) => match &tcx.sess.target.target.target_pointer_width[..] {
            "16" => TyInt(I16),
            "32" => TyInt(I32),
            "64" => TyInt(I64),
            _ => panic!("unsupported target word size")
        },
        TyUint(Us) => match &tcx.sess.target.target.target_pointer_width[..] {
            "16" => TyUint(U16),
            "32" => TyUint(U32),
            "64" => TyUint(U64),
            _ => panic!("unsupported target word size")
        },
        ref t @ TyUint(_) | ref t @ TyInt(_) => t.clone(),
        _ => panic!("tried to get overflow intrinsic for op applied to non-int type")
    };

    let name = match oop {
        OverflowOp::Add => match new_sty {
            TyInt(I8) => "llvm.sadd.with.overflow.i8",
            TyInt(I16) => "llvm.sadd.with.overflow.i16",
            TyInt(I32) => "llvm.sadd.with.overflow.i32",
            TyInt(I64) => "llvm.sadd.with.overflow.i64",
            TyInt(I128) => "llvm.sadd.with.overflow.i128",

            TyUint(U8) => "llvm.uadd.with.overflow.i8",
            TyUint(U16) => "llvm.uadd.with.overflow.i16",
            TyUint(U32) => "llvm.uadd.with.overflow.i32",
            TyUint(U64) => "llvm.uadd.with.overflow.i64",
            TyUint(U128) => "llvm.uadd.with.overflow.i128",

            _ => unreachable!(),
        },
        OverflowOp::Sub => match new_sty {
            TyInt(I8) => "llvm.ssub.with.overflow.i8",
            TyInt(I16) => "llvm.ssub.with.overflow.i16",
            TyInt(I32) => "llvm.ssub.with.overflow.i32",
            TyInt(I64) => "llvm.ssub.with.overflow.i64",
            TyInt(I128) => "llvm.ssub.with.overflow.i128",

            TyUint(U8) => "llvm.usub.with.overflow.i8",
            TyUint(U16) => "llvm.usub.with.overflow.i16",
            TyUint(U32) => "llvm.usub.with.overflow.i32",
            TyUint(U64) => "llvm.usub.with.overflow.i64",
            TyUint(U128) => "llvm.usub.with.overflow.i128",

            _ => unreachable!(),
        },
        OverflowOp::Mul => match new_sty {
            TyInt(I8) => "llvm.smul.with.overflow.i8",
            TyInt(I16) => "llvm.smul.with.overflow.i16",
            TyInt(I32) => "llvm.smul.with.overflow.i32",
            TyInt(I64) => "llvm.smul.with.overflow.i64",
            TyInt(I128) => "llvm.smul.with.overflow.i128",

            TyUint(U8) => "llvm.umul.with.overflow.i8",
            TyUint(U16) => "llvm.umul.with.overflow.i16",
            TyUint(U32) => "llvm.umul.with.overflow.i32",
            TyUint(U64) => "llvm.umul.with.overflow.i64",
            TyUint(U128) => "llvm.umul.with.overflow.i128",

            _ => unreachable!(),
        },
    };

    bcx.ccx.get_intrinsic(&name)
}

fn cast_int_to_float(bcx: &Builder,
                     signed: bool,
                     x: ValueRef,
                     int_ty: Type,
                     float_ty: Type) -> ValueRef {
    // Most integer types, even i128, fit into [-f32::MAX, f32::MAX] after rounding.
    // It's only u128 -> f32 that can cause overflows (i.e., should yield infinity).
    // LLVM's uitofp produces undef in those cases, so we manually check for that case.
    let is_u128_to_f32 = !signed && int_ty.int_width() == 128 && float_ty.float_width() == 32;
    if is_u128_to_f32 {
        // All inputs greater or equal to (f32::MAX + 0.5 ULP) are rounded to infinity,
        // and for everything else LLVM's uitofp works just fine.
        let max = C_uint_big(int_ty, MAX_F32_PLUS_HALF_ULP);
        let overflow = bcx.icmp(llvm::IntUGE, x, max);
        let infinity_bits = C_u32(bcx.ccx, ieee::Single::INFINITY.to_bits() as u32);
        let infinity = consts::bitcast(infinity_bits, float_ty);
        bcx.select(overflow, infinity, bcx.uitofp(x, float_ty))
    } else {
        if signed {
            bcx.sitofp(x, float_ty)
        } else {
            bcx.uitofp(x, float_ty)
        }
    }
}

fn cast_float_to_int(bcx: &Builder,
                     signed: bool,
                     x: ValueRef,
                     float_ty: Type,
                     int_ty: Type) -> ValueRef {
    let fptosui_result = if signed {
        bcx.fptosi(x, int_ty)
    } else {
        bcx.fptoui(x, int_ty)
    };

    if !bcx.sess().opts.debugging_opts.saturating_float_casts {
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
    // representable. Note that this only works if float_ty's exponent range is sufficently large.
    // f16 or 256 bit integers would break this property. Right now the smallest float type is f32
    // with exponents ranging up to 127, which is barely enough for i128::MIN = -2^127.
    // On the other hand, f_max works even if int_ty::MAX is greater than float_ty::MAX. Because
    // we're rounding towards zero, we just get float_ty::MAX (which is always an integer).
    // This already happens today with u128::MAX = 2^128 - 1 > f32::MAX.
    fn compute_clamp_bounds<F: Float>(signed: bool, int_ty: Type) -> (u128, u128) {
        let rounded_min = F::from_i128_r(int_min(signed, int_ty), Round::TowardZero);
        assert_eq!(rounded_min.status, Status::OK);
        let rounded_max = F::from_u128_r(int_max(signed, int_ty), Round::TowardZero);
        assert!(rounded_max.value.is_finite());
        (rounded_min.value.to_bits(), rounded_max.value.to_bits())
    }
    fn int_max(signed: bool, int_ty: Type) -> u128 {
        let shift_amount = 128 - int_ty.int_width();
        if signed {
            i128::MAX as u128 >> shift_amount
        } else {
            u128::MAX >> shift_amount
        }
    }
    fn int_min(signed: bool, int_ty: Type) -> i128 {
        if signed {
            i128::MIN >> (128 - int_ty.int_width())
        } else {
            0
        }
    }
    let float_bits_to_llval = |bits| {
        let bits_llval = match float_ty.float_width() {
            32 => C_u32(bcx.ccx, bits as u32),
            64 => C_u64(bcx.ccx, bits as u64),
            n => bug!("unsupported float width {}", n),
        };
        consts::bitcast(bits_llval, float_ty)
    };
    let (f_min, f_max) = match float_ty.float_width() {
        32 => compute_clamp_bounds::<ieee::Single>(signed, int_ty),
        64 => compute_clamp_bounds::<ieee::Double>(signed, int_ty),
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
    let less_or_nan = bcx.fcmp(llvm::RealULT, x, f_min);
    let greater = bcx.fcmp(llvm::RealOGT, x, f_max);
    let int_max = C_uint_big(int_ty, int_max(signed, int_ty));
    let int_min = C_uint_big(int_ty, int_min(signed, int_ty) as u128);
    let s0 = bcx.select(less_or_nan, int_min, fptosui_result);
    let s1 = bcx.select(greater, int_max, s0);

    // Step 3: NaN replacement.
    // For unsigned types, the above step already yielded int_ty::MIN == 0 if x is NaN.
    // Therefore we only need to execute this step for signed integer types.
    if signed {
        // LLVM has no isNaN predicate, so we use (x == x) instead
        bcx.select(bcx.fcmp(llvm::RealOEQ, x, x), s1, C_uint(int_ty, 0))
    } else {
        s1
    }
}
