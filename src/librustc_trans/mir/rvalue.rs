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
use rustc::mir::repr as mir;

use asm;
use base;
use callee::Callee;
use common::{self, val_ty, C_bool, C_null, C_uint, BlockAndBuilder, Result};
use datum::{Datum, Lvalue};
use debuginfo::DebugLoc;
use adt;
use machine;
use type_of;
use tvec;
use value::Value;
use Disr;

use super::MirContext;
use super::constant::const_scalar_checked_binop;
use super::operand::{OperandRef, OperandValue};
use super::lvalue::{LvalueRef, get_dataptr};

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_rvalue(&mut self,
                        bcx: BlockAndBuilder<'bcx, 'tcx>,
                        dest: LvalueRef<'tcx>,
                        rvalue: &mir::Rvalue<'tcx>,
                        debug_loc: DebugLoc)
                        -> BlockAndBuilder<'bcx, 'tcx>
    {
        debug!("trans_rvalue(dest.llval={:?}, rvalue={:?})",
               Value(dest.llval), rvalue);

        match *rvalue {
           mir::Rvalue::Use(ref operand) => {
               let tr_operand = self.trans_operand(&bcx, operand);
               // FIXME: consider not copying constants through stack. (fixable by translating
               // constants into OperandValue::Ref, why don’t we do that yet if we don’t?)
               self.store_operand(&bcx, dest.llval, tr_operand);
               bcx
           }

            mir::Rvalue::Cast(mir::CastKind::Unsize, ref source, cast_ty) => {
                let cast_ty = bcx.monomorphize(&cast_ty);

                if common::type_is_fat_ptr(bcx.tcx(), cast_ty) {
                    // into-coerce of a thin pointer to a fat pointer - just
                    // use the operand path.
                    let (bcx, temp) = self.trans_rvalue_operand(bcx, rvalue, debug_loc);
                    self.store_operand(&bcx, dest.llval, temp);
                    return bcx;
                }

                // Unsize of a nontrivial struct. I would prefer for
                // this to be eliminated by MIR translation, but
                // `CoerceUnsized` can be passed by a where-clause,
                // so the (generic) MIR may not be able to expand it.
                let operand = self.trans_operand(&bcx, source);
                let operand = operand.pack_if_pair(&bcx);
                bcx.with_block(|bcx| {
                    match operand.val {
                        OperandValue::Pair(..) => bug!(),
                        OperandValue::Immediate(llval) => {
                            // unsize from an immediate structure. We don't
                            // really need a temporary alloca here, but
                            // avoiding it would require us to have
                            // `coerce_unsized_into` use extractvalue to
                            // index into the struct, and this case isn't
                            // important enough for it.
                            debug!("trans_rvalue: creating ugly alloca");
                            let lltemp = base::alloc_ty(bcx, operand.ty, "__unsize_temp");
                            base::store_ty(bcx, llval, lltemp, operand.ty);
                            base::coerce_unsized_into(bcx,
                                                      lltemp, operand.ty,
                                                      dest.llval, cast_ty);
                        }
                        OperandValue::Ref(llref) => {
                            base::coerce_unsized_into(bcx,
                                                      llref, operand.ty,
                                                      dest.llval, cast_ty);
                        }
                    }
                });
                bcx
            }

            mir::Rvalue::Repeat(ref elem, ref count) => {
                let tr_elem = self.trans_operand(&bcx, elem);
                let size = count.value.as_u64(bcx.tcx().sess.target.uint_type);
                let size = C_uint(bcx.ccx(), size);
                let base = get_dataptr(&bcx, dest.llval);
                let bcx = bcx.map_block(|block| {
                    tvec::iter_vec_raw(block, base, tr_elem.ty, size, |block, llslot, _| {
                        self.store_operand_direct(block, llslot, tr_elem);
                        block
                    })
                });
                bcx
            }

            mir::Rvalue::Aggregate(ref kind, ref operands) => {
                match *kind {
                    mir::AggregateKind::Adt(adt_def, index, _) => {
                        let repr = adt::represent_type(bcx.ccx(), dest.ty.to_ty(bcx.tcx()));
                        let disr = Disr::from(adt_def.variants[index].disr_val);
                        bcx.with_block(|bcx| {
                            adt::trans_set_discr(bcx, &repr, dest.llval, Disr::from(disr));
                        });
                        for (i, operand) in operands.iter().enumerate() {
                            let op = self.trans_operand(&bcx, operand);
                            // Do not generate stores and GEPis for zero-sized fields.
                            if !common::type_is_zero_size(bcx.ccx(), op.ty) {
                                let val = adt::MaybeSizedValue::sized(dest.llval);
                                let lldest_i = adt::trans_field_ptr_builder(&bcx, &repr,
                                                                            val, disr, i);
                                self.store_operand(&bcx, lldest_i, op);
                            }
                        }
                    },
                    _ => {
                        // FIXME Shouldn't need to manually trigger closure instantiations.
                        if let mir::AggregateKind::Closure(def_id, substs) = *kind {
                            use rustc::hir;
                            use syntax::ast::DUMMY_NODE_ID;
                            use syntax::ptr::P;
                            use syntax_pos::DUMMY_SP;
                            use closure;

                            closure::trans_closure_expr(closure::Dest::Ignore(bcx.ccx()),
                                                        &hir::FnDecl {
                                                            inputs: P::new(),
                                                            output: hir::NoReturn(DUMMY_SP),
                                                            variadic: false
                                                        },
                                                        &hir::Block {
                                                            stmts: P::new(),
                                                            expr: None,
                                                            id: DUMMY_NODE_ID,
                                                            rules: hir::DefaultBlock,
                                                            span: DUMMY_SP
                                                        },
                                                        DUMMY_NODE_ID, def_id,
                                                        bcx.monomorphize(&substs));
                        }

                        for (i, operand) in operands.iter().enumerate() {
                            let op = self.trans_operand(&bcx, operand);
                            // Do not generate stores and GEPis for zero-sized fields.
                            if !common::type_is_zero_size(bcx.ccx(), op.ty) {
                                // Note: perhaps this should be StructGep, but
                                // note that in some cases the values here will
                                // not be structs but arrays.
                                let dest = bcx.gepi(dest.llval, &[0, i]);
                                self.store_operand(&bcx, dest, op);
                            }
                        }
                    }
                }
                bcx
            }

            mir::Rvalue::InlineAsm { ref asm, ref outputs, ref inputs } => {
                let outputs = outputs.iter().map(|output| {
                    let lvalue = self.trans_lvalue(&bcx, output);
                    Datum::new(lvalue.llval, lvalue.ty.to_ty(bcx.tcx()),
                               Lvalue::new("out"))
                }).collect();

                let input_vals = inputs.iter().map(|input| {
                    self.trans_operand(&bcx, input).immediate()
                }).collect();

                bcx.with_block(|bcx| {
                    asm::trans_inline_asm(bcx, asm, outputs, input_vals);
                });

                bcx
            }

            _ => {
                assert!(rvalue_creates_operand(&self.mir, &bcx, rvalue));
                let (bcx, temp) = self.trans_rvalue_operand(bcx, rvalue, debug_loc);
                self.store_operand(&bcx, dest.llval, temp);
                bcx
            }
        }
    }

    pub fn trans_rvalue_operand(&mut self,
                                bcx: BlockAndBuilder<'bcx, 'tcx>,
                                rvalue: &mir::Rvalue<'tcx>,
                                debug_loc: DebugLoc)
                                -> (BlockAndBuilder<'bcx, 'tcx>, OperandRef<'tcx>)
    {
        assert!(rvalue_creates_operand(&self.mir, &bcx, rvalue),
                "cannot trans {:?} to operand", rvalue);

        match *rvalue {
            mir::Rvalue::Cast(ref kind, ref source, cast_ty) => {
                let operand = self.trans_operand(&bcx, source);
                debug!("cast operand is {:?}", operand);
                let cast_ty = bcx.monomorphize(&cast_ty);

                let val = match *kind {
                    mir::CastKind::ReifyFnPointer => {
                        match operand.ty.sty {
                            ty::TyFnDef(def_id, substs, _) => {
                                OperandValue::Immediate(
                                    Callee::def(bcx.ccx(), def_id, substs)
                                        .reify(bcx.ccx()).val)
                            }
                            _ => {
                                bug!("{} cannot be reified to a fn ptr", operand.ty)
                            }
                        }
                    }
                    mir::CastKind::UnsafeFnPointer => {
                        // this is a no-op at the LLVM level
                        operand.val
                    }
                    mir::CastKind::Unsize => {
                        // unsize targets other than to a fat pointer currently
                        // can't be operands.
                        assert!(common::type_is_fat_ptr(bcx.tcx(), cast_ty));

                        match operand.val {
                            OperandValue::Pair(lldata, llextra) => {
                                // unsize from a fat pointer - this is a
                                // "trait-object-to-supertrait" coercion, for
                                // example,
                                //   &'a fmt::Debug+Send => &'a fmt::Debug,
                                // So we need to pointercast the base to ensure
                                // the types match up.
                                let llcast_ty = type_of::fat_ptr_base_ty(bcx.ccx(), cast_ty);
                                let lldata = bcx.pointercast(lldata, llcast_ty);
                                OperandValue::Pair(lldata, llextra)
                            }
                            OperandValue::Immediate(lldata) => {
                                // "standard" unsize
                                let (lldata, llextra) = bcx.with_block(|bcx| {
                                    base::unsize_thin_ptr(bcx, lldata,
                                                          operand.ty, cast_ty)
                                });
                                OperandValue::Pair(lldata, llextra)
                            }
                            OperandValue::Ref(_) => {
                                bug!("by-ref operand {:?} in trans_rvalue_operand",
                                     operand);
                            }
                        }
                    }
                    mir::CastKind::Misc if common::type_is_immediate(bcx.ccx(), operand.ty) => {
                        debug_assert!(common::type_is_immediate(bcx.ccx(), cast_ty));
                        let r_t_in = CastTy::from_ty(operand.ty).expect("bad input type for cast");
                        let r_t_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                        let ll_t_in = type_of::immediate_type_of(bcx.ccx(), operand.ty);
                        let ll_t_out = type_of::immediate_type_of(bcx.ccx(), cast_ty);
                        let llval = operand.immediate();
                        let signed = if let CastTy::Int(IntTy::CEnum) = r_t_in {
                            let repr = adt::represent_type(bcx.ccx(), operand.ty);
                            adt::is_discr_signed(&repr)
                        } else {
                            operand.ty.is_signed()
                        };

                        let newval = match (r_t_in, r_t_out) {
                            (CastTy::Int(_), CastTy::Int(_)) => {
                                let srcsz = ll_t_in.int_width();
                                let dstsz = ll_t_out.int_width();
                                if srcsz == dstsz {
                                    bcx.bitcast(llval, ll_t_out)
                                } else if srcsz > dstsz {
                                    bcx.trunc(llval, ll_t_out)
                                } else if signed {
                                    bcx.sext(llval, ll_t_out)
                                } else {
                                    bcx.zext(llval, ll_t_out)
                                }
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
                            (CastTy::Int(_), CastTy::Float) if signed =>
                                bcx.sitofp(llval, ll_t_out),
                            (CastTy::Int(_), CastTy::Float) =>
                                bcx.uitofp(llval, ll_t_out),
                            (CastTy::Float, CastTy::Int(IntTy::I)) =>
                                bcx.fptosi(llval, ll_t_out),
                            (CastTy::Float, CastTy::Int(_)) =>
                                bcx.fptoui(llval, ll_t_out),
                            _ => bug!("unsupported cast: {:?} to {:?}", operand.ty, cast_ty)
                        };
                        OperandValue::Immediate(newval)
                    }
                    mir::CastKind::Misc => { // Casts from a fat-ptr.
                        let ll_cast_ty = type_of::immediate_type_of(bcx.ccx(), cast_ty);
                        let ll_from_ty = type_of::immediate_type_of(bcx.ccx(), operand.ty);
                        if let OperandValue::Pair(data_ptr, meta_ptr) = operand.val {
                            if common::type_is_fat_ptr(bcx.tcx(), cast_ty) {
                                let ll_cft = ll_cast_ty.field_types();
                                let ll_fft = ll_from_ty.field_types();
                                let data_cast = bcx.pointercast(data_ptr, ll_cft[0]);
                                assert_eq!(ll_cft[1].kind(), ll_fft[1].kind());
                                OperandValue::Pair(data_cast, meta_ptr)
                            } else { // cast to thin-ptr
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr and
                                // pointer-cast of that pointer to desired pointer type.
                                let llval = bcx.pointercast(data_ptr, ll_cast_ty);
                                OperandValue::Immediate(llval)
                            }
                        } else {
                            bug!("Unexpected non-Pair operand")
                        }
                    }
                };
                let operand = OperandRef {
                    val: val,
                    ty: cast_ty
                };
                (bcx, operand)
            }

            mir::Rvalue::Ref(_, bk, ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(&bcx, lvalue);

                let ty = tr_lvalue.ty.to_ty(bcx.tcx());
                let ref_ty = bcx.tcx().mk_ref(
                    bcx.tcx().mk_region(ty::ReErased),
                    ty::TypeAndMut { ty: ty, mutbl: bk.to_mutbl_lossy() }
                );

                // Note: lvalues are indirect, so storing the `llval` into the
                // destination effectively creates a reference.
                let operand = if common::type_is_sized(bcx.tcx(), ty) {
                    OperandRef {
                        val: OperandValue::Immediate(tr_lvalue.llval),
                        ty: ref_ty,
                    }
                } else {
                    OperandRef {
                        val: OperandValue::Pair(tr_lvalue.llval,
                                                tr_lvalue.llextra),
                        ty: ref_ty,
                    }
                };
                (bcx, operand)
            }

            mir::Rvalue::Len(ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(&bcx, lvalue);
                let operand = OperandRef {
                    val: OperandValue::Immediate(tr_lvalue.len(bcx.ccx())),
                    ty: bcx.tcx().types.usize,
                };
                (bcx, operand)
            }

            mir::Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.trans_operand(&bcx, lhs);
                let rhs = self.trans_operand(&bcx, rhs);
                let llresult = if common::type_is_fat_ptr(bcx.tcx(), lhs.ty) {
                    match (lhs.val, rhs.val) {
                        (OperandValue::Pair(lhs_addr, lhs_extra),
                         OperandValue::Pair(rhs_addr, rhs_extra)) => {
                            bcx.with_block(|bcx| {
                                base::compare_fat_ptrs(bcx,
                                                       lhs_addr, lhs_extra,
                                                       rhs_addr, rhs_extra,
                                                       lhs.ty, op.to_hir_binop(),
                                                       debug_loc)
                            })
                        }
                        _ => bug!()
                    }

                } else {
                    self.trans_scalar_binop(&bcx, op,
                                            lhs.immediate(), rhs.immediate(),
                                            lhs.ty)
                };
                let operand = OperandRef {
                    val: OperandValue::Immediate(llresult),
                    ty: self.mir.binop_ty(bcx.tcx(), op, lhs.ty, rhs.ty),
                };
                (bcx, operand)
            }
            mir::Rvalue::CheckedBinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.trans_operand(&bcx, lhs);
                let rhs = self.trans_operand(&bcx, rhs);
                let result = self.trans_scalar_checked_binop(&bcx, op,
                                                             lhs.immediate(), rhs.immediate(),
                                                             lhs.ty);
                let val_ty = self.mir.binop_ty(bcx.tcx(), op, lhs.ty, rhs.ty);
                let operand_ty = bcx.tcx().mk_tup(vec![val_ty, bcx.tcx().types.bool]);
                let operand = OperandRef {
                    val: result,
                    ty: operand_ty
                };

                (bcx, operand)
            }

            mir::Rvalue::UnaryOp(op, ref operand) => {
                let operand = self.trans_operand(&bcx, operand);
                let lloperand = operand.immediate();
                let is_float = operand.ty.is_fp();
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
                    ty: operand.ty,
                })
            }

            mir::Rvalue::Box(content_ty) => {
                let content_ty: Ty<'tcx> = bcx.monomorphize(&content_ty);
                let llty = type_of::type_of(bcx.ccx(), content_ty);
                let llsize = machine::llsize_of(bcx.ccx(), llty);
                let align = type_of::align_of(bcx.ccx(), content_ty);
                let llalign = C_uint(bcx.ccx(), align);
                let llty_ptr = llty.ptr_to();
                let box_ty = bcx.tcx().mk_box(content_ty);
                let mut llval = None;
                let bcx = bcx.map_block(|bcx| {
                    let Result { bcx, val } = base::malloc_raw_dyn(bcx,
                                                                   llty_ptr,
                                                                   box_ty,
                                                                   llsize,
                                                                   llalign,
                                                                   debug_loc);
                    llval = Some(val);
                    bcx
                });
                let operand = OperandRef {
                    val: OperandValue::Immediate(llval.unwrap()),
                    ty: box_ty,
                };
                (bcx, operand)
            }

            mir::Rvalue::Use(ref operand) => {
                let operand = self.trans_operand(&bcx, operand);
                (bcx, operand)
            }
            mir::Rvalue::Repeat(..) |
            mir::Rvalue::Aggregate(..) |
            mir::Rvalue::InlineAsm { .. } => {
                bug!("cannot generate operand from rvalue {:?}", rvalue);

            }
        }
    }

    pub fn trans_scalar_binop(&mut self,
                              bcx: &BlockAndBuilder<'bcx, 'tcx>,
                              op: mir::BinOp,
                              lhs: ValueRef,
                              rhs: ValueRef,
                              input_ty: Ty<'tcx>) -> ValueRef {
        let is_float = input_ty.is_fp();
        let is_signed = input_ty.is_signed();
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
            mir::BinOp::Shl => {
                bcx.with_block(|bcx| {
                    common::build_unchecked_lshift(bcx,
                                                   lhs,
                                                   rhs,
                                                   DebugLoc::None)
                })
            }
            mir::BinOp::Shr => {
                bcx.with_block(|bcx| {
                    common::build_unchecked_rshift(bcx,
                                                   input_ty,
                                                   lhs,
                                                   rhs,
                                                   DebugLoc::None)
                })
            }
            mir::BinOp::Eq | mir::BinOp::Lt | mir::BinOp::Gt |
            mir::BinOp::Ne | mir::BinOp::Le | mir::BinOp::Ge => {
                bcx.with_block(|bcx| {
                    base::compare_scalar_types(bcx, lhs, rhs, input_ty,
                                               op.to_hir_binop(), DebugLoc::None)
                })
            }
        }
    }

    pub fn trans_scalar_checked_binop(&mut self,
                                      bcx: &BlockAndBuilder<'bcx, 'tcx>,
                                      op: mir::BinOp,
                                      lhs: ValueRef,
                                      rhs: ValueRef,
                                      input_ty: Ty<'tcx>) -> OperandValue {
        // This case can currently arise only from functions marked
        // with #[rustc_inherit_overflow_checks] and inlined from
        // another crate (mostly core::num generic/#[inline] fns),
        // while the current crate doesn't use overflow checks.
        if !bcx.ccx().check_overflow() {
            let val = self.trans_scalar_binop(bcx, op, lhs, rhs, input_ty);
            return OperandValue::Pair(val, C_bool(bcx.ccx(), false));
        }

        // First try performing the operation on constants, which
        // will only succeed if both operands are constant.
        // This is necessary to determine when an overflow Assert
        // will always panic at runtime, and produce a warning.
        match const_scalar_checked_binop(bcx.tcx(), op, lhs, rhs, input_ty) {
            Some((val, of)) => {
                return OperandValue::Pair(val, C_bool(bcx.ccx(), of));
            }
            None => {}
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
                let invert_mask = bcx.with_block(|bcx| {
                    common::shift_mask_val(bcx, lhs_llty, rhs_llty, true)
                });
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
}

pub fn rvalue_creates_operand<'bcx, 'tcx>(_mir: &mir::Mir<'tcx>,
                                          _bcx: &BlockAndBuilder<'bcx, 'tcx>,
                                          rvalue: &mir::Rvalue<'tcx>) -> bool {
    match *rvalue {
        mir::Rvalue::Ref(..) |
        mir::Rvalue::Len(..) |
        mir::Rvalue::Cast(..) | // (*)
        mir::Rvalue::BinaryOp(..) |
        mir::Rvalue::CheckedBinaryOp(..) |
        mir::Rvalue::UnaryOp(..) |
        mir::Rvalue::Box(..) |
        mir::Rvalue::Use(..) =>
            true,
        mir::Rvalue::Repeat(..) |
        mir::Rvalue::Aggregate(..) |
        mir::Rvalue::InlineAsm { .. } =>
            false,
    }

    // (*) this is only true if the type is suitable
}

#[derive(Copy, Clone)]
enum OverflowOp {
    Add, Sub, Mul
}

fn get_overflow_intrinsic(oop: OverflowOp, bcx: &BlockAndBuilder, ty: Ty) -> ValueRef {
    use syntax::ast::IntTy::*;
    use syntax::ast::UintTy::*;
    use rustc::ty::{TyInt, TyUint};

    let tcx = bcx.tcx();

    let new_sty = match ty.sty {
        TyInt(Is) => match &tcx.sess.target.target.target_pointer_width[..] {
            "32" => TyInt(I32),
            "64" => TyInt(I64),
            _ => panic!("unsupported target word size")
        },
        TyUint(Us) => match &tcx.sess.target.target.target_pointer_width[..] {
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

            TyUint(U8) => "llvm.uadd.with.overflow.i8",
            TyUint(U16) => "llvm.uadd.with.overflow.i16",
            TyUint(U32) => "llvm.uadd.with.overflow.i32",
            TyUint(U64) => "llvm.uadd.with.overflow.i64",

            _ => unreachable!(),
        },
        OverflowOp::Sub => match new_sty {
            TyInt(I8) => "llvm.ssub.with.overflow.i8",
            TyInt(I16) => "llvm.ssub.with.overflow.i16",
            TyInt(I32) => "llvm.ssub.with.overflow.i32",
            TyInt(I64) => "llvm.ssub.with.overflow.i64",

            TyUint(U8) => "llvm.usub.with.overflow.i8",
            TyUint(U16) => "llvm.usub.with.overflow.i16",
            TyUint(U32) => "llvm.usub.with.overflow.i32",
            TyUint(U64) => "llvm.usub.with.overflow.i64",

            _ => unreachable!(),
        },
        OverflowOp::Mul => match new_sty {
            TyInt(I8) => "llvm.smul.with.overflow.i8",
            TyInt(I16) => "llvm.smul.with.overflow.i16",
            TyInt(I32) => "llvm.smul.with.overflow.i32",
            TyInt(I64) => "llvm.smul.with.overflow.i64",

            TyUint(U8) => "llvm.umul.with.overflow.i8",
            TyUint(U16) => "llvm.umul.with.overflow.i16",
            TyUint(U32) => "llvm.umul.with.overflow.i32",
            TyUint(U64) => "llvm.umul.with.overflow.i64",

            _ => unreachable!(),
        },
    };

    bcx.ccx().get_intrinsic(&name)
}
