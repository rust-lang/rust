// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::ValueRef;
use rustc::middle::ty::{self, Ty};
use rustc_front::hir;
use rustc_mir::repr as mir;

use trans::asm;
use trans::base;
use trans::build;
use trans::common::{self, Block, Result};
use trans::debuginfo::DebugLoc;
use trans::declare;
use trans::expr;
use trans::machine;
use trans::type_::Type;
use trans::type_of;
use trans::tvec;

use super::MirContext;
use super::operand::{OperandRef, OperandValue};

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_rvalue(&mut self,
                        bcx: Block<'bcx, 'tcx>,
                        lldest: ValueRef,
                        rvalue: &mir::Rvalue<'tcx>)
                        -> Block<'bcx, 'tcx>
    {
        debug!("trans_rvalue(lldest={}, rvalue={:?})",
               bcx.val_to_string(lldest),
               rvalue);

        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                self.trans_operand_into(bcx, lldest, operand);
                bcx
            }

            mir::Rvalue::Cast(mir::CastKind::Unsize, ref operand, cast_ty) => {
                let expr_ty =
                    bcx.monomorphize(&self.mir.operand_ty(bcx.tcx(), operand));
                let cast_ty =
                    bcx.monomorphize(&cast_ty);
                if expr_ty == cast_ty {
                    debug!("trans_rvalue: trivial unsize at {:?}", expr_ty);
                    self.trans_operand_into(bcx, lldest, operand);
                    return bcx;
                }
                unimplemented!()
            }

            mir::Rvalue::Cast(..) => {
                unimplemented!()
            }

            mir::Rvalue::Repeat(ref elem, ref count) => {
                let elem = self.trans_operand(bcx, elem);
                let size = self.trans_constant(bcx, count).immediate();
                let base = expr::get_dataptr(bcx, lldest);
                tvec::iter_vec_raw(bcx, base, elem.ty, size, |bcx, llslot, _| {
                    self.store_operand(bcx, llslot, elem);
                    bcx
                })
            }

            mir::Rvalue::Aggregate(_, ref operands) => {
                for (i, operand) in operands.iter().enumerate() {
                    // Note: perhaps this should be StructGep, but
                    // note that in some cases the values here will
                    // not be structs but arrays.
                    let lldest_i = build::GEPi(bcx, lldest, &[0, i]);
                    self.trans_operand_into(bcx, lldest_i, operand);
                }
                bcx
            }

            mir::Rvalue::Slice { ref input, from_start, from_end } => {
                let ccx = bcx.ccx();
                let input = self.trans_lvalue(bcx, input);
                let (llbase, lllen) = tvec::get_base_and_len(bcx,
                                                             input.llval,
                                                             input.ty.to_ty(bcx.tcx()));
                let llbase1 = build::GEPi(bcx, llbase, &[from_start]);
                let adj = common::C_uint(ccx, from_start + from_end);
                let lllen1 = build::Sub(bcx, lllen, adj, DebugLoc::None);
                let lladdrdest = expr::get_dataptr(bcx, lldest);
                build::Store(bcx, llbase1, lladdrdest);
                let llmetadest = expr::get_meta(bcx, lldest);
                build::Store(bcx, lllen1, llmetadest);
                bcx
            }

            mir::Rvalue::InlineAsm(inline_asm) => {
                asm::trans_inline_asm(bcx, inline_asm)
            }

            _ => {
                assert!(rvalue_creates_operand(rvalue));
                let (bcx, temp) = self.trans_rvalue_operand(bcx, rvalue);
                self.store_operand(bcx, lldest, temp);
                bcx
            }
        }
    }

    pub fn trans_rvalue_operand(&mut self,
                                bcx: Block<'bcx, 'tcx>,
                                rvalue: &mir::Rvalue<'tcx>)
                                -> (Block<'bcx, 'tcx>, OperandRef<'tcx>)
    {
        assert!(rvalue_creates_operand(rvalue), "cannot trans {:?} to operand", rvalue);

        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                let operand = self.trans_operand(bcx, operand);
                (bcx, operand)
            }

            mir::Rvalue::Cast(mir::CastKind::Unsize, _, _) => {
                unimplemented!()
            }

            mir::Rvalue::Cast(..) => {
                unimplemented!()
            }

            mir::Rvalue::Ref(_, _, ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);

                let ty = tr_lvalue.ty.to_ty(bcx.tcx());
                // Note: lvalues are indirect, so storing the `llval` into the
                // destination effectively creates a reference.
                if common::type_is_sized(bcx.tcx(), ty) {
                    (bcx, OperandRef {
                        val: OperandValue::Imm(tr_lvalue.llval),
                        ty: ty,
                    })
                } else {
                    (bcx, OperandRef {
                        val: OperandValue::FatPtr(tr_lvalue.llval,
                                                  tr_lvalue.llextra),
                        ty: ty,
                    })
                }
            }

            mir::Rvalue::Len(ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);
                (bcx, OperandRef {
                    val: OperandValue::Imm(self.lvalue_len(bcx, tr_lvalue)),
                    ty: bcx.tcx().types.usize,
                })
            }

            mir::Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.trans_operand(bcx, lhs);
                let rhs = self.trans_operand(bcx, rhs);
                let llresult = if common::type_is_fat_ptr(bcx.tcx(), lhs.ty) {
                    match (lhs.val, rhs.val) {
                        (OperandValue::FatPtr(lhs_addr, lhs_extra),
                         OperandValue::FatPtr(rhs_addr, rhs_extra)) => {
                            base::compare_fat_ptrs(bcx,
                                                   lhs_addr, lhs_extra,
                                                   rhs_addr, rhs_extra,
                                                   lhs.ty, cmp_to_hir_cmp(op),
                                                   DebugLoc::None)
                        }
                        _ => unreachable!()
                    }

                } else {
                    self.trans_scalar_binop(bcx, op,
                                            lhs.immediate(), rhs.immediate(),
                                            lhs.ty, DebugLoc::None)
                };
                (bcx, OperandRef {
                    val: OperandValue::Imm(llresult),
                    ty: type_of_binop(bcx.tcx(), op, lhs.ty, rhs.ty),
                })
            }

            mir::Rvalue::UnaryOp(op, ref operand) => {
                let operand = self.trans_operand(bcx, operand);
                let lloperand = operand.immediate();
                let is_float = operand.ty.is_fp();
                let debug_loc = DebugLoc::None;
                let llval = match op {
                    mir::UnOp::Not => build::Not(bcx, lloperand, debug_loc),
                    mir::UnOp::Neg => if is_float {
                        build::FNeg(bcx, lloperand, debug_loc)
                    } else {
                        build::Neg(bcx, lloperand, debug_loc)
                    }
                };
                (bcx, OperandRef {
                    val: OperandValue::Imm(llval),
                    ty: operand.ty,
                })
            }

            mir::Rvalue::Box(content_ty) => {
                let content_ty: Ty<'tcx> = bcx.monomorphize(&content_ty);
                let llty = type_of::type_of(bcx.ccx(), content_ty);
                let llsize = machine::llsize_of(bcx.ccx(), llty);
                let align = type_of::align_of(bcx.ccx(), content_ty);
                let llalign = common::C_uint(bcx.ccx(), align);
                let llty_ptr = llty.ptr_to();
                let box_ty = bcx.tcx().mk_box(content_ty);
                let Result { bcx, val: llval } = base::malloc_raw_dyn(bcx,
                                                                      llty_ptr,
                                                                      box_ty,
                                                                      llsize,
                                                                      llalign,
                                                                      DebugLoc::None);
                (bcx, OperandRef {
                    val: OperandValue::Imm(llval),
                    ty: box_ty,
                })
            }

            mir::Rvalue::Repeat(..) |
            mir::Rvalue::Aggregate(..) |
            mir::Rvalue::Slice { .. } |
            mir::Rvalue::InlineAsm(..) => {
                bcx.tcx().sess.bug(&format!("cannot generate operand from rvalue {:?}", rvalue));
            }
        }
    }

    pub fn trans_scalar_binop(&mut self,
                              bcx: Block<'bcx, 'tcx>,
                              op: mir::BinOp,
                              lhs: ValueRef,
                              rhs: ValueRef,
                              input_ty: Ty<'tcx>,
                              debug_loc: DebugLoc) -> ValueRef {
        let is_float = input_ty.is_fp();
        let is_signed = input_ty.is_signed();
        match op {
            mir::BinOp::Add => if is_float {
                build::FAdd(bcx, lhs, rhs, debug_loc)
            } else {
                build::Add(bcx, lhs, rhs, debug_loc)
            },
            mir::BinOp::Sub => if is_float {
                build::FSub(bcx, lhs, rhs, debug_loc)
            } else {
                build::Sub(bcx, lhs, rhs, debug_loc)
            },
            mir::BinOp::Mul => if is_float {
                build::FMul(bcx, lhs, rhs, debug_loc)
            } else {
                build::Mul(bcx, lhs, rhs, debug_loc)
            },
            mir::BinOp::Div => if is_float {
                build::FDiv(bcx, lhs, rhs, debug_loc)
            } else if is_signed {
                build::SDiv(bcx, lhs, rhs, debug_loc)
            } else {
                build::UDiv(bcx, lhs, rhs, debug_loc)
            },
            mir::BinOp::Rem => if is_float {
                // LLVM currently always lowers the `frem` instructions appropriate
                // library calls typically found in libm. Notably f64 gets wired up
                // to `fmod` and f32 gets wired up to `fmodf`. Inconveniently for
                // us, 32-bit MSVC does not actually have a `fmodf` symbol, it's
                // instead just an inline function in a header that goes up to a
                // f64, uses `fmod`, and then comes back down to a f32.
                //
                // Although LLVM knows that `fmodf` doesn't exist on MSVC, it will
                // still unconditionally lower frem instructions over 32-bit floats
                // to a call to `fmodf`. To work around this we special case MSVC
                // 32-bit float rem instructions and instead do the call out to
                // `fmod` ourselves.
                //
                // Note that this is currently duplicated with src/libcore/ops.rs
                // which does the same thing, and it would be nice to perhaps unify
                // these two implementations one day! Also note that we call `fmod`
                // for both 32 and 64-bit floats because if we emit any FRem
                // instruction at all then LLVM is capable of optimizing it into a
                // 32-bit FRem (which we're trying to avoid).
                let tcx = bcx.tcx();
                let use_fmod = tcx.sess.target.target.options.is_like_msvc &&
                    tcx.sess.target.target.arch == "x86";
                if use_fmod {
                    let f64t = Type::f64(bcx.ccx());
                    let fty = Type::func(&[f64t, f64t], &f64t);
                    let llfn = declare::declare_cfn(bcx.ccx(), "fmod", fty,
                                                    tcx.types.f64);
                    if input_ty == tcx.types.f32 {
                        let lllhs = build::FPExt(bcx, lhs, f64t);
                        let llrhs = build::FPExt(bcx, rhs, f64t);
                        let llres = build::Call(bcx, llfn, &[lllhs, llrhs],
                                                None, debug_loc);
                        build::FPTrunc(bcx, llres, Type::f32(bcx.ccx()))
                    } else {
                        build::Call(bcx, llfn, &[lhs, rhs],
                                    None, debug_loc)
                    }
                } else {
                    build::FRem(bcx, lhs, rhs, debug_loc)
                }
            } else if is_signed {
                build::SRem(bcx, lhs, rhs, debug_loc)
            } else {
                build::URem(bcx, lhs, rhs, debug_loc)
            },
            mir::BinOp::BitOr => build::Or(bcx, lhs, rhs, debug_loc),
            mir::BinOp::BitAnd => build::And(bcx, lhs, rhs, debug_loc),
            mir::BinOp::BitXor => build::Xor(bcx, lhs, rhs, debug_loc),
            mir::BinOp::Shl => common::build_unchecked_lshift(bcx,
                                                              lhs,
                                                              rhs,
                                                              debug_loc),
            mir::BinOp::Shr => common::build_unchecked_rshift(bcx,
                                                              input_ty,
                                                              lhs,
                                                              rhs,
                                                              debug_loc),
            mir::BinOp::Eq | mir::BinOp::Lt | mir::BinOp::Gt |
            mir::BinOp::Ne | mir::BinOp::Le | mir::BinOp::Ge => {
                base::compare_scalar_types(bcx, lhs, rhs, input_ty,
                                           cmp_to_hir_cmp(op), debug_loc)
            }
        }
    }
}

pub fn rvalue_creates_operand<'tcx>(rvalue: &mir::Rvalue<'tcx>) -> bool {
    match *rvalue {
        mir::Rvalue::Use(..) | // (*)
        mir::Rvalue::Ref(..) |
        mir::Rvalue::Len(..) |
        mir::Rvalue::Cast(..) | // (*)
        mir::Rvalue::BinaryOp(..) |
        mir::Rvalue::UnaryOp(..) |
        mir::Rvalue::Box(..) =>
            true,
        mir::Rvalue::Repeat(..) |
        mir::Rvalue::Aggregate(..) |
        mir::Rvalue::Slice { .. } |
        mir::Rvalue::InlineAsm(..) =>
            false,
    }

    // (*) this is only true if the type is suitable
}

fn cmp_to_hir_cmp(op: mir::BinOp) -> hir::BinOp_ {
    match op {
        mir::BinOp::Eq => hir::BiEq,
        mir::BinOp::Ne => hir::BiNe,
        mir::BinOp::Lt => hir::BiLt,
        mir::BinOp::Le => hir::BiLe,
        mir::BinOp::Gt => hir::BiGt,
        mir::BinOp::Ge => hir::BiGe,
        _ => unreachable!()
    }
}

/// FIXME(nikomatsakis): I don't think this function should go here
fn type_of_binop<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    op: mir::BinOp,
    lhs_ty: Ty<'tcx>,
    rhs_ty: Ty<'tcx>)
    -> Ty<'tcx>
{
    match op {
        mir::BinOp::Add | mir::BinOp::Sub |
        mir::BinOp::Mul | mir::BinOp::Div | mir::BinOp::Rem |
        mir::BinOp::BitXor | mir::BinOp::BitAnd | mir::BinOp::BitOr => {
            // these should be integers or floats of the same size. We
            // probably want to dump all ops in some intrinsics framework
            // someday.
            assert_eq!(lhs_ty, rhs_ty);
            lhs_ty
        }
        mir::BinOp::Shl | mir::BinOp::Shr => {
            lhs_ty // lhs_ty can be != rhs_ty
        }
        mir::BinOp::Eq | mir::BinOp::Lt | mir::BinOp::Le |
        mir::BinOp::Ne | mir::BinOp::Ge | mir::BinOp::Gt => {
            tcx.types.bool
        }
    }
}
