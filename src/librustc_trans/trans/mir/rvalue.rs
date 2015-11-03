// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::abi;
use llvm::ValueRef;
use rustc::middle::ty::Ty;
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
use super::operand::OperandRef;

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

            mir::Rvalue::Cast(..) => {
                unimplemented!()
            }

            mir::Rvalue::Repeat(..) => {
                unimplemented!()
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
                build::Store(bcx, temp.llval, lldest);
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

            mir::Rvalue::Cast(..) => {
                unimplemented!()
            }

            mir::Rvalue::Ref(_, _, ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);

                // Note: lvalues are indirect, so storing the `llval` into the
                // destination effectively creates a reference.
                (bcx, OperandRef {
                    llval: tr_lvalue.llval,
                    ty: tr_lvalue.ty.to_ty(bcx.tcx()),
                })
            }

            mir::Rvalue::Len(ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);
                let (_, lllen) = tvec::get_base_and_len(bcx,
                                                        tr_lvalue.llval,
                                                        tr_lvalue.ty.to_ty(bcx.tcx()));
                (bcx, OperandRef {
                    llval: lllen,
                    ty: bcx.tcx().types.usize,
                })
            }

            mir::Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.trans_operand(bcx, lhs);
                let rhs = self.trans_operand(bcx, rhs);
                let is_float = lhs.ty.is_fp();
                let is_signed = lhs.ty.is_signed();
                let binop_debug_loc = DebugLoc::None;
                let llval = match op {
                    mir::BinOp::Add => if is_float {
                        build::FAdd(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    } else {
                        build::Add(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    },
                    mir::BinOp::Sub => if is_float {
                        build::FSub(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    } else {
                        build::Sub(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    },
                    mir::BinOp::Mul => if is_float {
                        build::FMul(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    } else {
                        build::Mul(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    },
                    mir::BinOp::Div => if is_float {
                        build::FDiv(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    } else if is_signed {
                        build::SDiv(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    } else {
                        build::UDiv(bcx, lhs.llval, rhs.llval, binop_debug_loc)
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
                            if lhs.ty == tcx.types.f32 {
                                let lllhs = build::FPExt(bcx, lhs.llval, f64t);
                                let llrhs = build::FPExt(bcx, rhs.llval, f64t);
                                let llres = build::Call(bcx, llfn, &[lllhs, llrhs],
                                                        None, binop_debug_loc);
                                build::FPTrunc(bcx, llres, Type::f32(bcx.ccx()))
                            } else {
                                build::Call(bcx, llfn, &[lhs.llval, rhs.llval],
                                            None, binop_debug_loc)
                            }
                        } else {
                            build::FRem(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                        }
                    } else if is_signed {
                        build::SRem(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    } else {
                        build::URem(bcx, lhs.llval, rhs.llval, binop_debug_loc)
                    },
                    mir::BinOp::BitOr => build::Or(bcx, lhs.llval, rhs.llval, binop_debug_loc),
                    mir::BinOp::BitAnd => build::And(bcx, lhs.llval, rhs.llval, binop_debug_loc),
                    mir::BinOp::BitXor => build::Xor(bcx, lhs.llval, rhs.llval, binop_debug_loc),
                    mir::BinOp::Shl => common::build_unchecked_lshift(bcx,
                                                                      lhs.llval,
                                                                      rhs.llval,
                                                                      binop_debug_loc),
                    mir::BinOp::Shr => common::build_unchecked_rshift(bcx,
                                                                      lhs.ty,
                                                                      lhs.llval,
                                                                      rhs.llval,
                                                                      binop_debug_loc),
                    mir::BinOp::Eq => base::compare_scalar_types(bcx, lhs.llval, rhs.llval, lhs.ty,
                                                                 hir::BiEq, binop_debug_loc),
                    mir::BinOp::Lt => base::compare_scalar_types(bcx, lhs.llval, rhs.llval, lhs.ty,
                                                                 hir::BiLt, binop_debug_loc),
                    mir::BinOp::Le => base::compare_scalar_types(bcx, lhs.llval, rhs.llval, lhs.ty,
                                                                 hir::BiLe, binop_debug_loc),
                    mir::BinOp::Ne => base::compare_scalar_types(bcx, lhs.llval, rhs.llval, lhs.ty,
                                                                 hir::BiNe, binop_debug_loc),
                    mir::BinOp::Ge => base::compare_scalar_types(bcx, lhs.llval, rhs.llval, lhs.ty,
                                                                 hir::BiGe, binop_debug_loc),
                    mir::BinOp::Gt => base::compare_scalar_types(bcx, lhs.llval, rhs.llval, lhs.ty,
                                                                 hir::BiGt, binop_debug_loc),
                };
                (bcx, OperandRef {
                    llval: llval,
                    ty: lhs.ty,
                })
            }

            mir::Rvalue::UnaryOp(op, ref operand) => {
                let operand = self.trans_operand(bcx, operand);
                let is_float = operand.ty.is_fp();
                let debug_loc = DebugLoc::None;
                let llval = match op {
                    mir::UnOp::Not => build::Not(bcx, operand.llval, debug_loc),
                    mir::UnOp::Neg => if is_float {
                        build::FNeg(bcx, operand.llval, debug_loc)
                    } else {
                        build::Neg(bcx, operand.llval, debug_loc)
                    }
                };
                (bcx, OperandRef {
                    llval: llval,
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
                    llval: llval,
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
