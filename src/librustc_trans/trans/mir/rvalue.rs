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
use middle::ty::cast::{CastTy, IntTy};
use rustc::mir::repr as mir;

use trans::asm;
use trans::base;
use trans::build;
use trans::common::{self, Block, Result};
use trans::debuginfo::DebugLoc;
use trans::declare;
use trans::expr;
use trans::adt;
use trans::machine;
use trans::type_::Type;
use trans::type_of;
use trans::tvec;

use super::MirContext;
use super::operand::{OperandRef, OperandValue};
use super::lvalue::LvalueRef;

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_rvalue(&mut self,
                        bcx: Block<'bcx, 'tcx>,
                        dest: LvalueRef<'tcx>,
                        rvalue: &mir::Rvalue<'tcx>)
                        -> Block<'bcx, 'tcx>
    {
        debug!("trans_rvalue(dest.llval={}, rvalue={:?})",
               bcx.val_to_string(dest.llval),
               rvalue);

        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                self.trans_operand_into(bcx, dest.llval, operand);
                bcx
            }

            mir::Rvalue::Cast(mir::CastKind::Unsize, ref operand, cast_ty) => {
                if common::type_is_fat_ptr(bcx.tcx(), cast_ty) {
                    // into-coerce of a thin pointer to a fat pointer - just
                    // use the operand path.
                    let (bcx, temp) = self.trans_rvalue_operand(bcx, rvalue);
                    self.store_operand(bcx, dest.llval, temp);
                    return bcx;
                }

                // Unsize of a nontrivial struct. I would prefer for
                // this to be eliminated by MIR translation, but
                // `CoerceUnsized` can be passed by a where-clause,
                // so the (generic) MIR may not be able to expand it.
                let operand = self.trans_operand(bcx, operand);
                match operand.val {
                    OperandValue::FatPtr(..) => unreachable!(),
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
                bcx
            }

            mir::Rvalue::Repeat(ref elem, ref count) => {
                let elem = self.trans_operand(bcx, elem);
                let size = self.trans_constant(bcx, count).immediate();
                let base = expr::get_dataptr(bcx, dest.llval);
                tvec::iter_vec_raw(bcx, base, elem.ty, size, |bcx, llslot, _| {
                    self.store_operand(bcx, llslot, elem);
                    bcx
                })
            }

            mir::Rvalue::Aggregate(ref kind, ref operands) => {
                match *kind {
                    // Unit struct or variant; both are translated very differently compared to any
                    // other aggregate
                    mir::AggregateKind::Adt(adt_def, index, _)
                    if adt_def.variants[index].kind() == ty::VariantKind::Unit => {
                        let repr = adt::represent_type(bcx.ccx(), dest.ty.to_ty(bcx.tcx()));
                        let disr = adt_def.variants[index].disr_val;
                        adt::trans_set_discr(bcx, &*repr, dest.llval, disr);
                    },
                    _ => {
                        for (i, operand) in operands.iter().enumerate() {
                            // Note: perhaps this should be StructGep, but
                            // note that in some cases the values here will
                            // not be structs but arrays.
                            let lldest_i = build::GEPi(bcx, dest.llval, &[0, i]);
                            self.trans_operand_into(bcx, lldest_i, operand);
                        }
                    }
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
                let lladdrdest = expr::get_dataptr(bcx, dest.llval);
                build::Store(bcx, llbase1, lladdrdest);
                let llmetadest = expr::get_meta(bcx, dest.llval);
                build::Store(bcx, lllen1, llmetadest);
                bcx
            }

            mir::Rvalue::InlineAsm(ref inline_asm) => {
                asm::trans_inline_asm(bcx, inline_asm)
            }

            _ => {
                assert!(rvalue_creates_operand(rvalue));
                let (bcx, temp) = self.trans_rvalue_operand(bcx, rvalue);
                self.store_operand(bcx, dest.llval, temp);
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

            mir::Rvalue::Cast(ref kind, ref operand, cast_ty) => {
                let operand = self.trans_operand(bcx, operand);
                debug!("cast operand is {}", operand.repr(bcx));
                let cast_ty = bcx.monomorphize(&cast_ty);

                let val = match *kind {
                    mir::CastKind::ReifyFnPointer |
                    mir::CastKind::UnsafeFnPointer => {
                        // these are no-ops at the LLVM level
                        operand.val
                    }
                    mir::CastKind::Unsize => {
                        // unsize targets other than to a fat pointer currently
                        // can't be operands.
                        assert!(common::type_is_fat_ptr(bcx.tcx(), cast_ty));

                        match operand.val {
                            OperandValue::FatPtr(..) => {
                                // unsize from a fat pointer - this is a
                                // "trait-object-to-supertrait" coercion, for
                                // example,
                                //   &'a fmt::Debug+Send => &'a fmt::Debug,
                                // and is a no-op at the LLVM level
                                operand.val
                            }
                            OperandValue::Immediate(lldata) => {
                                // "standard" unsize
                                let (lldata, llextra) =
                                    base::unsize_thin_ptr(bcx, lldata,
                                                          operand.ty, cast_ty);
                                OperandValue::FatPtr(lldata, llextra)
                            }
                            OperandValue::Ref(_) => {
                                bcx.sess().bug(
                                    &format!("by-ref operand {} in trans_rvalue_operand",
                                             operand.repr(bcx)));
                            }
                        }
                    }
                    mir::CastKind::Misc if common::type_is_immediate(bcx.ccx(), operand.ty) => {
                        debug_assert!(common::type_is_immediate(bcx.ccx(), cast_ty));
                        let r_t_in = CastTy::from_ty(operand.ty).expect("bad input type for cast");
                        let r_t_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                        let ll_t_in = type_of::arg_type_of(bcx.ccx(), operand.ty);
                        let ll_t_out = type_of::arg_type_of(bcx.ccx(), cast_ty);
                        let (llval, ll_t_in, signed) = if let CastTy::Int(IntTy::CEnum) = r_t_in {
                            let repr = adt::represent_type(bcx.ccx(), operand.ty);
                            let llval = operand.immediate();
                            let discr = adt::trans_get_discr(bcx, &*repr, llval, None);
                            (discr, common::val_ty(discr), adt::is_discr_signed(&*repr))
                        } else {
                            (operand.immediate(), ll_t_in, operand.ty.is_signed())
                        };

                        let newval = match (r_t_in, r_t_out) {
                            (CastTy::Int(_), CastTy::Int(_)) => {
                                let srcsz = ll_t_in.int_width();
                                let dstsz = ll_t_out.int_width();
                                if srcsz == dstsz {
                                    build::BitCast(bcx, llval, ll_t_out)
                                } else if srcsz > dstsz {
                                    build::Trunc(bcx, llval, ll_t_out)
                                } else if signed {
                                    build::SExt(bcx, llval, ll_t_out)
                                } else {
                                    build::ZExt(bcx, llval, ll_t_out)
                                }
                            }
                            (CastTy::Float, CastTy::Float) => {
                                let srcsz = ll_t_in.float_width();
                                let dstsz = ll_t_out.float_width();
                                if dstsz > srcsz {
                                    build::FPExt(bcx, llval, ll_t_out)
                                } else if srcsz > dstsz {
                                    build::FPTrunc(bcx, llval, ll_t_out)
                                } else {
                                    llval
                                }
                            }
                            (CastTy::Ptr(_), CastTy::Ptr(_)) |
                            (CastTy::FnPtr, CastTy::Ptr(_)) |
                            (CastTy::RPtr(_), CastTy::Ptr(_)) =>
                                build::PointerCast(bcx, llval, ll_t_out),
                            (CastTy::Ptr(_), CastTy::Int(_)) |
                            (CastTy::FnPtr, CastTy::Int(_)) =>
                                build::PtrToInt(bcx, llval, ll_t_out),
                            (CastTy::Int(_), CastTy::Ptr(_)) =>
                                build::IntToPtr(bcx, llval, ll_t_out),
                            (CastTy::Int(_), CastTy::Float) if signed =>
                                build::SIToFP(bcx, llval, ll_t_out),
                            (CastTy::Int(_), CastTy::Float) =>
                                build::UIToFP(bcx, llval, ll_t_out),
                            (CastTy::Float, CastTy::Int(IntTy::I)) =>
                                build::FPToSI(bcx, llval, ll_t_out),
                            (CastTy::Float, CastTy::Int(_)) =>
                                build::FPToUI(bcx, llval, ll_t_out),
                            _ => bcx.ccx().sess().bug(
                                &format!("unsupported cast: {:?} to {:?}", operand.ty, cast_ty)
                            )
                        };
                        OperandValue::Immediate(newval)
                    }
                    mir::CastKind::Misc => { // Casts from a fat-ptr.
                        let ll_cast_ty = type_of::arg_type_of(bcx.ccx(), cast_ty);
                        let ll_from_ty = type_of::arg_type_of(bcx.ccx(), operand.ty);
                        if let OperandValue::FatPtr(data_ptr, meta_ptr) = operand.val {
                            if common::type_is_fat_ptr(bcx.tcx(), cast_ty) {
                                let ll_cft = ll_cast_ty.field_types();
                                let ll_fft = ll_from_ty.field_types();
                                let data_cast = build::PointerCast(bcx, data_ptr, ll_cft[0]);
                                assert_eq!(ll_cft[1].kind(), ll_fft[1].kind());
                                OperandValue::FatPtr(data_cast, meta_ptr)
                            } else { // cast to thin-ptr
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr and
                                // pointer-cast of that pointer to desired pointer type.
                                let llval = build::PointerCast(bcx, data_ptr, ll_cast_ty);
                                OperandValue::Immediate(llval)
                            }
                        } else {
                            panic!("Unexpected non-FatPtr operand")
                        }
                    }
                };
                (bcx, OperandRef {
                    val: val,
                    ty: cast_ty
                })
            }

            mir::Rvalue::Ref(_, bk, ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);

                let ty = tr_lvalue.ty.to_ty(bcx.tcx());
                let ref_ty = bcx.tcx().mk_ref(
                    bcx.tcx().mk_region(ty::ReStatic),
                    ty::TypeAndMut { ty: ty, mutbl: bk.to_mutbl_lossy() }
                );

                // Note: lvalues are indirect, so storing the `llval` into the
                // destination effectively creates a reference.
                if common::type_is_sized(bcx.tcx(), ty) {
                    (bcx, OperandRef {
                        val: OperandValue::Immediate(tr_lvalue.llval),
                        ty: ref_ty,
                    })
                } else {
                    (bcx, OperandRef {
                        val: OperandValue::FatPtr(tr_lvalue.llval,
                                                  tr_lvalue.llextra),
                        ty: ref_ty,
                    })
                }
            }

            mir::Rvalue::Len(ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(bcx, lvalue);
                (bcx, OperandRef {
                    val: OperandValue::Immediate(self.lvalue_len(bcx, tr_lvalue)),
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
                                                   lhs.ty, op.to_hir_binop(),
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
                    val: OperandValue::Immediate(llresult),
                    ty: self.mir.binop_ty(bcx.tcx(), op, lhs.ty, rhs.ty),
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
                    val: OperandValue::Immediate(llval),
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
                    val: OperandValue::Immediate(llval),
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
                                           op.to_hir_binop(), debug_loc)
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
