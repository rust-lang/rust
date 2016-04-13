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
use rustc::ty::{self, Ty};
use rustc::ty::cast::{CastTy, IntTy};
use middle::const_val::ConstVal;
use rustc_const_math::ConstInt;
use rustc::mir::repr as mir;

use asm;
use base;
use callee::Callee;
use common::{self, C_uint, BlockAndBuilder, Result};
use datum::{Datum, Lvalue};
use debuginfo::DebugLoc;
use declare;
use adt;
use machine;
use type_::Type;
use type_of;
use tvec;
use value::Value;
use Disr;

use super::MirContext;
use super::operand::{OperandRef, OperandValue};
use super::lvalue::{LvalueRef, get_dataptr, get_meta};

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
               self.set_operand_dropped(&bcx, operand);
               bcx
           }

            mir::Rvalue::Cast(mir::CastKind::Unsize, ref source, cast_ty) => {
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
                bcx.with_block(|bcx| {
                    match operand.val {
                        OperandValue::FatPtr(..) => bug!(),
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
                self.set_operand_dropped(&bcx, source);
                bcx
            }

            mir::Rvalue::Repeat(ref elem, ref count) => {
                let tr_elem = self.trans_operand(&bcx, elem);
                let count = ConstVal::Integral(ConstInt::Usize(count.value));
                let size = self.trans_constval(&bcx, &count, bcx.tcx().types.usize).immediate();
                let base = get_dataptr(&bcx, dest.llval);
                let bcx = bcx.map_block(|block| {
                    tvec::iter_vec_raw(block, base, tr_elem.ty, size, |block, llslot, _| {
                        self.store_operand_direct(block, llslot, tr_elem);
                        block
                    })
                });
                self.set_operand_dropped(&bcx, elem);
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
                            self.set_operand_dropped(&bcx, operand);
                        }
                    },
                    _ => {
                        // FIXME Shouldn't need to manually trigger closure instantiations.
                        if let mir::AggregateKind::Closure(def_id, substs) = *kind {
                            use rustc::hir;
                            use syntax::ast::DUMMY_NODE_ID;
                            use syntax::codemap::DUMMY_SP;
                            use syntax::ptr::P;
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
                                                        &bcx.monomorphize(substs));
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
                            self.set_operand_dropped(&bcx, operand);
                        }
                    }
                }
                bcx
            }

            mir::Rvalue::Slice { ref input, from_start, from_end } => {
                let ccx = bcx.ccx();
                let input = self.trans_lvalue(&bcx, input);
                let ty = input.ty.to_ty(bcx.tcx());
                let (llbase1, lllen) = match ty.sty {
                    ty::TyArray(_, n) => {
                        (bcx.gepi(input.llval, &[0, from_start]), C_uint(ccx, n))
                    }
                    ty::TySlice(_) | ty::TyStr => {
                        (bcx.gepi(input.llval, &[from_start]), input.llextra)
                    }
                    _ => bug!("cannot slice {}", ty)
                };
                let adj = C_uint(ccx, from_start + from_end);
                let lllen1 = bcx.sub(lllen, adj);
                bcx.store(llbase1, get_dataptr(&bcx, dest.llval));
                bcx.store(lllen1, get_meta(&bcx, dest.llval));
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

                for input in inputs {
                    self.set_operand_dropped(&bcx, input);
                }
                bcx
            }

            _ => {
                assert!(rvalue_creates_operand(rvalue));
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
        assert!(rvalue_creates_operand(rvalue), "cannot trans {:?} to operand", rvalue);

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
                            OperandValue::FatPtr(..) => {
                                // unsize from a fat pointer - this is a
                                // "trait-object-to-supertrait" coercion, for
                                // example,
                                //   &'a fmt::Debug+Send => &'a fmt::Debug,
                                // and is a no-op at the LLVM level
                                self.set_operand_dropped(&bcx, source);
                                operand.val
                            }
                            OperandValue::Immediate(lldata) => {
                                // "standard" unsize
                                let (lldata, llextra) = bcx.with_block(|bcx| {
                                    base::unsize_thin_ptr(bcx, lldata,
                                                          operand.ty, cast_ty)
                                });
                                self.set_operand_dropped(&bcx, source);
                                OperandValue::FatPtr(lldata, llextra)
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
                        if let OperandValue::FatPtr(data_ptr, meta_ptr) = operand.val {
                            if common::type_is_fat_ptr(bcx.tcx(), cast_ty) {
                                let ll_cft = ll_cast_ty.field_types();
                                let ll_fft = ll_from_ty.field_types();
                                let data_cast = bcx.pointercast(data_ptr, ll_cft[0]);
                                assert_eq!(ll_cft[1].kind(), ll_fft[1].kind());
                                OperandValue::FatPtr(data_cast, meta_ptr)
                            } else { // cast to thin-ptr
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr and
                                // pointer-cast of that pointer to desired pointer type.
                                let llval = bcx.pointercast(data_ptr, ll_cast_ty);
                                OperandValue::Immediate(llval)
                            }
                        } else {
                            bug!("Unexpected non-FatPtr operand")
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
                    bcx.tcx().mk_region(ty::ReStatic),
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
                        val: OperandValue::FatPtr(tr_lvalue.llval,
                                                  tr_lvalue.llextra),
                        ty: ref_ty,
                    }
                };
                (bcx, operand)
            }

            mir::Rvalue::Len(ref lvalue) => {
                let tr_lvalue = self.trans_lvalue(&bcx, lvalue);
                let operand = OperandRef {
                    val: OperandValue::Immediate(self.lvalue_len(&bcx, tr_lvalue)),
                    ty: bcx.tcx().types.usize,
                };
                (bcx, operand)
            }

            mir::Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.trans_operand(&bcx, lhs);
                let rhs = self.trans_operand(&bcx, rhs);
                let llresult = if common::type_is_fat_ptr(bcx.tcx(), lhs.ty) {
                    match (lhs.val, rhs.val) {
                        (OperandValue::FatPtr(lhs_addr, lhs_extra),
                         OperandValue::FatPtr(rhs_addr, rhs_extra)) => {
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

            mir::Rvalue::Use(..) |
            mir::Rvalue::Repeat(..) |
            mir::Rvalue::Aggregate(..) |
            mir::Rvalue::Slice { .. } |
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
                    let llfn = declare::declare_cfn(bcx.ccx(), "fmod", fty);
                    if input_ty == tcx.types.f32 {
                        let lllhs = bcx.fpext(lhs, f64t);
                        let llrhs = bcx.fpext(rhs, f64t);
                        let llres = bcx.call(llfn, &[lllhs, llrhs], None);
                        bcx.fptrunc(llres, Type::f32(bcx.ccx()))
                    } else {
                        bcx.call(llfn, &[lhs, rhs], None)
                    }
                } else {
                    bcx.frem(lhs, rhs)
                }
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
}

pub fn rvalue_creates_operand<'tcx>(rvalue: &mir::Rvalue<'tcx>) -> bool {
    match *rvalue {
        mir::Rvalue::Ref(..) |
        mir::Rvalue::Len(..) |
        mir::Rvalue::Cast(..) | // (*)
        mir::Rvalue::BinaryOp(..) |
        mir::Rvalue::UnaryOp(..) |
        mir::Rvalue::Box(..) =>
            true,
        mir::Rvalue::Use(..) | // (**)
        mir::Rvalue::Repeat(..) |
        mir::Rvalue::Aggregate(..) |
        mir::Rvalue::Slice { .. } |
        mir::Rvalue::InlineAsm { .. } =>
            false,
    }

    // (*) this is only true if the type is suitable
    // (**) we need to zero-out the source operand after moving, so we are restricted to either
    // ensuring all users of `Use` zero it out themselves or not allowing to “create” operand for
    // it.
}
