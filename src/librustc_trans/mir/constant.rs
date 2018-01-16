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
use rustc_const_math::{ConstInt, ConstMathErr};
use rustc::middle::const_val::{ConstVal, ConstEvalErr};
use rustc_mir::interpret::{read_target_uint, const_val_field};
use rustc::hir::def_id::DefId;
use rustc::traits;
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;
use rustc::mir::interpret::{Allocation, GlobalId, MemoryPointer, PrimVal, Value as MiriValue};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::layout::{self, HasDataLayout, LayoutOf, Scalar};
use base;
use builder::Builder;
use common::{CodegenCx};
use common::{C_bytes, C_struct, C_uint_big, C_undef, C_usize};
use common::const_to_opt_u128;
use consts;
use type_of::LayoutLlvmExt;
use type_::Type;

use super::super::callee;
use super::MirContext;

fn to_const_int(value: ValueRef, t: Ty, tcx: TyCtxt) -> Option<ConstInt> {
    match t.sty {
        ty::TyInt(int_type) => const_to_opt_u128(value, true)
            .and_then(|input| ConstInt::new_signed(input as i128, int_type,
                                                   tcx.sess.target.isize_ty)),
        ty::TyUint(uint_type) => const_to_opt_u128(value, false)
            .and_then(|input| ConstInt::new_unsigned(input, uint_type,
                                                     tcx.sess.target.usize_ty)),
        _ => None

    }
}

pub fn const_scalar_binop(op: mir::BinOp,
                          lhs: ValueRef,
                          rhs: ValueRef,
                          input_ty: Ty) -> ValueRef {
    assert!(!input_ty.is_simd());
    let is_float = input_ty.is_fp();
    let signed = input_ty.is_signed();

    unsafe {
        match op {
            mir::BinOp::Add if is_float => llvm::LLVMConstFAdd(lhs, rhs),
            mir::BinOp::Add             => llvm::LLVMConstAdd(lhs, rhs),

            mir::BinOp::Sub if is_float => llvm::LLVMConstFSub(lhs, rhs),
            mir::BinOp::Sub             => llvm::LLVMConstSub(lhs, rhs),

            mir::BinOp::Mul if is_float => llvm::LLVMConstFMul(lhs, rhs),
            mir::BinOp::Mul             => llvm::LLVMConstMul(lhs, rhs),

            mir::BinOp::Div if is_float => llvm::LLVMConstFDiv(lhs, rhs),
            mir::BinOp::Div if signed   => llvm::LLVMConstSDiv(lhs, rhs),
            mir::BinOp::Div             => llvm::LLVMConstUDiv(lhs, rhs),

            mir::BinOp::Rem if is_float => llvm::LLVMConstFRem(lhs, rhs),
            mir::BinOp::Rem if signed   => llvm::LLVMConstSRem(lhs, rhs),
            mir::BinOp::Rem             => llvm::LLVMConstURem(lhs, rhs),

            mir::BinOp::BitXor => llvm::LLVMConstXor(lhs, rhs),
            mir::BinOp::BitAnd => llvm::LLVMConstAnd(lhs, rhs),
            mir::BinOp::BitOr  => llvm::LLVMConstOr(lhs, rhs),
            mir::BinOp::Shl    => {
                let rhs = base::cast_shift_const_rhs(op.to_hir_binop(), lhs, rhs);
                llvm::LLVMConstShl(lhs, rhs)
            }
            mir::BinOp::Shr    => {
                let rhs = base::cast_shift_const_rhs(op.to_hir_binop(), lhs, rhs);
                if signed { llvm::LLVMConstAShr(lhs, rhs) }
                else      { llvm::LLVMConstLShr(lhs, rhs) }
            }
            mir::BinOp::Eq | mir::BinOp::Ne |
            mir::BinOp::Lt | mir::BinOp::Le |
            mir::BinOp::Gt | mir::BinOp::Ge => {
                if is_float {
                    let cmp = base::bin_op_to_fcmp_predicate(op.to_hir_binop());
                    llvm::LLVMConstFCmp(cmp, lhs, rhs)
                } else {
                    let cmp = base::bin_op_to_icmp_predicate(op.to_hir_binop(),
                                                                signed);
                    llvm::LLVMConstICmp(cmp, lhs, rhs)
                }
            }
            mir::BinOp::Offset => unreachable!("BinOp::Offset in const-eval!")
        }
    }
}

pub fn const_scalar_checked_binop<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                            op: mir::BinOp,
                                            lllhs: ValueRef,
                                            llrhs: ValueRef,
                                            input_ty: Ty<'tcx>)
                                            -> Option<(ValueRef, bool)> {
    if let (Some(lhs), Some(rhs)) = (to_const_int(lllhs, input_ty, tcx),
                                     to_const_int(llrhs, input_ty, tcx)) {
        let result = match op {
            mir::BinOp::Add => lhs + rhs,
            mir::BinOp::Sub => lhs - rhs,
            mir::BinOp::Mul => lhs * rhs,
            mir::BinOp::Shl => lhs << rhs,
            mir::BinOp::Shr => lhs >> rhs,
            _ => {
                bug!("Operator `{:?}` is not a checkable operator", op)
            }
        };

        let of = match result {
            Ok(_) => false,
            Err(ConstMathErr::Overflow(_)) |
            Err(ConstMathErr::ShiftNegative) => true,
            Err(err) => {
                bug!("Operator `{:?}` on `{:?}` and `{:?}` errored: {}",
                     op, lhs, rhs, err.description());
            }
        };

        Some((const_scalar_binop(op, lllhs, llrhs, input_ty), of))
    } else {
        None
    }
}

pub fn primval_to_llvm(ccx: &CrateContext,
                       cv: PrimVal,
                       scalar: &Scalar,
                       llty: Type) -> ValueRef {
    let bits = if scalar.is_bool() { 1 } else { scalar.value.size(ccx).bits() };
    match cv {
        PrimVal::Undef => C_undef(Type::ix(ccx, bits)),
        PrimVal::Bytes(b) => {
            let llval = C_uint_big(Type::ix(ccx, bits), b);
            if scalar.value == layout::Pointer {
                unsafe { llvm::LLVMConstIntToPtr(llval, llty.to_ref()) }
            } else {
                consts::bitcast(llval, llty)
            }
        },
        PrimVal::Ptr(ptr) => {
            let interpret_interner = ccx.tcx().interpret_interner.borrow();
            if let Some(fn_instance) = interpret_interner.get_fn(ptr.alloc_id) {
                callee::get_fn(ccx, fn_instance)
            } else {
                let static_ = interpret_interner.get_corresponding_static_def_id(ptr.alloc_id);
                let base_addr = if let Some(def_id) = static_ {
                    assert!(ccx.tcx().is_static(def_id).is_some());
                    consts::get_static(ccx, def_id)
                } else if let Some(alloc) = interpret_interner.get_alloc(ptr.alloc_id) {
                    let init = global_initializer(ccx, alloc);
                    if alloc.mutable {
                        consts::addr_of_mut(ccx, init, alloc.align, "byte_str")
                    } else {
                        consts::addr_of(ccx, init, alloc.align, "byte_str")
                    }
                } else {
                    bug!("missing allocation {:?}", ptr.alloc_id);
                };

                let llval = unsafe { llvm::LLVMConstInBoundsGEP(
                    consts::bitcast(base_addr, Type::i8p(ccx)),
                    &C_usize(ccx, ptr.offset),
                    1,
                ) };
                if scalar.value != layout::Pointer {
                    unsafe { llvm::LLVMConstPtrToInt(llval, llty.to_ref()) }
                } else {
                    consts::bitcast(llval, llty)
                }
            }
        }
    }
}

pub fn global_initializer(ccx: &CrateContext, alloc: &Allocation) -> ValueRef {
    let mut llvals = Vec::with_capacity(alloc.relocations.len() + 1);
    let layout = ccx.data_layout();
    let pointer_size = layout.pointer_size.bytes() as usize;

    let mut next_offset = 0;
    for (&offset, &alloc_id) in &alloc.relocations {
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            llvals.push(C_bytes(ccx, &alloc.bytes[next_offset..offset]));
        }
        let ptr_offset = read_target_uint(
            layout.endian,
            &alloc.bytes[offset..(offset + pointer_size)],
        ).expect("global_initializer: could not read relocation pointer") as u64;
        llvals.push(primval_to_llvm(
            ccx,
            PrimVal::Ptr(MemoryPointer { alloc_id, offset: ptr_offset }),
            &Scalar {
                value: layout::Primitive::Pointer,
                valid_range: 0..=!0
            },
            Type::i8p(ccx)
        ));
        next_offset = offset + pointer_size;
    }
    if alloc.bytes.len() >= next_offset {
        llvals.push(C_bytes(ccx, &alloc.bytes[next_offset ..]));
    }

    C_struct(ccx, &llvals, true)
}

pub fn trans_static_initializer<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    def_id: DefId)
    -> Result<ValueRef, ConstEvalErr<'tcx>>
{
    let instance = ty::Instance::mono(ccx.tcx(), def_id);
    let cid = GlobalId {
        instance,
        promoted: None
    };
    let param_env = ty::ParamEnv::empty(traits::Reveal::All);
    ccx.tcx().const_eval(param_env.and(cid))?;

    let alloc_id = ccx
        .tcx()
        .interpret_interner
        .borrow()
        .get_cached(def_id)
        .expect("global not cached");

    let alloc = ccx
        .tcx()
        .interpret_interner
        .borrow()
        .get_alloc(alloc_id)
        .expect("miri allocation never successfully created");
    Ok(global_initializer(ccx, alloc))
}

impl<'a, 'tcx> FunctionCx<'a, 'tcx> {
    fn const_to_miri_value(
        &mut self,
        bcx: &Builder<'a, 'tcx>,
        constant: &'tcx ty::Const<'tcx>,
    ) -> Result<MiriValue, ConstEvalErr<'tcx>> {
        match constant.val {
            ConstVal::Unevaluated(def_id, ref substs) => {
                let tcx = bcx.tcx();
                let param_env = ty::ParamEnv::empty(traits::Reveal::All);
                let instance = ty::Instance::resolve(tcx, param_env, def_id, substs).unwrap();
                let cid = GlobalId {
                    instance,
                    promoted: None,
                };
                let c = tcx.const_eval(param_env.and(cid))?;
                self.const_to_miri_value(bcx, c)
            },
            ConstVal::Value(miri_val) => Ok(miri_val),
        }
    }

    pub fn mir_constant_to_miri_value(
        &mut self,
        bcx: &Builder<'a, 'tcx>,
        constant: &mir::Constant<'tcx>,
    ) -> Result<MiriValue, ConstEvalErr<'tcx>> {
        match constant.literal {
            mir::Literal::Promoted { index } => {
                let param_env = ty::ParamEnv::empty(traits::Reveal::All);
                let cid = mir::interpret::GlobalId {
                    instance: self.instance,
                    promoted: Some(index),
                };
                bcx.tcx().const_eval(param_env.and(cid))
            }
            mir::Literal::Value { value } => {
                Ok(self.monomorphize(&value))
            }
        }.and_then(|c| self.const_to_miri_value(bcx, c))
    }

    // Old version of trans_constant now used just for SIMD shuffle
    pub fn remove_me_shuffle_indices(&mut self,
                                      bcx: &Builder<'a, 'tcx>,
                                      constant: &mir::Constant<'tcx>)
                                      -> (ValueRef, Ty<'tcx>)
    {
        let layout = bcx.ccx.layout_of(constant.ty);
        self.mir_constant_to_miri_value(bcx, constant)
            .and_then(|c| {
                let llval = match c {
                    MiriValue::ByVal(val) => {
                        let scalar = match layout.abi {
                            layout::Abi::Scalar(ref x) => x,
                            _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                        };
                        primval_to_llvm(bcx.ccx, val, scalar, layout.immediate_llvm_type(bcx.ccx))
                    },
                    MiriValue::ByValPair(a_val, b_val) => {
                        let (a_scalar, b_scalar) = match layout.abi {
                            layout::Abi::ScalarPair(ref a, ref b) => (a, b),
                            _ => bug!("from_const: invalid ByValPair layout: {:#?}", layout)
                        };
                        let a_llval = primval_to_llvm(
                            bcx.ccx,
                            a_val,
                            a_scalar,
                            layout.scalar_pair_element_llvm_type(bcx.ccx, 0),
                        );
                        let b_llval = primval_to_llvm(
                            bcx.ccx,
                            b_val,
                            b_scalar,
                            layout.scalar_pair_element_llvm_type(bcx.ccx, 1),
                        );
                        C_struct(bcx.ccx, &[a_llval, b_llval], false)
                    },
                    MiriValue::ByRef(..) => {
                        let field_ty = constant.ty.builtin_index().unwrap();
                        let fields = match constant.ty.sty {
                            ty::TyArray(_, n) => n.val.unwrap_u64(),
                            ref other => bug!("invalid simd shuffle type: {}", other),
                        };
                        let values: Result<Vec<ValueRef>, _> = (0..fields).map(|field| {
                            let field = const_val_field(
                                bcx.tcx(),
                                ty::ParamEnv::empty(traits::Reveal::All),
                                self.instance,
                                None,
                                mir::Field::new(field as usize),
                                c,
                                constant.ty,
                            )?;
                            match field.val {
                                ConstVal::Value(MiriValue::ByVal(prim)) => {
                                    let layout = bcx.ccx.layout_of(field_ty);
                                    let scalar = match layout.abi {
                                        layout::Abi::Scalar(ref x) => x,
                                        _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                                    };
                                    Ok(primval_to_llvm(
                                        bcx.ccx, prim, scalar,
                                        layout.immediate_llvm_type(bcx.ccx),
                                    ))
                                },
                                other => bug!("simd shuffle field {:?}, {}", other, constant.ty),
                            }
                        }).collect();
                        C_struct(bcx.ccx, &values?, false)
                    },
                };
                Ok((llval, constant.ty))
            })
            .unwrap_or_else(|e| {
                e.report(bcx.tcx(), constant.span, "shuffle_indices");
                // We've errored, so we don't have to produce working code.
                let ty = self.monomorphize(&constant.ty);
                let llty = bcx.ccx.layout_of(ty).llvm_type(bcx.ccx);
                (C_undef(llty), ty)
            })
    }
}
