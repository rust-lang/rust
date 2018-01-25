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
use rustc::middle::const_val::{ConstVal, ConstEvalErr};
use rustc_mir::interpret::{read_target_uint, const_val_field};
use rustc::hir::def_id::DefId;
use rustc::traits;
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;
use rustc::mir::interpret::{Allocation, GlobalId, MemoryPointer, PrimVal, Value as MiriValue};
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, HasDataLayout, LayoutOf, Scalar};
use builder::Builder;
use common::{CodegenCx};
use common::{C_bytes, C_struct, C_uint_big, C_undef, C_usize};
use consts;
use type_of::LayoutLlvmExt;
use type_::Type;

use super::super::callee;
use super::FunctionCx;

pub fn primval_to_llvm(cx: &CodegenCx,
                       cv: PrimVal,
                       scalar: &Scalar,
                       llty: Type) -> ValueRef {
    let bits = if scalar.is_bool() { 1 } else { scalar.value.size(cx).bits() };
    match cv {
        PrimVal::Undef => C_undef(Type::ix(cx, bits)),
        PrimVal::Bytes(b) => {
            let llval = C_uint_big(Type::ix(cx, bits), b);
            if scalar.value == layout::Pointer {
                unsafe { llvm::LLVMConstIntToPtr(llval, llty.to_ref()) }
            } else {
                consts::bitcast(llval, llty)
            }
        },
        PrimVal::Ptr(ptr) => {
            if let Some(fn_instance) = cx.tcx.interpret_interner.get_fn(ptr.alloc_id) {
                callee::get_fn(cx, fn_instance)
            } else {
                let static_ = cx
                    .tcx
                    .interpret_interner
                    .get_corresponding_static_def_id(ptr.alloc_id);
                let base_addr = if let Some(def_id) = static_ {
                    assert!(cx.tcx.is_static(def_id).is_some());
                    consts::get_static(cx, def_id)
                } else if let Some(alloc) = cx.tcx.interpret_interner
                                              .get_alloc(ptr.alloc_id) {
                    let init = global_initializer(cx, alloc);
                    if alloc.mutable {
                        consts::addr_of_mut(cx, init, alloc.align, "byte_str")
                    } else {
                        consts::addr_of(cx, init, alloc.align, "byte_str")
                    }
                } else {
                    bug!("missing allocation {:?}", ptr.alloc_id);
                };

                let llval = unsafe { llvm::LLVMConstInBoundsGEP(
                    consts::bitcast(base_addr, Type::i8p(cx)),
                    &C_usize(cx, ptr.offset),
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

pub fn global_initializer(cx: &CodegenCx, alloc: &Allocation) -> ValueRef {
    let mut llvals = Vec::with_capacity(alloc.relocations.len() + 1);
    let layout = cx.data_layout();
    let pointer_size = layout.pointer_size.bytes() as usize;

    let mut next_offset = 0;
    for (&offset, &alloc_id) in &alloc.relocations {
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            llvals.push(C_bytes(cx, &alloc.bytes[next_offset..offset]));
        }
        let ptr_offset = read_target_uint(
            layout.endian,
            &alloc.bytes[offset..(offset + pointer_size)],
        ).expect("global_initializer: could not read relocation pointer") as u64;
        llvals.push(primval_to_llvm(
            cx,
            PrimVal::Ptr(MemoryPointer { alloc_id, offset: ptr_offset }),
            &Scalar {
                value: layout::Primitive::Pointer,
                valid_range: 0..=!0
            },
            Type::i8p(cx)
        ));
        next_offset = offset + pointer_size;
    }
    if alloc.bytes.len() >= next_offset {
        llvals.push(C_bytes(cx, &alloc.bytes[next_offset ..]));
    }

    C_struct(cx, &llvals, true)
}

pub fn trans_static_initializer<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    def_id: DefId)
    -> Result<ValueRef, ConstEvalErr<'tcx>>
{
    let instance = ty::Instance::mono(cx.tcx, def_id);
    let cid = GlobalId {
        instance,
        promoted: None
    };
    let param_env = ty::ParamEnv::empty(traits::Reveal::All);
    cx.tcx.const_eval(param_env.and(cid))?;

    let alloc_id = cx
        .tcx
        .interpret_interner
        .get_cached(def_id)
        .expect("global not cached");

    let alloc = cx
        .tcx
        .interpret_interner
        .get_alloc(alloc_id)
        .expect("miri allocation never successfully created");
    Ok(global_initializer(cx, alloc))
}

impl<'a, 'tcx> FunctionCx<'a, 'tcx> {
    fn const_to_miri_value(
        &mut self,
        bx: &Builder<'a, 'tcx>,
        constant: &'tcx ty::Const<'tcx>,
    ) -> Result<MiriValue, ConstEvalErr<'tcx>> {
        match constant.val {
            ConstVal::Unevaluated(def_id, ref substs) => {
                let tcx = bx.tcx();
                let param_env = ty::ParamEnv::empty(traits::Reveal::All);
                let instance = ty::Instance::resolve(tcx, param_env, def_id, substs).unwrap();
                let cid = GlobalId {
                    instance,
                    promoted: None,
                };
                let c = tcx.const_eval(param_env.and(cid))?;
                self.const_to_miri_value(bx, c)
            },
            ConstVal::Value(miri_val) => Ok(miri_val),
        }
    }

    pub fn mir_constant_to_miri_value(
        &mut self,
        bx: &Builder<'a, 'tcx>,
        constant: &mir::Constant<'tcx>,
    ) -> Result<MiriValue, ConstEvalErr<'tcx>> {
        match constant.literal {
            mir::Literal::Promoted { index } => {
                let param_env = ty::ParamEnv::empty(traits::Reveal::All);
                let cid = mir::interpret::GlobalId {
                    instance: self.instance,
                    promoted: Some(index),
                };
                bx.tcx().const_eval(param_env.and(cid))
            }
            mir::Literal::Value { value } => {
                Ok(self.monomorphize(&value))
            }
        }.and_then(|c| self.const_to_miri_value(bx, c))
    }

    // Old version of trans_constant now used just for SIMD shuffle
    pub fn remove_me_shuffle_indices(&mut self,
                                      bx: &Builder<'a, 'tcx>,
                                      constant: &mir::Constant<'tcx>)
                                      -> (ValueRef, Ty<'tcx>)
    {
        let layout = bx.cx.layout_of(constant.ty);
        self.mir_constant_to_miri_value(bx, constant)
            .and_then(|c| {
                let llval = match c {
                    MiriValue::ByVal(val) => {
                        let scalar = match layout.abi {
                            layout::Abi::Scalar(ref x) => x,
                            _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                        };
                        primval_to_llvm(bx.cx, val, scalar, layout.immediate_llvm_type(bx.cx))
                    },
                    MiriValue::ByValPair(a_val, b_val) => {
                        let (a_scalar, b_scalar) = match layout.abi {
                            layout::Abi::ScalarPair(ref a, ref b) => (a, b),
                            _ => bug!("from_const: invalid ByValPair layout: {:#?}", layout)
                        };
                        let a_llval = primval_to_llvm(
                            bx.cx,
                            a_val,
                            a_scalar,
                            layout.scalar_pair_element_llvm_type(bx.cx, 0),
                        );
                        let b_llval = primval_to_llvm(
                            bx.cx,
                            b_val,
                            b_scalar,
                            layout.scalar_pair_element_llvm_type(bx.cx, 1),
                        );
                        C_struct(bx.cx, &[a_llval, b_llval], false)
                    },
                    MiriValue::ByRef(..) => {
                        let field_ty = constant.ty.builtin_index().unwrap();
                        let fields = match constant.ty.sty {
                            ty::TyArray(_, n) => n.val.unwrap_u64(),
                            ref other => bug!("invalid simd shuffle type: {}", other),
                        };
                        let values: Result<Vec<ValueRef>, _> = (0..fields).map(|field| {
                            let field = const_val_field(
                                bx.tcx(),
                                ty::ParamEnv::empty(traits::Reveal::All),
                                self.instance,
                                None,
                                mir::Field::new(field as usize),
                                c,
                                constant.ty,
                            )?;
                            match field.val {
                                ConstVal::Value(MiriValue::ByVal(prim)) => {
                                    let layout = bx.cx.layout_of(field_ty);
                                    let scalar = match layout.abi {
                                        layout::Abi::Scalar(ref x) => x,
                                        _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                                    };
                                    Ok(primval_to_llvm(
                                        bx.cx, prim, scalar,
                                        layout.immediate_llvm_type(bx.cx),
                                    ))
                                },
                                other => bug!("simd shuffle field {:?}, {}", other, constant.ty),
                            }
                        }).collect();
                        C_struct(bx.cx, &values?, false)
                    },
                };
                Ok((llval, constant.ty))
            })
            .unwrap_or_else(|e| {
                e.report(bx.tcx(), constant.span, "shuffle_indices");
                // We've errored, so we don't have to produce working code.
                let ty = self.monomorphize(&constant.ty);
                let llty = bx.cx.layout_of(ty).llvm_type(bx.cx);
                (C_undef(llty), ty)
            })
    }
}
