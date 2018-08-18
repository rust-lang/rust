// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm;
use rustc::mir::interpret::ConstEvalErr;
use rustc_mir::interpret::{read_target_uint, const_val_field};
use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::sync::Lrc;
use rustc::mir::interpret::{GlobalId, Pointer, Scalar, Allocation, ConstValue, AllocType};
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, HasDataLayout, LayoutOf, Size};
use builder::Builder;
use common::{CodegenCx};
use common::{C_bytes, C_struct, C_uint_big, C_undef, C_usize};
use consts;
use type_of::LayoutLlvmExt;
use type_::Type;
use syntax::ast::Mutability;
use syntax::source_map::Span;
use value::Value;

use super::super::callee;
use super::FunctionCx;

pub fn scalar_to_llvm(
    cx: &CodegenCx<'ll, '_>,
    cv: Scalar,
    layout: &layout::Scalar,
    llty: &'ll Type,
) -> &'ll Value {
    let bitsize = if layout.is_bool() { 1 } else { layout.value.size(cx).bits() };
    match cv {
        Scalar::Bits { size: 0, .. } => {
            assert_eq!(0, layout.value.size(cx).bytes());
            C_undef(Type::ix(cx, 0))
        },
        Scalar::Bits { bits, size } => {
            assert_eq!(size as u64, layout.value.size(cx).bytes());
            let llval = C_uint_big(Type::ix(cx, bitsize), bits);
            if layout.value == layout::Pointer {
                unsafe { llvm::LLVMConstIntToPtr(llval, llty) }
            } else {
                consts::bitcast(llval, llty)
            }
        },
        Scalar::Ptr(ptr) => {
            let alloc_type = cx.tcx.alloc_map.lock().get(ptr.alloc_id);
            let base_addr = match alloc_type {
                Some(AllocType::Memory(alloc)) => {
                    let init = const_alloc_to_llvm(cx, alloc);
                    if alloc.runtime_mutability == Mutability::Mutable {
                        consts::addr_of_mut(cx, init, alloc.align, None)
                    } else {
                        consts::addr_of(cx, init, alloc.align, None)
                    }
                }
                Some(AllocType::Function(fn_instance)) => {
                    callee::get_fn(cx, fn_instance)
                }
                Some(AllocType::Static(def_id)) => {
                    assert!(cx.tcx.is_static(def_id).is_some());
                    consts::get_static(cx, def_id)
                }
                None => bug!("missing allocation {:?}", ptr.alloc_id),
            };
            let llval = unsafe { llvm::LLVMConstInBoundsGEP(
                consts::bitcast(base_addr, Type::i8p(cx)),
                &C_usize(cx, ptr.offset.bytes()),
                1,
            ) };
            if layout.value != layout::Pointer {
                unsafe { llvm::LLVMConstPtrToInt(llval, llty) }
            } else {
                consts::bitcast(llval, llty)
            }
        }
    }
}

pub fn const_alloc_to_llvm(cx: &CodegenCx<'ll, '_>, alloc: &Allocation) -> &'ll Value {
    let mut llvals = Vec::with_capacity(alloc.relocations.len() + 1);
    let layout = cx.data_layout();
    let pointer_size = layout.pointer_size.bytes() as usize;

    let mut next_offset = 0;
    for &(offset, alloc_id) in alloc.relocations.iter() {
        let offset = offset.bytes();
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            llvals.push(C_bytes(cx, &alloc.bytes[next_offset..offset]));
        }
        let ptr_offset = read_target_uint(
            layout.endian,
            &alloc.bytes[offset..(offset + pointer_size)],
        ).expect("const_alloc_to_llvm: could not read relocation pointer") as u64;
        llvals.push(scalar_to_llvm(
            cx,
            Pointer { alloc_id, offset: Size::from_bytes(ptr_offset) }.into(),
            &layout::Scalar {
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

pub fn codegen_static_initializer(
    cx: &CodegenCx<'ll, 'tcx>,
    def_id: DefId,
) -> Result<(&'ll Value, &'tcx Allocation), Lrc<ConstEvalErr<'tcx>>> {
    let instance = ty::Instance::mono(cx.tcx, def_id);
    let cid = GlobalId {
        instance,
        promoted: None,
    };
    let param_env = ty::ParamEnv::reveal_all();
    let static_ = cx.tcx.const_eval(param_env.and(cid))?;

    let alloc = match static_.val {
        ConstValue::ByRef(alloc, n) if n.bytes() == 0 => alloc,
        _ => bug!("static const eval returned {:#?}", static_),
    };
    Ok((const_alloc_to_llvm(cx, alloc), alloc))
}

impl FunctionCx<'a, 'll, 'tcx> {
    fn fully_evaluate(
        &mut self,
        bx: &Builder<'a, 'll, 'tcx>,
        constant: &'tcx ty::Const<'tcx>,
    ) -> Result<&'tcx ty::Const<'tcx>, Lrc<ConstEvalErr<'tcx>>> {
        match constant.val {
            ConstValue::Unevaluated(def_id, ref substs) => {
                let tcx = bx.tcx();
                let param_env = ty::ParamEnv::reveal_all();
                let instance = ty::Instance::resolve(tcx, param_env, def_id, substs).unwrap();
                let cid = GlobalId {
                    instance,
                    promoted: None,
                };
                tcx.const_eval(param_env.and(cid))
            },
            _ => Ok(constant),
        }
    }

    pub fn eval_mir_constant(
        &mut self,
        bx: &Builder<'a, 'll, 'tcx>,
        constant: &mir::Constant<'tcx>,
    ) -> Result<&'tcx ty::Const<'tcx>, Lrc<ConstEvalErr<'tcx>>> {
        let c = self.monomorphize(&constant.literal);
        self.fully_evaluate(bx, c)
    }

    /// process constant containing SIMD shuffle indices
    pub fn simd_shuffle_indices(
        &mut self,
        bx: &Builder<'a, 'll, 'tcx>,
        span: Span,
        ty: Ty<'tcx>,
        constant: Result<&'tcx ty::Const<'tcx>, Lrc<ConstEvalErr<'tcx>>>,
    ) -> (&'ll Value, Ty<'tcx>) {
        constant
            .and_then(|c| {
                let field_ty = c.ty.builtin_index().unwrap();
                let fields = match c.ty.sty {
                    ty::TyArray(_, n) => n.unwrap_usize(bx.tcx()),
                    ref other => bug!("invalid simd shuffle type: {}", other),
                };
                let values: Result<Vec<_>, Lrc<_>> = (0..fields).map(|field| {
                    let field = const_val_field(
                        bx.tcx(),
                        ty::ParamEnv::reveal_all(),
                        self.instance,
                        None,
                        mir::Field::new(field as usize),
                        c,
                    )?;
                    if let Some(prim) = field.val.try_to_scalar() {
                        let layout = bx.cx.layout_of(field_ty);
                        let scalar = match layout.abi {
                            layout::Abi::Scalar(ref x) => x,
                            _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                        };
                        Ok(scalar_to_llvm(
                            bx.cx, prim, scalar,
                            layout.immediate_llvm_type(bx.cx),
                        ))
                    } else {
                        bug!("simd shuffle field {:?}", field)
                    }
                }).collect();
                let llval = C_struct(bx.cx, &values?, false);
                Ok((llval, c.ty))
            })
            .unwrap_or_else(|e| {
                e.report_as_error(
                    bx.tcx().at(span),
                    "could not evaluate shuffle_indices at compile time",
                );
                // We've errored, so we don't have to produce working code.
                let ty = self.monomorphize(&ty);
                let llty = bx.cx.layout_of(ty).llvm_type(bx.cx);
                (C_undef(llty), ty)
            })
    }
}
