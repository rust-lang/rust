// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::interpret::{ErrorHandled, read_target_uint};
use rustc_mir::const_eval::const_field;
use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;
use rustc::mir::interpret::{GlobalId, Pointer, Allocation, ConstValue};
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, HasDataLayout, LayoutOf, Size};
use common::CodegenCx;
use syntax::source_map::Span;
use value::Value;
use interfaces::*;

use super::FunctionCx;

pub fn const_alloc_to_llvm(cx: &CodegenCx<'ll, '_>, alloc: &Allocation) -> &'ll Value {
    let mut llvals = Vec::with_capacity(alloc.relocations.len() + 1);
    let dl = cx.data_layout();
    let pointer_size = dl.pointer_size.bytes() as usize;

    let mut next_offset = 0;
    for &(offset, ((), alloc_id)) in alloc.relocations.iter() {
        let offset = offset.bytes();
        assert_eq!(offset as usize as u64, offset);
        let offset = offset as usize;
        if offset > next_offset {
            llvals.push(cx.const_bytes(&alloc.bytes[next_offset..offset]));
        }
        let ptr_offset = read_target_uint(
            dl.endian,
            &alloc.bytes[offset..(offset + pointer_size)],
        ).expect("const_alloc_to_llvm: could not read relocation pointer") as u64;
        llvals.push(cx.scalar_to_backend(
            Pointer::new(alloc_id, Size::from_bytes(ptr_offset)).into(),
            &layout::Scalar {
                value: layout::Primitive::Pointer,
                valid_range: 0..=!0
            },
            cx.type_i8p()
        ));
        next_offset = offset + pointer_size;
    }
    if alloc.bytes.len() >= next_offset {
        llvals.push(cx.const_bytes(&alloc.bytes[next_offset ..]));
    }

    cx.const_struct(&llvals, true)
}

pub fn codegen_static_initializer(
    cx: &CodegenCx<'ll, 'tcx>,
    def_id: DefId,
) -> Result<(&'ll Value, &'tcx Allocation), ErrorHandled> {
    let instance = ty::Instance::mono(cx.tcx, def_id);
    let cid = GlobalId {
        instance,
        promoted: None,
    };
    let param_env = ty::ParamEnv::reveal_all();
    let static_ = cx.tcx.const_eval(param_env.and(cid))?;

    let alloc = match static_.val {
        ConstValue::ByRef(_, alloc, n) if n.bytes() == 0 => alloc,
        _ => bug!("static const eval returned {:#?}", static_),
    };
    Ok((const_alloc_to_llvm(cx, alloc), alloc))
}

impl<'a, 'tcx: 'a, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    fn fully_evaluate(
        &mut self,
        bx: &Bx,
        constant: &'tcx ty::Const<'tcx>,
    ) -> Result<&'tcx ty::Const<'tcx>, ErrorHandled> {
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
        bx: &Bx,
        constant: &mir::Constant<'tcx>,
    ) -> Result<&'tcx ty::Const<'tcx>, ErrorHandled> {
        let c = self.monomorphize(&constant.literal);
        self.fully_evaluate(bx, c)
    }

    /// process constant containing SIMD shuffle indices
    pub fn simd_shuffle_indices(
        &mut self,
        bx: &Bx,
        span: Span,
        ty: Ty<'tcx>,
        constant: Result<&'tcx ty::Const<'tcx>, ErrorHandled>,
    ) -> (Bx::Value, Ty<'tcx>) {
        constant
            .and_then(|c| {
                let field_ty = c.ty.builtin_index().unwrap();
                let fields = match c.ty.sty {
                    ty::Array(_, n) => n.unwrap_usize(bx.tcx()),
                    ref other => bug!("invalid simd shuffle type: {}", other),
                };
                let values: Result<Vec<_>, ErrorHandled> = (0..fields).map(|field| {
                    let field = const_field(
                        bx.tcx(),
                        ty::ParamEnv::reveal_all(),
                        self.instance,
                        None,
                        mir::Field::new(field as usize),
                        c,
                    )?;
                    if let Some(prim) = field.val.try_to_scalar() {
                        let layout = bx.cx().layout_of(field_ty);
                        let scalar = match layout.abi {
                            layout::Abi::Scalar(ref x) => x,
                            _ => bug!("from_const: invalid ByVal layout: {:#?}", layout)
                        };
                        Ok(bx.cx().scalar_to_backend(
                            prim, scalar,
                            bx.cx().immediate_backend_type(layout),
                        ))
                    } else {
                        bug!("simd shuffle field {:?}", field)
                    }
                }).collect();
                let llval = bx.cx().const_struct(&values?, false);
                Ok((llval, c.ty))
            })
            .unwrap_or_else(|_| {
                bx.tcx().sess.span_err(
                    span,
                    "could not evaluate shuffle_indices at compile time",
                );
                // We've errored, so we don't have to produce working code.
                let ty = self.monomorphize(&ty);
                let llty = bx.cx().backend_type(bx.cx().layout_of(ty));
                (bx.cx().const_undef(llty), ty)
            })
    }
}
