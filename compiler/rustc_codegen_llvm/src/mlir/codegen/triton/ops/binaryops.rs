/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use itertools::Itertools;
use melior::ir::operation::OperationLike;
use melior::ir::r#type::{IntegerType, RankedTensorType};
use melior::ir::{
    BlockLike, BlockRef, Location, Operation, ShapedTypeLike, TypeLike, Value, ValueLike,
};
use rustc_middle::mir::{BasicBlock, Body, CallSource, Operand, Place, UnwindAction};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_mlir::shared::arith::{
    FpPredicate, Predicate, create_addf, create_addi, create_cmpf, create_cmpi, create_extsi,
    create_mulf, create_muli, create_muli_tensor, create_subf, create_subi,
};
use rustc_mlir::shared::builtin::{tensor_type, tensor_type_like};
use rustc_mlir::triton::tensor::add_ptr;
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::{CodegenState, TritonCodegen};
use crate::mlir::errors::MlirError;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_mul_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 2, "TritonCodegen::codegen_mul_call: args length must be 2");

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        println!("[DEBUG] TritonCodegen::codegen_mul_call: arg0: {:?}", arg0);
        println!("[DEBUG] TritonCodegen::codegen_mul_call: arg1: {:?}", arg1);

        let arg0_value = self.codegen_operand(
            tcx,
            instance,
            arg0,
            arg0.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let arg1_value = self.codegen_operand(
            tcx,
            instance,
            arg1,
            arg1.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        self.codegen_mul(tcx, location, arg0_value, arg1_value, mlir_block)
    }

    pub fn codegen_add_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 2, "TritonCodegen::codegen_add_call: args length must be 2");

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        println!("[DEBUG] TritonCodegen::codegen_add_call: arg0: {:?}", arg0);
        println!("[DEBUG] TritonCodegen::codegen_add_call: arg1: {:?}", arg1);

        let lhs = self.codegen_operand(
            tcx,
            instance,
            arg0,
            arg0.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            arg1,
            arg1.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        self.codegen_add(tcx, location, lhs, rhs, mlir_block)
    }

    pub fn codegen_sub_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 2, "TritonCodegen::codegen_sub_call: args length must be 2");

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        let lhs = self.codegen_operand(
            tcx,
            instance,
            arg0,
            arg0.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            arg1,
            arg1.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        self.codegen_sub(tcx, location, lhs, rhs, mlir_block)
    }

    pub fn codegen_lt_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 2, "TritonCodegen::codegen_lt_call: args length must be 2");

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        let lhs = self.codegen_operand(
            tcx,
            instance,
            arg0,
            arg0.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            arg1,
            arg1.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        self.codegen_cmpi(tcx, Predicate::SLT, location, lhs, rhs, mlir_block)
    }

    pub fn codegen_cmpi<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        predicate: Predicate,
        location: Location<'a>,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs_is_tensor = lhs.r#type().is_tensor();
        let rhs_is_tensor = rhs.r#type().is_tensor();

        let (lhs, rhs) = match (lhs_is_tensor, rhs_is_tensor) {
            (true, true) => (lhs, rhs),
            (true, false) => (lhs, self.like_tensor(tcx, location, lhs, rhs, mlir_block)?),
            (false, true) => (self.like_tensor(tcx, location, rhs, lhs, mlir_block)?, rhs),
            (false, false) => {
                todo!("TritonCodegen::codegen_lt: {:?}-> {:?} {:?}", lhs, lhs.r#type(), rhs,)
            }
        };

        let result_ty = tensor_type_like(
            lhs.r#type()
                .try_into()
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?,
            IntegerType::new(self.module.context(), 1).into(),
        )
        .map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        let lt_op: Operation<'a> =
            create_cmpi(self.module.context(), location, predicate, lhs, rhs, result_ty.into())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = lt_op.result(0).expect("LT operation result not found");
        mlir_block.append_operation(lt_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_cmpf<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        predicate: FpPredicate,
        location: Location<'a>,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs_is_tensor = lhs.r#type().is_tensor();
        let rhs_is_tensor = rhs.r#type().is_tensor();

        let (lhs, rhs) = match (lhs_is_tensor, rhs_is_tensor) {
            (true, true) => (lhs, rhs),
            (true, false) => (lhs, self.like_tensor(tcx, location, lhs, rhs, mlir_block)?),
            (false, true) => (self.like_tensor(tcx, location, rhs, lhs, mlir_block)?, rhs),
            (false, false) => {
                todo!("TritonCodegen::codegen_cmpf scalar: {:?} {:?}", lhs.r#type(), rhs.r#type())
            }
        };

        let result_ty = tensor_type_like(
            lhs.r#type()
                .try_into()
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?,
            IntegerType::new(self.module.context(), 1).into(),
        )
        .map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        let cmp_op: Operation<'a> =
            create_cmpf(self.module.context(), location, predicate, lhs, rhs, result_ty.into())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = cmp_op.result(0).expect("cmpf operation result not found");
        mlir_block.append_operation(cmp_op);
        Ok(Some(result.into()))
    }

    /// Dispatch a two-operand comparison to `arith.cmpf` (floats) or `arith.cmpi` (integers).
    fn codegen_binary_cmp<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        fp_pred: FpPredicate,
        int_pred: Predicate,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 2, "comparison requires 2 args");
        let arg0 = &args[0].node;
        let arg1 = &args[1].node;
        let lhs = self.codegen_operand(tcx, instance, arg0, arg0.ty(mir, tcx), location, mlir_block, state)?;
        let rhs = self.codegen_operand(tcx, instance, arg1, arg1.ty(mir, tcx), location, mlir_block, state)?;

        // Determine element type (unwrap tensor if needed).
        let lhs_ty = lhs.r#type();
        let elem_ty = if lhs_ty.is_tensor() {
            RankedTensorType::try_from(lhs_ty)
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?
                .element()
        } else {
            lhs_ty
        };

        if elem_ty.is_float() {
            self.codegen_cmpf(tcx, fp_pred, location, lhs, rhs, mlir_block)
        } else {
            self.codegen_cmpi(tcx, int_pred, location, lhs, rhs, mlir_block)
        }
    }

    pub fn codegen_gt_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OGT, Predicate::SGT, location, mlir_block, state)
    }

    pub fn codegen_ge_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OGE, Predicate::SGE, location, mlir_block, state)
    }

    pub fn codegen_triton_lt_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OLT, Predicate::SLT, location, mlir_block, state)
    }

    pub fn codegen_le_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OLE, Predicate::SLE, location, mlir_block, state)
    }

    pub fn codegen_eq_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OEQ, Predicate::EQ, location, mlir_block, state)
    }

    pub fn codegen_ne_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::ONE, Predicate::NE, location, mlir_block, state)
    }

    // Scalar comparison handlers — (tensor, scalar) args; broadcasting handled by codegen_binary_cmp.
    pub fn codegen_lt_scalar_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OLT, Predicate::SLT, location, mlir_block, state)
    }

    pub fn codegen_le_scalar_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OLE, Predicate::SLE, location, mlir_block, state)
    }

    pub fn codegen_gt_scalar_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OGT, Predicate::SGT, location, mlir_block, state)
    }

    pub fn codegen_ge_scalar_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OGE, Predicate::SGE, location, mlir_block, state)
    }

    pub fn codegen_eq_scalar_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::OEQ, Predicate::EQ, location, mlir_block, state)
    }

    pub fn codegen_ne_scalar_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        self.codegen_binary_cmp(tcx, instance, mir, args, FpPredicate::ONE, Predicate::NE, location, mlir_block, state)
    }

    pub fn codegen_mul<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        location: Location<'a>,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        let lhs_is_tensor = lhs_ty.is_tensor();
        let rhs_is_tensor = rhs_ty.is_tensor();

        let (lhs, rhs) = match (lhs_is_tensor, rhs_is_tensor) {
            (true, true) => (lhs, rhs),
            (true, false) => (lhs, self.like_tensor(tcx, location, lhs, rhs, mlir_block)?),
            (false, true) => (self.like_tensor(tcx, location, rhs, lhs, mlir_block)?, rhs),
            (false, false) => {
                if lhs_ty.is_integer() {
                    let mul_op: Operation =
                        create_muli(self.module.context(), location, lhs, rhs)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();
                    let result = mul_op.result(0).expect("Mul operation result not found");
                    mlir_block.append_operation(mul_op);
                    return Ok(Some(result.into()));
                }
                todo!("TritonCodegen::codegen_mul scalar: {:?} {:?}", lhs_ty, rhs_ty)
            }
        };

        let lhs_ty: RankedTensorType<'a> = lhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;

        let rhs_ty: RankedTensorType<'a> = rhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;

        let mul_op: Operation<'a> =
            if lhs_ty.element().is_integer() && rhs_ty.element().is_integer() {
                create_muli_tensor(self.module.context(), location, lhs, rhs)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
            } else {
                create_mulf(self.module.context(), location, lhs, rhs)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
            };

        let result = mul_op.result(0).expect("Mul operation result not found");
        mlir_block.append_operation(mul_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_sub<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        location: Location<'a>,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        let lhs_is_tensor = lhs_ty.is_tensor();
        let rhs_is_tensor = rhs_ty.is_tensor();

        let (lhs, rhs) = match (lhs_is_tensor, rhs_is_tensor) {
            (true, true) => (lhs, rhs),
            (true, false) => (lhs, self.like_tensor(tcx, location, lhs, rhs, mlir_block)?),
            (false, true) => (self.like_tensor(tcx, location, rhs, lhs, mlir_block)?, rhs),
            (false, false) => {
                if lhs_ty.is_integer() {
                    let sub_op: Operation<'a> =
                        create_subi(self.module.context(), location, lhs, rhs)
                            .map_err(|e| MlirError::CreateOperation { err: e })?;
                    let result = sub_op.result(0).expect("Sub operation result not found");
                    mlir_block.append_operation(sub_op);
                    return Ok(Some(result.into()));
                }
                todo!("TritonCodegen::codegen_sub scalar float: {:?} {:?}", lhs_ty, rhs_ty)
            }
        };

        let lhs_ty: RankedTensorType<'a> = lhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;

        let rhs_ty: RankedTensorType<'a> = rhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;

        let sub_op: Operation<'a> =
            if lhs_ty.element().is_integer() && rhs_ty.element().is_integer() {
                create_subi(self.module.context(), location, lhs, rhs)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
            } else {
                create_subf(self.module.context(), location, lhs, rhs)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
            };

        let result = sub_op.result(0).expect("Sub operation result not found");
        mlir_block.append_operation(sub_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_add<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        location: Location<'a>,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        let lhs_is_tensor = lhs_ty.is_tensor();
        let rhs_is_tensor = rhs_ty.is_tensor();

        let (lhs, rhs) = match (lhs_is_tensor, rhs_is_tensor) {
            (true, true) => (lhs, rhs),
            (true, false) => (lhs, self.like_tensor(tcx, location, lhs, rhs, mlir_block)?),
            (false, true) => (self.like_tensor(tcx, location, rhs, lhs, mlir_block)?, rhs),
            (false, false) => todo!(
                "TritonCodegen::codegen_add: {:?}-> {:?} {:?}-> {:?}",
                lhs,
                lhs_ty,
                rhs,
                rhs_ty
            ),
        };

        let lhs_ty: RankedTensorType<'a> = lhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;

        let rhs_ty: RankedTensorType<'a> = rhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;

        let add_op: Operation<'a> =
            if lhs_ty.element().is_integer() && rhs_ty.element().is_integer() {
                create_addi(self.module.context(), location, lhs, rhs)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into()
            } else {
                create_addf(self.module.context(), location, lhs, rhs)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into()
            };

        let result = add_op.result(0).expect("Add operation result not found");

        mlir_block.append_operation(add_op);
        Ok(Some(result.into()))
    }
}
