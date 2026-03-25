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
    Predicate, create_addf, create_addi, create_cmpi, create_extsi, create_muli,
};
use rustc_mlir::shared::builtin::{tensor_type, tensor_type_like};
use rustc_mlir::triton::tensor::add_ptr;
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::{SsaValues, TritonCodegen};
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
        mlir_block: &BlockRef<'a, 'a>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 2, "TritonCodegen::codegen_mul_call: args length must be 2");

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        println!("[DEBUG] TritonCodegen::codegen_mul_call: arg0: {:?}", arg0);
        println!("[DEBUG] TritonCodegen::codegen_mul_call: arg1: {:?}", arg1);

        let arg0_value =
            self.codegen_operand(tcx, instance, arg0, arg0.ty(mir, tcx), mlir_block, ssa_values)?;
        let arg1_value =
            self.codegen_operand(tcx, instance, arg1, arg1.ty(mir, tcx), mlir_block, ssa_values)?;

        self.codegen_mul(arg0_value, arg1_value, mlir_block)
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
        mlir_block: &BlockRef<'a, 'a>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 2, "TritonCodegen::codegen_add_call: args length must be 2");

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        println!("[DEBUG] TritonCodegen::codegen_add_call: arg0: {:?}", arg0);
        println!("[DEBUG] TritonCodegen::codegen_add_call: arg1: {:?}", arg1);

        let lhs =
            self.codegen_operand(tcx, instance, arg0, arg0.ty(mir, tcx), mlir_block, ssa_values)?;
        let rhs =
            self.codegen_operand(tcx, instance, arg1, arg1.ty(mir, tcx), mlir_block, ssa_values)?;

        self.codegen_add(tcx, lhs, rhs, mlir_block)
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
        mlir_block: &BlockRef<'a, 'a>,
        ssa_values: &mut SsaValues<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 2, "TritonCodegen::codegen_lt_call: args length must be 2");

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        let lhs =
            self.codegen_operand(tcx, instance, arg0, arg0.ty(mir, tcx), mlir_block, ssa_values)?;
        let rhs =
            self.codegen_operand(tcx, instance, arg1, arg1.ty(mir, tcx), mlir_block, ssa_values)?;

        self.codegen_cmpi(tcx, Predicate::SLT, lhs, rhs, mlir_block)
    }

    pub fn codegen_cmpi<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        predicate: Predicate,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs_is_tensor = lhs.r#type().is_tensor();
        let rhs_is_tensor = rhs.r#type().is_tensor();
        let location = Location::unknown(self.module.context());

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

        let lt_op: Operation<'a> = create_cmpi(
            self.module.context(),
            Location::unknown(self.module.context()),
            predicate,
            lhs,
            rhs,
            result_ty.into(),
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = lt_op.result(0).expect("LT operation result not found");
        mlir_block.append_operation(lt_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_mul(
        &self,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        if lhs_ty.is_integer() {
            let mul_op: Operation = create_muli(
                self.module.context(),
                Location::unknown(self.module.context()),
                lhs,
                rhs,
            )
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
            let result = mul_op.result(0).expect("Mul operation result not found");
            mlir_block.append_operation(mul_op);
            Ok(Some(result.into()))
        } else {
            todo!("TritonCodegen::codegen_mul: {:?} {:?}", lhs_ty, rhs_ty);
        }
    }

    pub fn codegen_add<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
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
            (true, false) => (
                lhs,
                self.like_tensor(
                    tcx,
                    Location::unknown(self.module.context()),
                    lhs,
                    rhs,
                    mlir_block,
                )?,
            ),
            (false, true) => (
                self.like_tensor(
                    tcx,
                    Location::unknown(self.module.context()),
                    rhs,
                    lhs,
                    mlir_block,
                )?,
                rhs,
            ),
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

        let add_op: Operation<'a> = if lhs_ty.element().is_integer()
            && rhs_ty.element().is_integer()
        {
            create_addi(self.module.context(), Location::unknown(self.module.context()), lhs, rhs)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into()
        } else {
            create_addf(self.module.context(), Location::unknown(self.module.context()), lhs, rhs)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into()
        };

        let result = add_op.result(0).expect("Add operation result not found");

        mlir_block.append_operation(add_op);
        Ok(Some(result.into()))
    }
}
