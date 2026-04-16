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

use std::collections::HashMap;

use melior::ir::operation::OperationLike;
use melior::ir::attribute::IntegerAttribute;
use melior::ir::r#type::IntegerType;
use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Operation, RegionLike, TypeLike, Value,
    ValueLike,
};
use melior::utility::register_all_llvm_translations;
use rustc_abi::FieldIdx;
use rustc_ast::{IntTy, MutTy, UintTy};
use rustc_index::IndexVec;
use rustc_abi::FieldsShape;
use rustc_middle::mir::interpret::{GlobalAlloc, Scalar, alloc_range};
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BasicBlockData, BinOp, Body, CastKind, Const, ConstOperand,
    ConstValue, Local, NonDivergingIntrinsic, Operand, Place, ProjectionElem, Rvalue, Statement,
    StatementKind,
};
use rustc_middle::ty::layout::MaybeResult;
use rustc_middle::ty::{
    self, AdtDef, ConstKind, EarlyBinder, GenericArg, Instance, ParamConst, ScalarInt, Ty, TyCtxt,
    TyKind, TypingEnv,
};
use rustc_mlir::load_all_dialects;
use rustc_mlir::shared::arith::{Int, create_constant, create_int_constant};
use rustc_mlir::shared::attr::create_scalar_attr;
use rustc_mlir::shared::builtin::{tensor_type, tensor_type_like};
use rustc_mlir::shared::ub::create_ub_poison;
use rustc_mlir::triton::tensor::splat;
use rustc_mlir::triton::{create_func, int_to_ptr, load_triton_dialect, pointer_type};

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::codegen::triton::location::span_to_location;
use crate::mlir::codegen::triton::types::TypeMapper;
use crate::mlir::errors::MlirError;

mod location;
mod ops;
mod types;

use types::is_option_ty;

type SsaValues<'c, 'p> = HashMap<Local, Value<'c, 'p>>;

/// Tracks `Option<T>` MIR locals that must not be materialised as MLIR SSA values.
/// `None` entry  → the local holds the `None` variant (no inner value).
/// `Some(v)` entry → the local holds `Some(v)` with this MLIR value as the inner.
/// These locals are NEVER inserted into `SsaValues`.
type OptionTable<'c, 'p> = HashMap<Local, Option<Value<'c, 'p>>>;

/// Tracks tuple MIR locals (including the return place `_0` for tuple-returning functions).
/// The vec contains the individual MLIR values for each tuple field in order.
type TupleTable<'c, 'p> = HashMap<Local, Vec<Value<'c, 'p>>>;

/// Constant integer arrays (e.g. shape arrays like `[BLOCK_SIZE]`).
/// Keyed by the MIR `Local` that holds the array; value is the elements as `i64`.
type ConstArrays = HashMap<Local, Vec<i64>>;

/// Maps a raw-pointer local (from `&raw const arr`) to the `ConstArrays` key it was derived from.
type PtrToConstArray = HashMap<Local, Local>;

/// Maps a MIR local that holds `&[i32]` (built from a const array via `slice_from_raw_parts` /
/// `transmute`) to the shape extracted from the underlying `ConstArrays` entry.
type SliceShape = HashMap<Local, Vec<i64>>;

/// Codegen state threaded through all statement/terminator handlers.
pub(crate) struct CodegenState<'c, 'p> {
    pub(crate) ssa_values: SsaValues<'c, 'p>,
    pub(crate) option_table: OptionTable<'c, 'p>,
    pub(crate) tuple_fields: TupleTable<'c, 'p>,
    /// Constant integer arrays derived from aggregate literals.
    pub(crate) const_arrays: ConstArrays,
    /// `local → array_local`: a raw pointer derived from a const-array alloca.
    pub(crate) ptr_to_const_array: PtrToConstArray,
    /// `local → shape`: a fat-pointer slice `&[i32]` whose shape is statically known.
    pub(crate) slice_shape: SliceShape,
}

impl<'c, 'p> CodegenState<'c, 'p> {
    fn new() -> Self {
        Self {
            ssa_values: HashMap::new(),
            option_table: HashMap::new(),
            tuple_fields: HashMap::new(),
            const_arrays: HashMap::new(),
            ptr_to_const_array: HashMap::new(),
            slice_shape: HashMap::new(),
        }
    }
}

pub(crate) struct TritonCodegen<'a> {
    module: &'a MlirModule<'static>,
    type_mapper: TypeMapper,
}

impl<'a> TritonCodegen<'a> {
    pub fn new(module: &'a MlirModule<'static>) -> Self {
        let context = module.context();

        load_all_dialects(context);
        register_all_llvm_translations(context);
        load_triton_dialect(context);

        Self { module, type_mapper: TypeMapper::new() }
    }

    fn to_scalar_int<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        node: &Operand<'tcx>,
    ) -> Result<ScalarInt, MlirError> {
        match node {
            Operand::Constant(c) => {
                // We expect the constant to have a value that tells us the discriminant/variant
                match c.const_ {
                    Const::Val(ConstValue::Scalar(Scalar::Int(scalar_int)), _) => Ok(scalar_int),
                    Const::Ty(ty, const_val) => {
                        // Handle const generic parameters of integer type
                        if !ty.is_integral() {
                            return Err(MlirError::InvalidScalar { node: format!("{:?}", node) });
                        }
                        match const_val.kind() {
                            ConstKind::Param(param) => {
                                let value = instance.args.const_at(param.index as usize).to_value();
                                let scalar_int = value.try_to_leaf().ok_or_else(|| {
                                    MlirError::InvalidScalar { node: format!("{:?}", node) }
                                })?;
                                Ok(scalar_int)
                            }
                            _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
                        }
                    }
                    _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
                }
            }
            _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
        }
    }

    /// The value is assumed to be a scalar of the same type as the tensor.
    /// The result is a tensor of the same shape as the provided tensor, with the scalar values repeated.
    fn like_tensor<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        location: Location<'a>,
        tensor: Value<'a, 'a>,
        value: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let tensor_type = tensor_type_like(
            tensor
                .r#type()
                .try_into()
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?,
            value.r#type(),
        )
        .map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        let splat_op: Operation<'_> =
            splat(&self.module.context(), location, value, tensor_type.into())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = splat_op.result(0).unwrap();

        mlir_block.append_operation(splat_op);
        Ok(result.into())
    }

    fn codegen_function<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        fn_ty: Ty<'tcx>,
        instance: &Instance<'tcx>,
    ) -> Result<(), MlirError> {
        let mut state = CodegenState::new();
        let mut basic_blocks: HashMap<BasicBlock, BlockRef> = HashMap::new();

        // Downcast to a FnSig
        let fn_sig = fn_ty.fn_sig(tcx);
        let fn_sig = fn_sig.skip_binder(); // Remove late-bound lifetimes

        // Extract a friendly function name, preferring unmangled if possible
        let func_name = tcx.symbol_name(*instance).name;

        // Try to demangle using the Rust symbol demangling crate if available.
        // Since in rustc we don't always bring in the rustc-demangle crate, but
        // the symbol_name should generally be readable for non-generic items.
        // Otherwise, fallback to `def_path_str` (should give a crate-relative path).
        let friendly_name = if func_name.starts_with("_R") {
            // Looks like a Rust-mangled symbol. Try to show a better name.
            tcx.def_path_str(instance.def_id())
        } else {
            func_name.to_string()
        };

        eprintln!(
            "[DEBUG] TritonCodegen codegen_function: function name: {} (raw symbol: {})",
            friendly_name, func_name
        );

        // Skip Triton intrinsic function bodies — calls to these are intercepted at call-sites
        // by the codegen dispatch table, so the actual body is never compiled for GPU execution.
        // Their signatures may contain types (like `&[T]`) that have no valid static MLIR form.
        if fn_sig.inputs().iter().any(|ty| {
            matches!(ty.kind(), TyKind::Ref(_, inner, _) if matches!(inner.kind(), TyKind::Slice(_)))
        }) {
            eprintln!(
                "[DEBUG] TritonCodegen codegen_function: skipping intrinsic stub (has &[T] param): {}",
                friendly_name
            );
            return Ok(());
        }

        // Arguments — Option<T> params are excluded from the MLIR signature; they
        // are tracked in the option_table as None (absent) by default.
        let inputs: Vec<Ty<'tcx>> = fn_sig.inputs().iter().copied().collect();
        let arg_types: Vec<_> = inputs
            .iter()
            .filter(|ty| !is_option_ty(tcx, **ty))
            .map(|ty| self.type_mapper.map_type(self.module.context(), &tcx, ty))
            .collect();

        // Result type — flatten top-level Rust tuples into multiple MLIR return types.
        let ret_type = fn_sig.output();
        let ret_types: Vec<_> = if ret_type.is_unit() {
            vec![]
        } else if let TyKind::Tuple(elem_tys) = ret_type.kind() {
            elem_tys
                .iter()
                .map(|ty| self.type_mapper.map_type(self.module.context(), &tcx, &ty))
                .collect()
        } else {
            match self.type_mapper.map_type(self.module.context(), &tcx, &ret_type).to_result() {
                Ok(t) => vec![t],
                Err(_) => vec![],
            }
        };

        // Skip functions whose MLIR signature contains dynamic tensor types (tensor<?x...>).
        // These are Triton intrinsic stubs whose call-sites are intercepted by the dispatch table;
        // their generic tensor parameters have no statically-known shape and cannot be verified.
        let has_dynamic_tensor = arg_types.iter().chain(ret_types.iter()).any(|ty| ty.is_tensor());
        if has_dynamic_tensor {
            eprintln!(
                "[DEBUG] TritonCodegen codegen_function: skipping stub (has tensor<> param): {}",
                friendly_name
            );
            return Ok(());
        }

        // DEBUG output: print argument and result types
        eprintln!("[DEBUG] TritonCodegen: instance function signature (argument types):");
        for (i, arg_ty) in arg_types.iter().enumerate() {
            eprintln!("    arg[{}]: {}", i, arg_ty);
        }
        eprintln!(
            "[DEBUG] TritonCodegen: instance function signature (return type): {:?}",
            ret_types
        );

        // Iterate over MIR basic blocks and codegen each one
        let visibility = if func_name.ends_with("entry_point") { "public" } else { "private" };

        let func_loc =
            span_to_location(self.module.context(), tcx, tcx.def_span(instance.def_id()));
        let func_op: Operation = create_func(
            self.module.context(),
            func_loc,
            func_name,
            visibility,
            &arg_types,
            &ret_types,
            16,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        let mir = tcx.instance_mir(instance.def);
        let location = func_loc;

        if func_name.contains("program_id") {
            println!(
                "[DEBUG] TritonCodegen::codegen_function: func_name: {:?}, arg_types: {:?}",
                func_name, arg_types
            );
            println!("[DEBUG] TritonCodegen::codegen_function: mir: {:?}", mir);
        }

        for (bb, _) in mir.basic_blocks.iter_enumerated() {
            let block = Block::new(&[]);
            if bb.index() == 0 {
                // Add non-Option function arguments as block arguments to the entry block.
                // Option<T> params are pre-populated into option_table as None.
                let mut mlir_arg_idx = 0;
                for (param_idx, input_ty) in inputs.iter().enumerate() {
                    let local = Local::from_usize(param_idx + 1);
                    if is_option_ty(tcx, *input_ty) {
                        state.option_table.insert(local, None);
                    } else {
                        let value = block.add_argument(arg_types[mlir_arg_idx], location);
                        state.ssa_values.insert(local, value);
                        mlir_arg_idx += 1;
                    }
                }
            }

            let block_ref =
                func_op.region(0).expect("tt.func must have a body region").append_block(block);
            basic_blocks.insert(bb, block_ref);
        }

        for (bb, bb_data) in mir.basic_blocks.iter_enumerated() {
            self.codegen_basic_block(
                tcx,
                instance,
                &mir,
                bb,
                bb_data,
                &func_op,
                &mut state,
                &basic_blocks,
            )?;
        }

        println!("[DEBUG] TritonCodegen::codegen_function end: ssa_values: {:?}", state.ssa_values);
        self.module.mlir.body().append_operation(func_op.into());

        Ok(())
    }

    fn codegen_basic_block<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        bb: BasicBlock,
        bb_data: &BasicBlockData<'tcx>,
        _func_op: &Operation,
        state: &mut CodegenState<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
    ) -> Result<(), MlirError> {
        let mlir_block = basic_blocks.get(&bb).expect("block not found");

        // Codegen each MIR statement in order.
        for stmt in &bb_data.statements {
            self.codegen_statement(tcx, instance, mir, stmt, mlir_block, state)?;
        }

        // Codegen the block terminator.
        self.codegen_terminator(
            tcx,
            instance,
            mir,
            bb_data.terminator(),
            mlir_block,
            state,
            basic_blocks,
        )?;

        Ok(())
    }

    fn codegen_statement<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        stmt: &Statement<'tcx>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_statement: ssa_values: {:?}", state.ssa_values);
        let location = span_to_location(self.module.context(), tcx, stmt.source_info.span);
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let (place, rvalue) = assign.as_ref();
                println!(
                    "[DEBUG] TritonCodegen::codegen_statement: Assign: {:?}, {:?} {:?}",
                    stmt, place, rvalue
                );
                self.codegen_assign(
                    tcx, instance, mir, place, rvalue, location, mlir_block, state,
                )
            }
            StatementKind::SetDiscriminant { place, variant_index } => {
                self.codegen_set_discriminant(tcx, instance, mir, place, *variant_index, mlir_block, state)
            }
            StatementKind::StorageLive(local) => self.codegen_storage_live(tcx, *local, mlir_block),
            StatementKind::StorageDead(local) => self.codegen_storage_dead(tcx, *local, mlir_block),
            StatementKind::Intrinsic(intrinsic) => {
                self.codegen_intrinsic(tcx, intrinsic, mlir_block)
            }
            // Runtime no-ops or analysis-only statements that require no codegen.
            StatementKind::Nop
            | StatementKind::ConstEvalCounter
            | StatementKind::FakeRead(_)
            | StatementKind::PlaceMention(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(_)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Retag(..) => Ok(()),
        }?;

        println!("[DEBUG] TritonCodegen::codegen_statement: ssa_values: {:?}", state.ssa_values);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_assign<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        match rvalue {
            Rvalue::Use(operand) => {
                let ty = operand.ty(mir, tcx);
                let typing_env = TypingEnv::fully_monomorphized();
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(ty),
                );

                // Route Option<T> locals into the option_table instead of ssa_values.
                if is_option_ty(tcx, normalized_ty) {
                    let opt_value = match operand {
                        Operand::Copy(src) | Operand::Move(src) => {
                            *state.option_table.get(&src.local).unwrap_or_else(|| {
                                panic!("Option local {:?} not found in option_table", src.local)
                            })
                        }
                        // `None::<T>` as a constant — no inner value.
                        Operand::Constant(_) => None,
                        Operand::RuntimeChecks(_) => todo!("RuntimeChecks operand not yet supported"),
                    };
                    state.option_table.insert(place.local, opt_value);
                    return Ok(());
                }

                // Route Tuple<T...> locals into tuple_fields.
                if let TyKind::Tuple(elem_tys) = normalized_ty.kind() {
                    let fields = match operand {
                        Operand::Copy(src) | Operand::Move(src) => {
                            state.tuple_fields.get(&src.local).cloned().unwrap_or_else(|| {
                                panic!("Tuple local {:?} not found in tuple_fields", src.local)
                            })
                        }
                        Operand::Constant(const_op) => self.codegen_tuple_constant(
                            tcx, instance, const_op, elem_tys, location, mlir_block,
                        )?,
                        Operand::RuntimeChecks(_) => todo!("RuntimeChecks for tuple"),
                    };
                    state.tuple_fields.insert(place.local, fields);
                    return Ok(());
                }

                let result = self.codegen_operand(
                    tcx,
                    instance,
                    operand,
                    normalized_ty,
                    location,
                    mlir_block,
                    state,
                )?;
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 1: result: Place: {:?}, Result: {:?}",
                    place, result
                );
                state.ssa_values.insert(place.local, result);
            }
            Rvalue::Cast(cast_kind, operand, ty) => {
                println!("Cast cast_kind: {:?}, operand: {:?}, ty: {:?}", cast_kind, operand, ty);

                // PtrToPtr cast on a const-array pointer — propagate the side-table entry
                // without emitting an MLIR op (const arrays have no MLIR value).
                if matches!(cast_kind, CastKind::PtrToPtr) {
                    if let Operand::Copy(src_place) | Operand::Move(src_place) = operand {
                        if let Some(&arr_local) = state.ptr_to_const_array.get(&src_place.local) {
                            state.ptr_to_const_array.insert(place.local, arr_local);
                            return Ok(());
                        }
                    }
                }

                // Transmute of a slice-shape tuple ((*const T, usize) → &[T]) — propagate shape.
                if matches!(cast_kind, CastKind::Transmute) {
                    if let Operand::Copy(src_place) | Operand::Move(src_place) = operand {
                        if let Some(shape) = state.slice_shape.get(&src_place.local).cloned() {
                            state.slice_shape.insert(place.local, shape);
                            return Ok(());
                        }
                    }
                }

                let result = self.codegen_cast(
                    tcx, instance, cast_kind, operand, ty, location, mlir_block, state,
                )?;

                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 2: result: Place: {:?}, Result: {:?}",
                    place, result
                );

                state.ssa_values.insert(place.local, result);
            }
            Rvalue::Aggregate(aggregate_kind, index_vec) => {
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign: Aggregate: {:?}, index_vec: {:?}",
                    aggregate_kind, index_vec
                );
                println!("[DEBUG] TritonCodegen::codegen_assign: ssa_values: {:?}", state.ssa_values);

                // Route Option<T> aggregates (Some/None) into the option_table.
                if let AggregateKind::Adt(def_id, variant_index, _, _, _) = aggregate_kind.as_ref() {
                    let adt_def = tcx.adt_def(*def_id);
                    let norm_place_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                        tcx,
                        TypingEnv::fully_monomorphized(),
                        EarlyBinder::bind(place.ty(mir, tcx).ty),
                    );
                    if is_option_ty(tcx, norm_place_ty) {
                        return self.codegen_option_aggregate(
                            tcx, instance, mir, place, adt_def, *variant_index, index_vec,
                            location, mlir_block, state,
                        );
                    }
                }

                // Route Tuple aggregates into tuple_fields (MLIR uses multiple return values).
                if let AggregateKind::Tuple = aggregate_kind.as_ref() {
                    // Special case: `(*const T, usize)` fat-pointer tuple where the pointer
                    // comes from a const array.  Route to `slice_shape` instead — no MLIR value.
                    if let Some(first_op) = index_vec.iter().next() {
                        if let Operand::Copy(p) | Operand::Move(p) = first_op {
                            if let Some(&arr_local) = state.ptr_to_const_array.get(&p.local) {
                                if let Some(shape) = state.const_arrays.get(&arr_local) {
                                    state.slice_shape.insert(place.local, shape.clone());
                                    return Ok(());
                                }
                            }
                        }
                    }

                    let fields = index_vec
                        .iter()
                        .map(|op| match op {
                            Operand::Copy(p) | Operand::Move(p) => self.codegen_copy(&p, state),
                            _ => todo!("Tuple aggregate with non-copy/move operand: {:?}", op),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    state.tuple_fields.insert(place.local, fields);
                    return Ok(());
                }

                // Constant integer arrays (e.g. shape arrays `[BLOCK_SIZE]` for zeros/reshape).
                // We only track them as constant metadata; no MLIR value is emitted.
                if let AggregateKind::Array(_elem_ty) = aggregate_kind.as_ref() {
                    let elems: Option<Vec<i64>> = index_vec
                        .iter()
                        .map(|op| {
                            self.to_scalar_int(tcx, instance, op).ok().map(|s| {
                                match s.size().bytes() {
                                    1 => s.to_u8() as i64,
                                    2 => s.to_i16() as i64,
                                    4 => s.to_i32() as i64,
                                    8 => s.to_i64(),
                                    n => todo!("ScalarInt size {} bytes", n),
                                }
                            })
                        })
                        .collect();
                    if let Some(elems) = elems {
                        state.const_arrays.insert(place.local, elems);
                        return Ok(());
                    }
                    // Not all-constant: fall through to todo!() below.
                    todo!("AggregateKind::Array (non-constant elements): {:?}", index_vec);
                }

                let result = self.codegen_aggregate_create(
                    tcx,
                    instance,
                    mir,
                    aggregate_kind,
                    index_vec,
                    location,
                    mlir_block,
                    state,
                )?;
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 3: result: Place: {:?}, Result: {:?}",
                    place, result
                );

                if let Some(result) = result {
                    println!(
                        "codegen_aggregate_create: result: ** {:?} ** {:?}",
                        place.local, result
                    );
                    state.ssa_values.insert(place.local, result);
                } else {
                    println!(
                        "[DEBUG] TritonCodegen::codegen_assign: result is None: {:?} {:?}",
                        place.local, rvalue
                    );
                }
            }
            Rvalue::Repeat(operand, _) => todo!("Repeat: {:?}", operand),
            Rvalue::Ref(region, borrow_kind, place) => {
                todo!("Ref: {:?} {:?} {:?}", region, borrow_kind, place)
            }
            Rvalue::ThreadLocalRef(def_id) => todo!("ThreadLocalRef: {:?}", def_id),
            Rvalue::RawPtr(_raw_ptr_kind, src_place) => {
                // If this is `&raw const const_array_local`, record the mapping so that
                // downstream `zeros` / `slice_from_raw_parts` calls can recover the shape.
                if src_place.projection.is_empty()
                    && state.const_arrays.contains_key(&src_place.local)
                {
                    state.ptr_to_const_array.insert(place.local, src_place.local);
                    return Ok(());
                }
                todo!("RawPtr: {:?} {:?}", _raw_ptr_kind, src_place)
            }
            Rvalue::BinaryOp(bin_op, operands) => {
                let value = self.codegen_binary_op(
                    tcx, instance, mir, place, bin_op, operands, location, mlir_block, state,
                )?;
                if let Some(value) = value {
                    state.ssa_values.insert(place.local, value);
                }
            }
            Rvalue::UnaryOp(un_op, operand) => todo!("UnaryOp: {:?} {:?}", un_op, operand),
            Rvalue::Discriminant(src_place) => {
                let src_ty = src_place.ty(mir, tcx).ty;
                let norm_src_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(src_ty),
                );
                if is_option_ty(tcx, norm_src_ty) {
                    // The discriminant of an Option is statically known from the option_table.
                    let discr: i64 = match state.option_table.get(&src_place.local) {
                        Some(None)    => 0, // None variant
                        Some(Some(_)) => 1, // Some variant
                        None => panic!(
                            "Option local {:?} not found in option_table for Discriminant",
                            src_place.local
                        ),
                    };
                    let int_val = Int::I8(discr as u8);
                    let const_op: Operation<'a> = create_int_constant(
                        self.module.context(), location, int_val,
                    ).map_err(|e| MlirError::CreateOperation { err: e })?.into();
                    let result = const_op.result(0)
                        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                    mlir_block.append_operation(const_op);
                    state.ssa_values.insert(place.local, result.into());
                    return Ok(());
                }
                todo!("Discriminant for non-Option: {:?}", src_place)
            }
            Rvalue::ShallowInitBox(operand, ty) => todo!("ShallowInitBox: {:?} {:?}", operand, ty),
            Rvalue::CopyForDeref(place) => todo!("CopyForDeref: {:?}", place),
            Rvalue::WrapUnsafeBinder(operand, ty) => {
                todo!("WrapUnsafeBinder: {:?} {:?}", operand, ty)
            }
        }

        // todo!("[TODO] TritonCodegen::codegen_assign: {:?} {:?}", place, rvalue)
        Ok(())
    }

    fn codegen_aggregate_create<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        aggregate_kind: &AggregateKind<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        println!("codegen_aggregate_assign: {:?} {:?} {:?}", aggregate_kind, index_vec, mlir_block);

        match aggregate_kind {
            AggregateKind::Adt(def_id, _, raw_list, _, _) => {
                // Get the ADT definition and a human-readable name for debugging.
                let adt_def = tcx.adt_def(*def_id);
                let adt_name = format!("{:?}", adt_def);

                if "triton::llvm::triton::tensor::Tensor" == adt_name {
                    self.codegen_create_tensor(
                        tcx, instance, mir, index_vec, location, mlir_block, state,
                    )
                } else if "triton::llvm::triton::pointer::Pointer" == adt_name {
                    self.codegen_create_pointer(
                        tcx, instance, mir, index_vec, location, mlir_block, state,
                    )
                } else {
                    todo!(
                        "codegen_aggregate_create: {:?} {:?} {:?}",
                        adt_name,
                        adt_def,
                        index_vec.as_slice()
                    );
                }
            }
            _ => todo!("AggregateKind: {:?}", aggregate_kind),
        }
    }

    fn codegen_create_tensor<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let arg1 = index_vec.get(FieldIdx::from_usize(0)).expect("arg1 not found");
        let arg1_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(arg1.ty(mir, tcx)),
        );

        let pointee_ty = match arg1_ty.kind() {
            TyKind::RawPtr(pointee_ty, _) => {
                self.type_mapper.map_type(self.module.context(), &tcx, &pointee_ty)
            }
            _ => todo!("codegen_create_tensor: arg1_ty: {:?}", arg1_ty),
        };
        let tensor_type = tensor_type(&[i64::MIN], pointee_ty).into();
        let tensor_op = create_ub_poison(self.module.context(), location, tensor_type)
            .map_err(|e| MlirError::CreateOperation { err: e })?;

        let tensor_result = tensor_op.result(0).unwrap();
        mlir_block.append_operation(tensor_op);
        Ok(Some(tensor_result.into()))
    }

    fn codegen_create_pointer<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let arg1 = index_vec.get(FieldIdx::from_usize(0)).expect("arg1 not found");
        let arg1_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(arg1.ty(mir, tcx)),
        );
        // `Pointer<T>` is a newtype wrapper around a raw pointer in Triton DSL.
        // Preserve the wrapped pointer SSA value rather than materializing poison.
        let pointer_value =
            self.codegen_operand(tcx, instance, arg1, arg1_ty, location, mlir_block, state)?;
        Ok(Some(pointer_value))
    }

    fn codegen_const_adt<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        adt_def: &AdtDef<'tcx>,
        value: Value<'a, 'a>,
        raw_list: &[GenericArg<'tcx>],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let name = format!("{:?}", adt_def);
        let map_ty = |idx: usize| {
            let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(raw_list[idx].expect_ty()),
            );
            self.type_mapper.map_type(self.module.context(), &tcx, &ty)
        };

        // If the name of the ADT is tensor, then we create a poison operation.
        // This is because the tensor creation is part of the dsl dead code which
        // will be eliminated by the optimizer.
        if name == "triton::llvm::triton::tensor::Tensor" {
            let ty = map_ty(0);
            let tensor_type = tensor_type(&[i64::MIN], ty).into();
            let tensor_op = create_ub_poison(self.module.context(), location, tensor_type)
                .map_err(|e| MlirError::CreateOperation { err: e })?;

            let tensor_result = tensor_op.result(0).unwrap();
            mlir_block.append_operation(tensor_op);
            Ok(tensor_result.into())
        } else if name == "triton::llvm::triton::pointer::Pointer" {
            // `Pointer<T>` is a newtype wrapper — pass through the wrapped value directly.
            Ok(value)
        } else if adt_def.is_enum() {
            // For enum ADTs (e.g. `Axis`, `PaddingOption`, `CacheModifier`, etc.),
            // `codegen_scalar_const_value` already emitted the discriminant as an integer.
            // Pass through the value unchanged.
            Ok(value)
        } else {
            todo!("Adt: {:?}", adt_def)
        }
    }

    fn codegen_binary_op<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        bin_op: &BinOp,
        operands: &(Operand<'tcx>, Operand<'tcx>),
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let (lhs_op, rhs_op) = operands;

        let lhs_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(lhs_op.ty(mir, tcx)),
        );
        let rhs_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(rhs_op.ty(mir, tcx)),
        );
        let lhs =
            self.codegen_operand(tcx, instance, lhs_op, lhs_ty, location, mlir_block, state)?;
        let rhs =
            self.codegen_operand(tcx, instance, rhs_op, rhs_ty, location, mlir_block, state)?;

        match bin_op {
            BinOp::Add => todo!(),
            BinOp::AddUnchecked => todo!(),
            BinOp::AddWithOverflow => todo!(),
            BinOp::Sub => todo!(),
            BinOp::SubUnchecked => todo!(),
            BinOp::SubWithOverflow => todo!(),
            BinOp::Mul => self.codegen_mul(tcx, location, lhs, rhs, mlir_block),
            BinOp::MulUnchecked => todo!(),
            BinOp::MulWithOverflow => todo!(),
            BinOp::Div => todo!(),
            BinOp::Rem => todo!(),
            BinOp::BitXor => todo!(),
            BinOp::BitAnd => todo!(),
            BinOp::BitOr => todo!(),
            BinOp::Shl => todo!(),
            BinOp::ShlUnchecked => todo!(),
            BinOp::Shr => todo!(),
            BinOp::ShrUnchecked => todo!(),
            BinOp::Eq => todo!(),
            BinOp::Lt => todo!(),
            BinOp::Le => todo!(),
            BinOp::Ne => todo!(),
            BinOp::Ge => todo!(),
            BinOp::Gt => todo!(),
            BinOp::Cmp => todo!(),
            BinOp::Offset => todo!(),
        }
    }

    /// Codegen construction of an `Option<T>` aggregate (either `None` or `Some(inner)`).
    /// The result is stored in `state.option_table` rather than `state.ssa_values`.
    #[allow(clippy::too_many_arguments)]
    fn codegen_option_aggregate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        _adt_def: rustc_middle::ty::AdtDef<'tcx>,
        variant_index: rustc_abi::VariantIdx,
        index_vec: &rustc_index::IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        if variant_index.as_usize() == 0 {
            // None variant — no inner value.
            state.option_table.insert(place.local, None);
        } else {
            // Some(inner) — codegen the single field and stash it.
            let inner_op = index_vec.get(FieldIdx::from_usize(0))
                .expect("Option::Some aggregate must have exactly one field");
            let inner_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(inner_op.ty(mir, tcx)),
            );
            let inner_value = self.codegen_operand(
                tcx, instance, inner_op, inner_ty, location, mlir_block, state,
            )?;
            state.option_table.insert(place.local, Some(inner_value));
        }
        Ok(())
    }

    fn codegen_set_discriminant<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        variant_index: rustc_abi::VariantIdx,
        _mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        let place_ty = place.ty(mir, tcx).ty;
        let norm_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(place_ty),
        );

        if is_option_ty(tcx, norm_ty) {
            if variant_index.as_usize() == 0 {
                // Two-phase None: discriminant set to 0.
                state.option_table.insert(place.local, None);
            } else {
                // Two-phase Some: inner value was written into ssa_values[place.local] by
                // a preceding field-assignment via Place::Downcast projection.  Promote it.
                let inner = state.ssa_values.remove(&place.local).unwrap_or_else(|| {
                    panic!(
                        "SetDiscriminant Some: expected inner value in ssa_values for {:?}",
                        place.local
                    )
                });
                state.option_table.insert(place.local, Some(inner));
            }
            return Ok(());
        }

        todo!("[TODO] TritonCodegen::codegen_set_discriminant for non-Option types")
    }

    fn codegen_storage_live<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_storage_live: local: {:?}", _local);
        // NO-OP: In the context of Triton and MLIR, storage live is a no-op.
        Ok(())
    }

    fn codegen_storage_dead<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_storage_dead: local: {:?}", _local);
        // NO-OP: In the context of Triton and MLIR, storage dead is a no-op.
        Ok(())
    }

    fn codegen_intrinsic<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _intrinsic: &NonDivergingIntrinsic<'tcx>,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        todo!("[TODO] TritonCodegen::codegen_intrinsic")
    }

    fn codegen_cast<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        cast_kind: &CastKind,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match cast_kind {
            CastKind::PointerWithExposedProvenance => {
                self.codegen_pointer_with_exposed_provenance(
                    tcx, instance, operand, ty, location, mlir_block, state,
                )
            }
            CastKind::PtrToPtr => {
                self.codegen_ptr_to_ptr(tcx, instance, operand, ty, location, mlir_block, state)
            }
            CastKind::IntToInt => {
                self.codegen_int_to_int(tcx, instance, operand, ty, location, mlir_block, state)
            }
            _ => {
                // Unhandled cast kinds (Transmute, PointerCoercion, ReifyFnPointer, etc.).
                // Emit ub.poison of the destination type as a safe placeholder.
                let typing_env = TypingEnv::fully_monomorphized();
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(*ty),
                );
                println!(
                    "[DEBUG] codegen_cast fallback: cast_kind={:?} ty={:?}",
                    cast_kind, normalized_ty
                );
                let mlir_ty =
                    self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);
                let ub_op: Operation<'a> = create_ub_poison(self.module.context(), location, mlir_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
                let result = ub_op.result(0).expect("ub.poison result").into();
                mlir_block.append_operation(ub_op);
                Ok(result)
            }
        }
    }

    fn codegen_pointer_with_exposed_provenance<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );

        println!(
            "[DEBUG] TritonCodegen::codegen_pointer_with_exposed_provenance: provenance: {:?} ty: {:?} normalized: {:?}",
            operand, ty, normalized_ty
        );

        self.codegen_operand(tcx, instance, operand, normalized_ty, location, mlir_block, state)
    }

    fn codegen_ptr_to_ptr<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );

        self.codegen_operand(tcx, instance, operand, normalized_ty, location, mlir_block, state)
    }

    fn codegen_int_to_int<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );
        self.codegen_operand(tcx, instance, operand, normalized_ty, location, mlir_block, state)
    }

    fn codegen_operand<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        normalized_ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_operand: ssa_values: {:?} operand: {:?}",
            state.ssa_values, operand
        );

        // For MLIR move is the same as copy
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.codegen_copy(place, state),
            Operand::Constant(const_operand) => self.codegen_constant_cast(
                tcx,
                instance,
                const_operand,
                normalized_ty,
                location,
                mlir_block,
            ),
            Operand::RuntimeChecks(_) => todo!("RuntimeChecks operand not yet supported"),
        }
    }

    /// Resolve an operand that may be `Option<T>`.
    /// Returns the inner MLIR value if the option is `Some` (or if the operand is not an Option),
    /// or `None` if the option is absent.
    pub(crate) fn codegen_option_operand<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        operand: &Operand<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(operand.ty(mir, tcx)),
        );

        if is_option_ty(tcx, ty) {
            let inner = match operand {
                Operand::Copy(p) | Operand::Move(p) => {
                    *state.option_table.get(&p.local).unwrap_or_else(|| {
                        panic!("Option local {:?} not found in option_table", p.local)
                    })
                }
                // `None::<T>` constant — no inner value.
                Operand::Constant(_) => None,
                Operand::RuntimeChecks(_) => todo!("RuntimeChecks operand not yet supported"),
            };
            Ok(inner)
        } else {
            let value = self.codegen_operand(tcx, instance, operand, ty, location, mlir_block, state)?;
            Ok(Some(value))
        }
    }

    /// Decode a constant tuple into individual MLIR scalar values.
    /// Reads each element from the underlying memory allocation using layout offsets.
    fn codegen_tuple_constant<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_op: &ConstOperand<'tcx>,
        elem_tys: &[Ty<'tcx>],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let normalized_const = match &const_op.const_ {
            Const::Val(const_val, _) => *const_val,
            other => todo!("codegen_tuple_constant: unexpected const kind {:?}", other),
        };

        let (alloc_id, base_offset) = match normalized_const {
            ConstValue::Indirect { alloc_id, offset } => (alloc_id, offset),
            other => todo!("codegen_tuple_constant: unexpected ConstValue {:?}", other),
        };

        let GlobalAlloc::Memory(const_alloc) = tcx.global_alloc(alloc_id) else {
            todo!("codegen_tuple_constant: non-memory alloc for {:?}", alloc_id);
        };
        let alloc = const_alloc.inner();

        // Get the layout of the full tuple to obtain per-field offsets.
        let tuple_ty = const_op.const_.ty();
        let tuple_layout = tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(tuple_ty))
            .map_err(|e| MlirError::CodegenFailed { err: format!("layout_of failed: {:?}", e) })?;

        let FieldsShape::Arbitrary { ref offsets, .. } = tuple_layout.fields else {
            todo!("codegen_tuple_constant: unexpected FieldsShape for tuple");
        };

        let mut values = Vec::with_capacity(elem_tys.len());
        for (field_idx, field_ty) in elem_tys.iter().enumerate() {
            let normalized_field_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(*field_ty),
            );
            let field_layout = tcx
                .layout_of(TypingEnv::fully_monomorphized().as_query_input(normalized_field_ty))
                .map_err(|e| MlirError::CodegenFailed {
                    err: format!("layout_of field failed: {:?}", e),
                })?;

            let field_offset = base_offset + offsets[FieldIdx::from_usize(field_idx)];
            let range = alloc_range(field_offset, field_layout.size);
            let scalar = alloc
                .read_scalar(&tcx, range, false)
                .map_err(|e| MlirError::CodegenFailed { err: format!("read_scalar: {:?}", e) })?;

            let value =
                self.codegen_scalar(normalized_field_ty, scalar, location, mlir_block)?;
            values.push(value);
        }

        Ok(values)
    }

    fn codegen_copy<'tcx>(
        &self,
        place: &Place<'tcx>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_copy: Local: {:?}, projection: {:?}, ssa_values: {:?}",
            place.local, place.projection, state.ssa_values
        );

        debug_assert!(
            !state.option_table.contains_key(&place.local),
            "BUG: Option local {:?} used as a direct MLIR value; use codegen_option_operand instead",
            place.local
        );

        // Handle a single Field projection on a tuple local stored in tuple_fields.
        if let [ProjectionElem::Field(field_idx, _)] = place.projection.as_slice() {
            if let Some(fields) = state.tuple_fields.get(&place.local) {
                let idx = field_idx.index();
                return Ok(*fields.get(idx).unwrap_or_else(|| {
                    panic!("Tuple field {} not found for local {:?}", idx, place.local)
                }));
            }
        }

        Ok(state.ssa_values
            .get(&place.local)
            .copied()
            .expect(format!("Value not found for local: {:?}", place.local).as_str()))
    }

    fn codegen_constant_cast<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_operand: &ConstOperand<'tcx>,
        normalized_ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_constant_cast: {:?}, {:?}",
            const_operand, normalized_ty
        );

        match const_operand.const_ {
            Const::Val(const_val, ty) => {
                let value =
                    self.codegen_const_value(tcx, instance, const_val, ty, location, mlir_block)?;

                match normalized_ty.kind() {
                    TyKind::RawPtr(_, _) => {
                        println!("[DEBUG] TritonCodegen::codegen_constant_cast: RawPtr");
                        let value_ty = value.r#type();
                        debug_assert!(
                            value_ty.is_integer(),
                            "Triton supports only integer pointer casts"
                        );
                        let ptr_ty =
                            self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);
                        let cast_op: Operation = int_to_ptr(
                            self.module.context(),
                            location,
                            value.into(),
                            ptr_ty,
                        )
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();

                        let result = cast_op.result(0).unwrap();
                        mlir_block.append_operation(cast_op);
                        Ok(result.into())
                    }
                    TyKind::Adt(adt_def, args) => {
                        println!("[DEBUG] TritonCodegen::codegen_constant_cast: Adt");
                        let result = self.codegen_const_adt(
                            tcx,
                            instance,
                            adt_def,
                            value,
                            args.as_slice(),
                            location,
                            mlir_block,
                        )?;
                        Ok(result)
                    }
                    TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) | TyKind::Bool => {
                        // Constant already has the right primitive type — return as-is.
                        Ok(value)
                    }
                    TyKind::Ref(_, _, _) | TyKind::Str => {
                        // &str, &[T], and other reference types are represented as i64
                        // fat-pointer stand-ins in the type mapper. The value already
                        // carries the right i64 type from codegen_const_value.
                        Ok(value)
                    }
                    TyKind::FnPtr(_, _) | TyKind::FnDef(_, _) => {
                        // Function pointer types are represented as i64 — return as-is.
                        Ok(value)
                    }
                    _ => todo!("Constant cast normalized_ty: {:?}", normalized_ty),
                }
            }
            Const::Ty(ty, const_val) => match const_val.kind() {
                ConstKind::Param(param) => {
                    self.codegen_param_const(tcx, instance, ty, param, location, mlir_block)
                }
                ConstKind::Infer(_infer_const) => todo!("ConstKind::Infer"),
                ConstKind::Bound(_bound_var_index_kind, _) => todo!("ConstKind::Bound"),
                ConstKind::Placeholder(_) => todo!("ConstKind::Placeholder"),
                ConstKind::Unevaluated(_unevaluated_const) => todo!("ConstKind::Unevaluated"),
                ConstKind::Value(_) => todo!("ConstKind::Value"),
                ConstKind::Error(_) => todo!("ConstKind::Error"),
                ConstKind::Expr(_) => todo!("ConstKind::Expr"),
            },
            _ => todo!("Const: {:?}", const_operand.const_),
        }
    }

    fn codegen_param_const<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        _ty: Ty<'tcx>,
        param: ParamConst,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let value = instance.args.const_at(param.index as usize).to_value();
        self.codegen_param_value(tcx, value, location, mlir_block)
    }

    fn codegen_param_value<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        value: ty::Value<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let scalar = value.try_to_scalar();

        if let Some(scalar) = scalar {
            self.codegen_scalar(value.ty, scalar, location, mlir_block)
        } else {
            todo!("codegen_param_value: {:?} scalar: {:?}", value, scalar);
        }
    }

    fn codegen_scalar<'tcx>(
        &self,
        ty: Ty<'tcx>,
        scalar: Scalar,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match scalar {
            Scalar::Int(scalar_int) => {
                let scalar_attr = create_scalar_attr(self.module.context(), ty, scalar_int)
                    .map_err(|e| MlirError::CreateOperation { err: e })?;

                let op = create_constant(
                    self.module.context(),
                    location,
                    scalar_attr.0,
                    scalar_attr.1,
                )
                .map_err(|e| MlirError::CreateOperation { err: e })?;

                let op: Operation = op.into();
                let result = op.result(0).expect("Constant operation result not found");
                mlir_block.append_operation(op);
                Ok(result.into())
            }
            Scalar::Ptr(pointer, _) => todo!("Scalar::Ptr: {:?}", pointer),
        }
    }

    fn codegen_const_value<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_val: ConstValue,
        ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match const_val {
            ConstValue::Scalar(scalar) => {
                self.codegen_scalar_const_value(tcx, instance, ty, scalar, location, mlir_block)
            }
            ConstValue::ZeroSized => {
                // Zero-sized values (fn items, unit, zero-sized structs).
                // Emit i64(0) as a harmless placeholder — these are never read
                // by any real MLIR op in the stub-based codegen.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("zero const").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
            ConstValue::Slice { alloc_id: _, meta: _ } => {
                // String literals and other slice constants (&str, &[T]).
                // The type mapper maps &str and &[T] to i64 (fat-pointer stand-in).
                // Emit i64(0) as a placeholder; the callee (device_print etc.) is a stub.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("slice const placeholder").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
            ConstValue::Indirect { alloc_id: _, offset: _ } => {
                // Indirect constants (references to static allocations).
                // Emit i64(0) as a placeholder; these are not exercised by GPU stubs.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("indirect const placeholder").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
        }
    }

    fn codegen_scalar_const_value<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        ty: Ty<'tcx>,
        scalar: Scalar,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match scalar {
            Scalar::Int(scalar_int) => match ty.kind() {
                TyKind::Uint(_) | TyKind::Int(_) => {
                    let value =
                        Int::from_scalar(ty, scalar_int).map_err(|e| MlirError::InvalidScalar {
                            node: format!("Invalid scalar: {:?} {:?} {:?}", e, ty, scalar_int),
                        })?;

                    let const_op: Operation<'a> =
                        create_int_constant(self.module.context(), location, value)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();

                    let result = const_op.result(0)
                        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                TyKind::Adt(adt_def, args) => {
                    println!(
                        "[DEBUG] TritonCodegen::codegen_scalar_const_value: Adt: {:?} {:?} {:?}",
                        scalar, adt_def, args
                    );

                    let scalar_int = match scalar {
                        Scalar::Int(s) => s,
                        Scalar::Ptr(pointer, _) => todo!("Scalar::Ptr: {:?}", pointer),
                    };

                    // For enum ADTs (e.g. `#[repr(i32)] enum Axis { X=0, Y=1, Z=2 }`),
                    // emit the discriminant as the underlying integer directly.
                    let int_val = if adt_def.is_enum() {
                        match scalar_int.size().bytes() {
                            1 => Int::I8(scalar_int.to_u8()),
                            2 => Int::I16(scalar_int.to_u16()),
                            4 => Int::I32(scalar_int.to_u32()),
                            8 => Int::I64(scalar_int.to_u64()),
                            n => todo!("Enum scalar size {} bytes", n),
                        }
                    } else {
                        // For scalar newtype ADTs (e.g. `struct I32(pub i32)`), get the inner
                        // field's primitive type so Int::from_scalar can determine the
                        // correct MLIR integer kind.
                        let variant = adt_def.non_enum_variant();
                        let inner_ty = tcx
                            .type_of(variant.fields[FieldIdx::from_usize(0)].did)
                            .instantiate(tcx, args);
                        Int::from_scalar(inner_ty, scalar_int).map_err(|e| MlirError::InvalidScalar {
                            node: format!("Invalid scalar: {:?} {:?} {:?}", e, inner_ty, scalar_int),
                        })?
                    };

                    let const_op: Operation<'a> =
                        create_int_constant(self.module.context(), location, int_val)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();
                    let result = const_op.result(0).unwrap();
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                TyKind::Bool => {
                    // bool → i1 constant (0 = false, 1 = true)
                    let val = scalar_int.to_u8();
                    let i1_ty = IntegerType::new(self.module.context(), 1);
                    let attr = IntegerAttribute::new(i1_ty.into(), val as i64);
                    let const_op: Operation<'a> =
                        melior::dialect::arith::constant(
                            self.module.context(),
                            attr.into(),
                            location,
                        )
                        .into();
                    let result = const_op.result(0).unwrap();
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                _ => todo!("Scalar::Int ty: {:?} {:?}", ty.kind(), ty),
            },
            Scalar::Ptr(ptr, size) => todo!("Ptr ptr: {:?}, size: {:?}", ptr, size),
        }
    }
}

impl<'a> Codegen for TritonCodegen<'a> {
    fn codegen<'tcx>(&mut self, tcx: TyCtxt<'tcx>, item: &MonoItem<'tcx>) -> Result<(), MlirError> {
        match item {
            MonoItem::Fn(instance) => {
                let fn_ty = instance.ty(tcx, TypingEnv::fully_monomorphized());
                let is_fn_ty = matches!(
                    fn_ty.kind(),
                    rustc_middle::ty::TyKind::FnDef(..) | rustc_middle::ty::TyKind::FnPtr(_, _)
                );

                if !is_fn_ty {
                    todo!(
                        "[DEBUG] TritonCodegen: instance.ty(tcx) is not a function type: {:?}",
                        fn_ty
                    );
                }

                self.codegen_function(tcx, fn_ty, instance)
            }
            MonoItem::Static(_def_id) => {
                // TODO: Implement Triton codegen for statics
                todo!()
            }
            MonoItem::GlobalAsm(_item_id) => {
                // TODO: Implement Triton codegen for global asm
                todo!()
            }
        }
    }
}
