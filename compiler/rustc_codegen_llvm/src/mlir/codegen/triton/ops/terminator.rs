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

use melior::dialect::scf;
use melior::ir::attribute::IntegerAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::r#type::{IntegerType, TupleType};
use melior::ir::{BlockLike, BlockRef, Location, Operation, TypeLike, Value, ValueLike};
use rustc_middle::mir::{BasicBlock, Body, CallSource, Operand, Place, SwitchTargets, Terminator, UnwindAction};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, EarlyBinder, Instance, TyCtxt, TyKind, TypingEnv};
use rustc_mlir::shared::arith::{Predicate, create_cmpi};
use rustc_mlir::shared::cf::create_cf_br;
use rustc_mlir::triton::call;
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use crate::mlir::codegen::triton::location::span_to_location;
use crate::mlir::codegen::triton::{CodegenState, TritonCodegen};
use crate::mlir::errors::MlirError;

// Used inside codegen_terminator_call where 'a and 'tcx are concrete — no HRTB needed.
type LocalCallHandler<'a, 'tcx> = fn(
    &TritonCodegen<'a>,
    TyCtxt<'tcx>,
    &Instance<'tcx>,
    &Body<'tcx>,
    &Operand<'tcx>,
    &str,
    &[Spanned<Operand<'tcx>>],
    &Place<'tcx>,
    &Option<BasicBlock>,
    &UnwindAction,
    &CallSource,
    &Span,
    Location<'a>,
    &BlockRef<'a, 'a>,
    &mut CodegenState<'a, 'a>,
) -> Result<Option<Value<'a, 'a>>, MlirError>;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_terminator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        terminator: &Terminator<'tcx>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
    ) -> Result<(), MlirError> {
        //println!("[DEBUG] TritonCodegen::codegen_terminator: ssa_values: {:?} terminator: {:?}", state.ssa_values, terminator);

        let location =
            span_to_location(self.module.context(), tcx, terminator.source_info.span);

        match &terminator.kind {
            rustc_middle::mir::TerminatorKind::Return => {
                self.codegen_return(location, terminator, mlir_block, state)
            }
            rustc_middle::mir::TerminatorKind::Goto { target } => {
                self.codegen_goto(location, target, mlir_block, basic_blocks, state)
            }
            rustc_middle::mir::TerminatorKind::SwitchInt { discr, targets } => {
                self.codegen_switch_int(tcx, instance, mir, discr, targets, location, mlir_block, basic_blocks, state)
            }
            rustc_middle::mir::TerminatorKind::UnwindResume => todo!("UnwindResume"),
            rustc_middle::mir::TerminatorKind::UnwindTerminate(unwind_terminate_reason) => {
                todo!("UnwindTerminate: {:?}", unwind_terminate_reason)
            }
            rustc_middle::mir::TerminatorKind::Unreachable => todo!("Unreachable"),
            rustc_middle::mir::TerminatorKind::Drop { target, .. } => {
                // All kernel types are Copy with no destructors; treat as goto target.
                self.codegen_goto(location, target, mlir_block, basic_blocks, state)
            }
            rustc_middle::mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                call_source,
                fn_span,
            } => {
                // Use the call-site span for a more precise location on Call terminators.
                let call_loc = span_to_location(self.module.context(), tcx, *fn_span);
                self.codegen_terminator_call(
                    tcx,
                    instance,
                    mir,
                    func,
                    args,
                    destination,
                    target,
                    unwind,
                    call_source,
                    fn_span,
                    call_loc,
                    mlir_block,
                    basic_blocks,
                    state,
                )
            }
            rustc_middle::mir::TerminatorKind::TailCall { func, args, fn_span } => {
                todo!("TailCall: {:?} {:?} {:?}", func, args, fn_span)
            }
            rustc_middle::mir::TerminatorKind::Assert { target, .. } => {
                // GPU kernels have no panic infrastructure — treat Assert as an unconditional
                // branch to the success target (the assertion is assumed to hold).
                self.codegen_goto(location, target, mlir_block, basic_blocks, state)
            }
            rustc_middle::mir::TerminatorKind::Yield { .. } => todo!("Yield"),
            rustc_middle::mir::TerminatorKind::CoroutineDrop => todo!("CoroutineDrop"),
            rustc_middle::mir::TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                todo!("FalseEdge: {:?} {:?}", real_target, imaginary_target)
            }
            rustc_middle::mir::TerminatorKind::FalseUnwind { real_target, unwind } => {
                todo!("FalseUnwind: {:?} {:?}", real_target, unwind)
            }
            rustc_middle::mir::TerminatorKind::InlineAsm {
                asm_macro,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind,
            } => todo!(
                "InlineAsm: {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                asm_macro,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind
            ),
        }
    }

    fn codegen_terminator_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        let func_name = match func {
            Operand::Constant(c) => {
                if let TyKind::FnDef(def_id, _) = c.ty().kind() {
                    with_no_trimmed_paths!(tcx.def_path_str(*def_id))
                } else {
                    format!("{:?}", func)
                }
            }
            _ => format!("XX{:?}", func),
        };

        println!(
            "[DEBUG] TritonCodegen::codegen_terminator_call: func: {:?} func_name: {:?}",
            func, func_name
        );

        let method: LocalCallHandler<'a, 'tcx> = match func_name.as_str() {
            "core::ops::Mul::mul" => TritonCodegen::codegen_mul_call as LocalCallHandler<'a, 'tcx>,
            "core::ops::Add::add" => TritonCodegen::codegen_add_call as LocalCallHandler<'a, 'tcx>,
            "core::ops::Sub::sub" => TritonCodegen::codegen_sub_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::program_id" => {
                TritonCodegen::codegen_program_id as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::arange" => TritonCodegen::codegen_arange as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::load" => TritonCodegen::codegen_load as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::store" => TritonCodegen::codegen_store as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::maximum" => {
                TritonCodegen::codegen_maximum as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::zeros" => {
                TritonCodegen::codegen_zeros as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::zeros_like" => {
                TritonCodegen::codegen_zeros_like as LocalCallHandler<'a, 'tcx>
            }
            "transmute" | "triton_kitchen_sink::transmute" => {
                TritonCodegen::codegen_transmute_slice as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::Comparison::lt" => {
                TritonCodegen::codegen_lt_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::gt" => TritonCodegen::codegen_gt_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::ge" => TritonCodegen::codegen_ge_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::lt" => TritonCodegen::codegen_triton_lt_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::le" => TritonCodegen::codegen_le_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::eq" => TritonCodegen::codegen_eq_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::ne" => TritonCodegen::codegen_ne_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::lt_scalar" => TritonCodegen::codegen_lt_scalar_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::le_scalar" => TritonCodegen::codegen_le_scalar_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::gt_scalar" => TritonCodegen::codegen_gt_scalar_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::ge_scalar" => TritonCodegen::codegen_ge_scalar_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::eq_scalar" => TritonCodegen::codegen_eq_scalar_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::ne_scalar" => TritonCodegen::codegen_ne_scalar_call as LocalCallHandler<'a, 'tcx>,
            "triton::types::AddOffsets::add_offsets" => {
                TritonCodegen::codegen_add_ptr as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::cast" => {
                TritonCodegen::codegen_cast_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::cat" => {
                TritonCodegen::codegen_cat_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::broadcast" => {
                TritonCodegen::codegen_broadcast_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::broadcast_to" => {
                TritonCodegen::codegen_broadcast_to_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::expand_dims" => {
                TritonCodegen::codegen_expand_dims_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::permute" => {
                TritonCodegen::codegen_permute_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::reshape" => {
                TritonCodegen::codegen_reshape_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::trans" => {
                TritonCodegen::codegen_trans_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::ravel" => {
                TritonCodegen::codegen_ravel_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::view" => {
                TritonCodegen::codegen_view_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::join" => {
                TritonCodegen::codegen_join_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::interleave" => {
                TritonCodegen::codegen_interleave_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::split" => {
                TritonCodegen::codegen_split_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::dot" => {
                TritonCodegen::codegen_dot_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::dot_scaled" => {
                TritonCodegen::codegen_dot_scaled_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::make_block_ptr" => {
                TritonCodegen::codegen_make_block_ptr_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::advance" => {
                TritonCodegen::codegen_advance_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::make_tensor_descriptor" => {
                TritonCodegen::codegen_make_tensor_descriptor_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::load_tensor_descriptor" => {
                TritonCodegen::codegen_load_tensor_descriptor_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::store_tensor_descriptor" => {
                TritonCodegen::codegen_store_tensor_descriptor_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::num_programs" => {
                TritonCodegen::codegen_num_programs_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::full" => {
                TritonCodegen::codegen_full_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::flip" => {
                TritonCodegen::codegen_flip_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::gather" => {
                TritonCodegen::codegen_gather_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::abs" => TritonCodegen::codegen_abs_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::ceil" => TritonCodegen::codegen_ceil_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::floor" => TritonCodegen::codegen_floor_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::cos" => TritonCodegen::codegen_cos_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::sin" => TritonCodegen::codegen_sin_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::exp" => TritonCodegen::codegen_exp_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::exp2" => TritonCodegen::codegen_exp2_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::log" => TritonCodegen::codegen_log_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::log2" => TritonCodegen::codegen_log2_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::rsqrt" => TritonCodegen::codegen_rsqrt_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::sigmoid" => TritonCodegen::codegen_sigmoid_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::sqrt" => TritonCodegen::codegen_sqrt_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::sqrt_rn" => TritonCodegen::codegen_sqrt_rn_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::erf" => TritonCodegen::codegen_erf_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::softmax" => TritonCodegen::codegen_softmax_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::minimum" => TritonCodegen::codegen_minimum_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::clamp" => TritonCodegen::codegen_clamp_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::fma" => TritonCodegen::codegen_fma_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::fdiv" => TritonCodegen::codegen_fdiv_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::div_rn" => TritonCodegen::codegen_div_rn_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::cdiv" => TritonCodegen::codegen_cdiv_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::swizzle2d" => TritonCodegen::codegen_swizzle2d_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::sum" => TritonCodegen::codegen_sum_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::max" => TritonCodegen::codegen_max_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::max_with_indices" => {
                TritonCodegen::codegen_max_with_indices_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::min" => TritonCodegen::codegen_min_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::min_with_indices" => {
                TritonCodegen::codegen_min_with_indices_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::argmax" => TritonCodegen::codegen_argmax_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::argmin" => TritonCodegen::codegen_argmin_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::xor_sum" => TritonCodegen::codegen_xor_sum_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::cumsum" => TritonCodegen::codegen_cumsum_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::cumprod" => TritonCodegen::codegen_cumprod_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::sort" => TritonCodegen::codegen_sort_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::histogram" => TritonCodegen::codegen_histogram_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::reduce" => TritonCodegen::codegen_reduce_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::associative_scan" => {
                TritonCodegen::codegen_associative_scan_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_add" => TritonCodegen::codegen_atomic_add_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::atomic_max" => TritonCodegen::codegen_atomic_max_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::atomic_min" => TritonCodegen::codegen_atomic_min_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::atomic_xchg" => TritonCodegen::codegen_atomic_xchg_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::atomic_cas" => TritonCodegen::codegen_atomic_cas_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::atomic_and" => TritonCodegen::codegen_atomic_and_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::atomic_or" => TritonCodegen::codegen_atomic_or_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::atomic_xor" => TritonCodegen::codegen_atomic_xor_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::umulhi" => TritonCodegen::codegen_umulhi_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::rand" => TritonCodegen::codegen_rand_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::randn" => TritonCodegen::codegen_randn_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::randint" => TritonCodegen::codegen_randint_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::randint4x" => TritonCodegen::codegen_randint4x_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::inline_asm_elementwise" => {
                TritonCodegen::codegen_inline_asm_elementwise_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::multiple_of" => {
                TritonCodegen::codegen_multiple_of_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::max_contiguous" => {
                TritonCodegen::codegen_max_contiguous_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::max_constancy" => {
                TritonCodegen::codegen_max_constancy_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::where_" => TritonCodegen::codegen_where_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::assume" => TritonCodegen::codegen_assume_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::device_assert" => {
                TritonCodegen::codegen_device_assert_call as LocalCallHandler<'a, 'tcx>
            }
            // device_print is generic (has tensor<> params) so its monomorphization is
            // skipped by codegen_function; intercept the call here to avoid dangling tt.call.
            "triton::Triton::device_print" => {
                TritonCodegen::codegen_device_print_call as LocalCallHandler<'a, 'tcx>
            }
            _ => TritonCodegen::codegen_call as LocalCallHandler<'a, 'tcx>,
        };

        let value = method(
            self,
            tcx,
            instance,
            mir,
            func,
            func_name.as_str(),
            args,
            destination,
            target,
            unwind,
            call_source,
            fn_span,
            location,
            mlir_block,
            state,
        )?;

        if let Some(value) = value {
            state.ssa_values.insert(destination.local, value);
        }
        self.codegen_goto(location, &target.expect("target must be Some"), mlir_block, basic_blocks, state)?;
        Ok(())
    }

    pub fn codegen_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        func_name: &str,
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
        let args = args
            .iter()
            .map(|arg| {
                self.codegen_operand(
                    tcx,
                    instance,
                    &arg.node,
                    arg.node.ty(mir, tcx),
                    location,
                    mlir_block,
                    state,
                )
            })
            .collect::<Result<Vec<_>, MlirError>>()?;

        let fn_ty = func.ty(mir, tcx);
        let fn_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(fn_ty),
        );
        let ret_ty = match fn_ty.kind() {
            // Function definitions (e.g., closures and fn items).
            TyKind::FnDef(def_id, substs) => {
                let sig = tcx.fn_sig(*def_id).instantiate(tcx, substs);
                sig.output().skip_binder()
            }
            // For function pointers, combine binder + header and get the output type.
            TyKind::FnPtr(binder, header) => {
                let full_sig = binder.with(*header);
                full_sig.output().skip_binder()
            }
            // Otherwise, fallback to just using the type itself.
            _ => fn_ty,
        };

        // Get the callee from the func operand: this should be the mangled function name.
        // This branch is for both `TyKind::FnDef` and function pointer cases.
        let callee_name = match func {
            Operand::Constant(constant) => {
                let ty = constant.const_.ty();
                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(ty),
                );
                println!("[DEBUG] AXM TritonCodegen::codegen_call: ty: {:?}", ty);
                match ty.kind() {
                    TyKind::FnDef(def_id, substs) => {
                        let typing_env = TypingEnv::post_analysis(tcx, *def_id);
                        if let Some(instance) =
                            Instance::resolve_for_fn_ptr(tcx, typing_env, *def_id, substs)
                        {
                            tcx.symbol_name(instance).name.to_string()
                        } else {
                            func_name.to_string()
                        }
                    }
                    TyKind::FnPtr(binder, header) => {
                        todo!("FnPtr: {:?} {:?}", binder, header);
                    }
                    _ => func_name.to_string(),
                }
            }
            // Try to resolve the function pointer to a DefId if possible.
            // Most common "direct call" case handled above, fallback to func_name param.
            _ => func_name.to_string(),
        };

        eprintln!(
            "[DEBUG] AXM TritonCodegen::codegen_call: callee_name: {:?} {:?}",
            func, callee_name
        );

        // Flatten the return type: unit → [], tuple → multiple types, scalar → one type.
        let result_types: Vec<_> = if ret_ty.is_unit() {
            vec![]
        } else if let TyKind::Tuple(elem_tys) = ret_ty.kind() {
            elem_tys
                .iter()
                .map(|ty| self.type_mapper.map_type(self.module.context(), &tcx, &ty))
                .collect()
        } else {
            vec![self.type_mapper.map_type(self.module.context(), &tcx, &ret_ty)]
        };

        let call_op: Operation<'a> = call(
            self.module.context(),
            location,
            callee_name.as_str(),
            &args,
            &result_types,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        eprintln!("[DEBUG] AXM TritonCodegen::codegen_call: call_op: {:?}", call_op.to_string());

        let result = match result_types.len() {
            0 => None,
            1 => {
                let result = call_op.result(0).expect("Call operation result not found");
                Some(result.into())
            }
            _ => {
                // Multiple return values — caller must route them into tuple_fields.
                // Return the first value here; the tuple routing is done by the caller.
                let result = call_op.result(0).expect("Call operation result not found");
                Some(result.into())
            }
        };

        mlir_block.append_operation(call_op);
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_switch_int<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        discr: &Operand<'tcx>,
        targets: &SwitchTargets,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        // Constant-fold SwitchInt when the discriminant is statically known.
        // This avoids emitting cf.cond_br referencing pruned dead MLIR blocks.
        use crate::mlir::codegen::triton::extract_switch_const;
        if let Some(const_val) = extract_switch_const(tcx, instance, discr, &state.const_disc_locals) {
            let target = targets
                .iter()
                .find(|(val, _)| *val == const_val as u128)
                .map(|(_, bb)| bb)
                .unwrap_or_else(|| targets.otherwise());
            println!(
                "[DEBUG] codegen_switch_int: constant-folding discriminant (val={}) → {:?}",
                const_val, target
            );
            return self.codegen_goto(location, &target, mlir_block, basic_blocks, state);
        }

        let discr_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(discr.ty(mir, tcx)),
        );

        let discr_value =
            self.codegen_operand(tcx, instance, discr, discr_ty, location, mlir_block, state)?;

        let ctx = self.module.context();
        let otherwise_bb = targets.otherwise();
        let cases: Vec<(u128, BasicBlock)> = targets.iter().collect();

        match cases.as_slice() {
            [] => self.codegen_goto(location, &otherwise_bb, mlir_block, basic_blocks, state),
            [(val, target_bb)] => {
                // Emit: %const = arith.constant *val : T
                //        %cmp  = arith.cmpi eq, %discr, %const : T
                //        cf.cond_br %cmp, ^target_bb, ^otherwise_bb
                let discr_mlir_ty = discr_value.r#type();
                let val_attr = IntegerAttribute::new(discr_mlir_ty, *val as i64);
                let const_op: Operation<'a> =
                    melior::dialect::arith::constant(ctx, val_attr.into(), location).into();
                let val_const: Value<'a, 'a> = const_op.result(0).expect("switch const").into();
                mlir_block.append_operation(const_op);

                let i1_ty = IntegerType::new(ctx, 1).into();
                let cmp_op: Operation<'a> =
                    create_cmpi(ctx, location, Predicate::EQ, discr_value, val_const, i1_ty)
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let cmp_result: Value<'a, 'a> = cmp_op.result(0).expect("cmpi result").into();
                mlir_block.append_operation(cmp_op);

                let true_block = *basic_blocks.get(target_bb).expect("switch target block");
                let false_block = *basic_blocks.get(&otherwise_bb).expect("switch otherwise block");

                // Collect phi args with lazy block-arg creation (same as codegen_goto).
                let make_phi_args = |target: &BasicBlock,
                                     target_block: BlockRef<'a, 'a>,
                                     state: &mut CodegenState<'a, 'a>|
                 -> Vec<Value<'a, 'a>> {
                    if let Some(phi_locals) = state.phi_join_locals.get(target).cloned() {
                        phi_locals
                            .iter()
                            .map(|local| {
                                let ssa_val = *state.ssa_values.get(local).unwrap_or_else(|| {
                                    panic!("cond_br: phi local {:?} not in ssa_values", local)
                                });
                                if !state.phi_block_args.contains_key(&(*target, *local)) {
                                    let phi_val =
                                        target_block.add_argument(ssa_val.r#type(), location);
                                    state.phi_block_args.insert((*target, *local), phi_val);
                                }
                                ssa_val
                            })
                            .collect()
                    } else {
                        vec![]
                    }
                };
                let true_phi_args = make_phi_args(target_bb, true_block, state);
                let false_phi_args = make_phi_args(&otherwise_bb, false_block, state);

                let cond_br_op = melior::dialect::cf::cond_br(
                    ctx,
                    cmp_result,
                    &*true_block,
                    &*false_block,
                    &true_phi_args,
                    &false_phi_args,
                    location,
                );
                mlir_block.append_operation(cond_br_op);
                Ok(())
            }
            _ => todo!("SwitchInt with {} cases: {:?}", cases.len(), targets),
        }
    }

    pub(crate) fn codegen_goto(
        &self,
        location: Location<'a>,
        target: &BasicBlock,
        mlir_block: &BlockRef<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        // Back-edge detection: emit scf.yield instead of cf.br when inside a loop body.
        if let Some(header_bb) = state.loop_header_bb {
            if *target == header_bb {
                let yield_vals: Vec<Value<'a, 'a>> = state
                    .loop_iter_carry_locals
                    .iter()
                    .map(|local| {
                        *state.ssa_values.get(local).unwrap_or_else(|| {
                            panic!("scf.yield: iter-carry local {:?} not in ssa_values", local)
                        })
                    })
                    .collect();
                let yield_op = scf::r#yield(&yield_vals, location);
                mlir_block.append_operation(yield_op);
                return Ok(());
            }
            // Within-body jump: all body BBs share a single MLIR block, so skip cf.br.
            if state.loop_body_bbs.contains(target) {
                return Ok(());
            }
        }

        let target_block = *basic_blocks.get(target).unwrap();

        // Collect phi values when branching to a join block.
        // Phi block args are created lazily here (first predecessor wins) so tensor locals
        // get their concrete shape type from the actual SSA value rather than the generic
        // tensor<?xf32> that the MIR type declaration would produce.
        let phi_args: Vec<Value<'a, 'a>> =
            if let Some(phi_locals) = state.phi_join_locals.get(target).cloned() {
                phi_locals
                    .iter()
                    .map(|local| {
                        if let Some(&existing_arg) = state.phi_block_args.get(&(*target, *local)) {
                            // Block arg already created by an earlier predecessor.
                            // We are a later predecessor (processed after the join block in DFS).
                            // ssa_values may hold the join block's own arg (stale) — use the
                            // saved pre-join value instead if available.
                            let current = state.ssa_values.get(local).copied();
                            if current == Some(existing_arg) {
                                // Stale: ssa_values has the join block's own arg.
                                // Use the pre-join saved value from before the join was processed.
                                *state.pre_join_ssa_values.get(&(*target, *local))
                                    .unwrap_or_else(|| panic!(
                                        "codegen_goto: stale phi local {:?} at {:?} but no pre-join save",
                                        local, target
                                    ))
                            } else {
                                // The predecessor redefined this local on its path — use that.
                                current.unwrap_or_else(|| panic!(
                                    "codegen_goto: phi local {:?} not in ssa_values at branch to {:?}",
                                    local, target
                                ))
                            }
                        } else {
                            // First predecessor to reach this join block — create the block arg.
                            let ssa_val = *state.ssa_values.get(local).unwrap_or_else(|| {
                                panic!(
                                    "codegen_goto: phi local {:?} not in ssa_values at branch to {:?}",
                                    local, target
                                )
                            });
                            let phi_val = target_block.add_argument(ssa_val.r#type(), location);
                            state.phi_block_args.insert((*target, *local), phi_val);
                            println!(
                                "[PHI] lazy: added arg for local {:?} at {:?} type {:?}",
                                local, target, ssa_val.r#type()
                            );
                            ssa_val
                        }
                    })
                    .collect()
            } else {
                vec![]
            };

        let br_op: Operation<'a> = if phi_args.is_empty() {
            create_cf_br(self.module.context(), location, &*target_block)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into()
        } else {
            println!(
                "[PHI] codegen_goto: br to {:?} with {} phi args",
                target,
                phi_args.len()
            );
            melior::dialect::cf::br(&*target_block, &phi_args, location)
        };

        mlir_block.append_operation(br_op);
        Ok(())
    }
}
