#![warn(clippy::arithmetic_side_effects)]

mod atomic;
mod simd;

use rand::Rng;
use rustc_abi::Size;
use rustc_apfloat::{Float, Round};
use rustc_middle::mir;
use rustc_middle::ty::{self, FloatTy};
use rustc_span::{Symbol, sym};

use self::atomic::EvalContextExt as _;
use self::helpers::{ToHost, ToSoft, check_intrinsic_arg_count};
use self::simd::EvalContextExt as _;
use crate::math::apply_random_float_error_to_imm;
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        let this = self.eval_context_mut();

        // Force use of fallback body, if available.
        if this.machine.force_intrinsic_fallback
            && !this.tcx.intrinsic(instance.def_id()).unwrap().must_be_overridden
        {
            return interp_ok(Some(ty::Instance {
                def: ty::InstanceKind::Item(instance.def_id()),
                args: instance.args,
            }));
        }

        // See if the core engine can handle this intrinsic.
        if this.eval_intrinsic(instance, args, dest, ret)? {
            return interp_ok(None);
        }
        let intrinsic_name = this.tcx.item_name(instance.def_id());
        let intrinsic_name = intrinsic_name.as_str();

        // FIXME: avoid allocating memory
        let dest = this.force_allocation(dest)?;

        match this.emulate_intrinsic_by_name(intrinsic_name, instance.args, args, &dest, ret)? {
            EmulateItemResult::NotSupported => {
                // We haven't handled the intrinsic, let's see if we can use a fallback body.
                if this.tcx.intrinsic(instance.def_id()).unwrap().must_be_overridden {
                    throw_unsup_format!("unimplemented intrinsic: `{intrinsic_name}`")
                }
                let intrinsic_fallback_is_spec = Symbol::intern("intrinsic_fallback_is_spec");
                if this
                    .tcx
                    .get_attrs_by_path(instance.def_id(), &[sym::miri, intrinsic_fallback_is_spec])
                    .next()
                    .is_none()
                {
                    throw_unsup_format!(
                        "Miri can only use intrinsic fallback bodies that exactly reflect the specification: they fully check for UB and are as non-deterministic as possible. After verifying that `{intrinsic_name}` does so, add the `#[miri::intrinsic_fallback_is_spec]` attribute to it; also ping @rust-lang/miri when you do that"
                    );
                }
                interp_ok(Some(ty::Instance {
                    def: ty::InstanceKind::Item(instance.def_id()),
                    args: instance.args,
                }))
            }
            EmulateItemResult::NeedsReturn => {
                trace!("{:?}", this.dump_place(&dest.clone().into()));
                this.return_to_block(ret)?;
                interp_ok(None)
            }
            EmulateItemResult::NeedsUnwind => {
                // Jump to the unwind block to begin unwinding.
                this.unwind_to_block(unwind)?;
                interp_ok(None)
            }
            EmulateItemResult::AlreadyJumped => interp_ok(None),
        }
    }

    /// Emulates a Miri-supported intrinsic (not supported by the core engine).
    /// Returns `Ok(true)` if the intrinsic was handled.
    fn emulate_intrinsic_by_name(
        &mut self,
        intrinsic_name: &str,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        if let Some(name) = intrinsic_name.strip_prefix("atomic_") {
            return this.emulate_atomic_intrinsic(name, generic_args, args, dest);
        }
        if let Some(name) = intrinsic_name.strip_prefix("simd_") {
            return this.emulate_simd_intrinsic(name, generic_args, args, dest);
        }

        match intrinsic_name {
            // Basic control flow
            "abort" => {
                throw_machine_stop!(TerminationInfo::Abort(
                    "the program aborted execution".to_owned()
                ));
            }
            "catch_unwind" => {
                this.handle_catch_unwind(args, dest, ret)?;
                // This pushed a stack frame, don't jump to `ret`.
                return interp_ok(EmulateItemResult::AlreadyJumped);
            }

            // Raw memory accesses
            "volatile_load" => {
                let [place] = check_intrinsic_arg_count(args)?;
                let place = this.deref_pointer(place)?;
                this.copy_op(&place, dest)?;
            }
            "volatile_store" => {
                let [place, dest] = check_intrinsic_arg_count(args)?;
                let place = this.deref_pointer(place)?;
                this.copy_op(dest, &place)?;
            }

            "volatile_set_memory" => {
                let [ptr, val_byte, count] = check_intrinsic_arg_count(args)?;
                this.write_bytes_intrinsic(ptr, val_byte, count, "volatile_set_memory")?;
            }

            // Memory model / provenance manipulation
            "ptr_mask" => {
                let [ptr, mask] = check_intrinsic_arg_count(args)?;

                let ptr = this.read_pointer(ptr)?;
                let mask = this.read_target_usize(mask)?;

                let masked_addr = Size::from_bytes(ptr.addr().bytes() & mask);

                this.write_pointer(Pointer::new(ptr.provenance, masked_addr), dest)?;
            }

            // We want to return either `true` or `false` at random, or else something like
            // ```
            // if !is_val_statically_known(0) { unreachable_unchecked(); }
            // ```
            // Would not be considered UB, or the other way around (`is_val_statically_known(0)`).
            "is_val_statically_known" => {
                let [_arg] = check_intrinsic_arg_count(args)?;
                // FIXME: should we check for validity here? It's tricky because we do not have a
                // place. Codegen does not seem to set any attributes like `noundef` for intrinsic
                // calls, so we don't *have* to do anything.
                let branch: bool = this.machine.rng.get_mut().random();
                this.write_scalar(Scalar::from_bool(branch), dest)?;
            }

            "floorf16" | "ceilf16" | "truncf16" | "roundf16" | "round_ties_even_f16" => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f16()?;
                let mode = match intrinsic_name {
                    "floorf16" => Round::TowardNegative,
                    "ceilf16" => Round::TowardPositive,
                    "truncf16" => Round::TowardZero,
                    "roundf16" => Round::NearestTiesToAway,
                    "round_ties_even_f16" => Round::NearestTiesToEven,
                    _ => bug!(),
                };
                let res = f.round_to_integral(mode).value;
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "floorf32" | "ceilf32" | "truncf32" | "roundf32" | "round_ties_even_f32" => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                let mode = match intrinsic_name {
                    "floorf32" => Round::TowardNegative,
                    "ceilf32" => Round::TowardPositive,
                    "truncf32" => Round::TowardZero,
                    "roundf32" => Round::NearestTiesToAway,
                    "round_ties_even_f32" => Round::NearestTiesToEven,
                    _ => bug!(),
                };
                let res = f.round_to_integral(mode).value;
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "floorf64" | "ceilf64" | "truncf64" | "roundf64" | "round_ties_even_f64" => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                let mode = match intrinsic_name {
                    "floorf64" => Round::TowardNegative,
                    "ceilf64" => Round::TowardPositive,
                    "truncf64" => Round::TowardZero,
                    "roundf64" => Round::NearestTiesToAway,
                    "round_ties_even_f64" => Round::NearestTiesToEven,
                    _ => bug!(),
                };
                let res = f.round_to_integral(mode).value;
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "floorf128" | "ceilf128" | "truncf128" | "roundf128" | "round_ties_even_f128" => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f128()?;
                let mode = match intrinsic_name {
                    "floorf128" => Round::TowardNegative,
                    "ceilf128" => Round::TowardPositive,
                    "truncf128" => Round::TowardZero,
                    "roundf128" => Round::NearestTiesToAway,
                    "round_ties_even_f128" => Round::NearestTiesToEven,
                    _ => bug!(),
                };
                let res = f.round_to_integral(mode).value;
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            "sqrtf32" => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                // Sqrt is specified to be fully precise.
                let res = math::sqrt(f);
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "sqrtf64" => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                // Sqrt is specified to be fully precise.
                let res = math::sqrt(f);
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            #[rustfmt::skip]
            | "sinf32"
            | "cosf32"
            | "expf32"
            | "exp2f32"
            | "logf32"
            | "log10f32"
            | "log2f32"
            => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                // Using host floats (but it's fine, these operations do not have
                // guaranteed precision).
                let host = f.to_host();
                let res = match intrinsic_name {
                    "sinf32" => host.sin(),
                    "cosf32" => host.cos(),
                    "expf32" => host.exp(),
                    "exp2f32" => host.exp2(),
                    "logf32" => host.ln(),
                    "log10f32" => host.log10(),
                    "log2f32" => host.log2(),
                    _ => bug!(),
                };
                let res = res.to_soft();
                // Apply a relative error of 16ULP to introduce some non-determinism
                // simulating imprecise implementations and optimizations.
                // FIXME: temporarily disabled as it breaks std tests.
                // let res = apply_random_float_error_ulp(
                //     this,
                //     res,
                //     4, // log2(16)
                // );
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "sinf64"
            | "cosf64"
            | "expf64"
            | "exp2f64"
            | "logf64"
            | "log10f64"
            | "log2f64"
            => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                // Using host floats (but it's fine, these operations do not have
                // guaranteed precision).
                let host = f.to_host();
                let res = match intrinsic_name {
                    "sinf64" => host.sin(),
                    "cosf64" => host.cos(),
                    "expf64" => host.exp(),
                    "exp2f64" => host.exp2(),
                    "logf64" => host.ln(),
                    "log10f64" => host.log10(),
                    "log2f64" => host.log2(),
                    _ => bug!(),
                };
                let res = res.to_soft();
                // Apply a relative error of 16ULP to introduce some non-determinism
                // simulating imprecise implementations and optimizations.
                // FIXME: temporarily disabled as it breaks std tests.
                // let res = apply_random_float_error_ulp(
                //     this,
                //     res,
                //     4, // log2(16)
                // );
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            "fmaf32" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let c = this.read_scalar(c)?.to_f32()?;
                // FIXME: Using host floats, to work around https://github.com/rust-lang/rustc_apfloat/issues/11
                let res = a.to_host().mul_add(b.to_host(), c.to_host()).to_soft();
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }
            "fmaf64" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let c = this.read_scalar(c)?.to_f64()?;
                // FIXME: Using host floats, to work around https://github.com/rust-lang/rustc_apfloat/issues/11
                let res = a.to_host().mul_add(b.to_host(), c.to_host()).to_soft();
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }

            "fmuladdf32" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let c = this.read_scalar(c)?.to_f32()?;
                let fuse: bool = this.machine.rng.get_mut().random();
                let res = if fuse {
                    // FIXME: Using host floats, to work around https://github.com/rust-lang/rustc_apfloat/issues/11
                    a.to_host().mul_add(b.to_host(), c.to_host()).to_soft()
                } else {
                    ((a * b).value + c).value
                };
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }
            "fmuladdf64" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let c = this.read_scalar(c)?.to_f64()?;
                let fuse: bool = this.machine.rng.get_mut().random();
                let res = if fuse {
                    // FIXME: Using host floats, to work around https://github.com/rust-lang/rustc_apfloat/issues/11
                    a.to_host().mul_add(b.to_host(), c.to_host()).to_soft()
                } else {
                    ((a * b).value + c).value
                };
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }

            "powf32" => {
                // FIXME: apply random relative error but without altering behaviour of powf
                let [f1, f2] = check_intrinsic_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f32()?;
                let f2 = this.read_scalar(f2)?.to_f32()?;
                // Using host floats (but it's fine, this operation does not have guaranteed precision).
                let res = f1.to_host().powf(f2.to_host()).to_soft();
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }
            "powf64" => {
                // FIXME: apply random relative error but without altering behaviour of powf
                let [f1, f2] = check_intrinsic_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f64()?;
                let f2 = this.read_scalar(f2)?.to_f64()?;
                // Using host floats (but it's fine, this operation does not have guaranteed precision).
                let res = f1.to_host().powf(f2.to_host()).to_soft();
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }

            "powif32" => {
                // FIXME: apply random relative error but without altering behaviour of powi
                let [f, i] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                let i = this.read_scalar(i)?.to_i32()?;
                // Using host floats (but it's fine, this operation does not have guaranteed precision).
                let res = f.to_host().powi(i).to_soft();
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "powif64" => {
                // FIXME: apply random relative error but without altering behaviour of powi
                let [f, i] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                let i = this.read_scalar(i)?.to_i32()?;
                // Using host floats (but it's fine, this operation does not have guaranteed precision).
                let res = f.to_host().powi(i).to_soft();
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            #[rustfmt::skip]
            | "fadd_fast"
            | "fsub_fast"
            | "fmul_fast"
            | "fdiv_fast"
            | "frem_fast"
            => {
                let [a, b] = check_intrinsic_arg_count(args)?;
                let a = this.read_immediate(a)?;
                let b = this.read_immediate(b)?;
                let op = match intrinsic_name {
                    "fadd_fast" => mir::BinOp::Add,
                    "fsub_fast" => mir::BinOp::Sub,
                    "fmul_fast" => mir::BinOp::Mul,
                    "fdiv_fast" => mir::BinOp::Div,
                    "frem_fast" => mir::BinOp::Rem,
                    _ => bug!(),
                };
                let float_finite = |x: &ImmTy<'tcx>| -> InterpResult<'tcx, bool> {
                    let ty::Float(fty) = x.layout.ty.kind() else {
                        bug!("float_finite: non-float input type {}", x.layout.ty)
                    };
                    interp_ok(match fty {
                        FloatTy::F16 => x.to_scalar().to_f16()?.is_finite(),
                        FloatTy::F32 => x.to_scalar().to_f32()?.is_finite(),
                        FloatTy::F64 => x.to_scalar().to_f64()?.is_finite(),
                        FloatTy::F128 => x.to_scalar().to_f128()?.is_finite(),
                    })
                };
                match (float_finite(&a)?, float_finite(&b)?) {
                    (false, false) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as both parameters",
                    ),
                    (false, _) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as first parameter",
                    ),
                    (_, false) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as second parameter",
                    ),
                    _ => {}
                }
                let res = this.binary_op(op, &a, &b)?;
                // This cannot be a NaN so we also don't have to apply any non-determinism.
                // (Also, `binary_op` already called `generate_nan` if needed.)
                if !float_finite(&res)? {
                    throw_ub_format!("`{intrinsic_name}` intrinsic produced non-finite value as result");
                }
                // Apply a relative error of 4ULP to simulate non-deterministic precision loss
                // due to optimizations.
                let res = apply_random_float_error_to_imm(this, res, 2 /* log2(4) */)?;
                this.write_immediate(*res, dest)?;
            }

            "float_to_int_unchecked" => {
                let [val] = check_intrinsic_arg_count(args)?;
                let val = this.read_immediate(val)?;

                let res = this
                    .float_to_int_checked(&val, dest.layout, Round::TowardZero)?
                    .ok_or_else(|| {
                        err_ub_format!(
                            "`float_to_int_unchecked` intrinsic called on {val} which cannot be represented in target type `{:?}`",
                            dest.layout.ty
                        )
                    })?;

                this.write_immediate(*res, dest)?;
            }

            // Other
            "breakpoint" => {
                let [] = check_intrinsic_arg_count(args)?;
                // normally this would raise a SIGTRAP, which aborts if no debugger is connected
                throw_machine_stop!(TerminationInfo::Abort(format!("trace/breakpoint trap")))
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
