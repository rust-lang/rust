#![warn(clippy::arithmetic_side_effects)]

mod atomic;
mod simd;

use std::ops::Neg;

use rand::Rng;
use rustc_abi::Size;
use rustc_apfloat::ieee::{IeeeFloat, Semantics};
use rustc_apfloat::{self, Float, Round};
use rustc_middle::mir;
use rustc_middle::ty::{self, FloatTy, ScalarInt};
use rustc_span::{Symbol, sym};

use self::atomic::EvalContextExt as _;
use self::helpers::{ToHost, ToSoft, check_intrinsic_arg_count};
use self::simd::EvalContextExt as _;
use crate::math::{IeeeExt, apply_random_float_error_ulp};
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

                let res = fixed_float_value(intrinsic_name, &[f]).unwrap_or_else(||{
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

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = apply_random_float_error_ulp(
                        this,
                        res,
                        2, // log2(4)
                    );

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    clamp_float_value(intrinsic_name, res)
                });
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

                let res = fixed_float_value(intrinsic_name, &[f]).unwrap_or_else(||{
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

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = apply_random_float_error_ulp(
                        this,
                        res,
                        2, // log2(4)
                    );

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    clamp_float_value(intrinsic_name, res)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            "fmaf32" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let c = this.read_scalar(c)?.to_f32()?;
                let res = a.mul_add(b, c).value;
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }
            "fmaf64" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let c = this.read_scalar(c)?.to_f64()?;
                let res = a.mul_add(b, c).value;
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }

            "fmuladdf32" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let c = this.read_scalar(c)?.to_f32()?;
                let fuse: bool = this.machine.float_nondet && this.machine.rng.get_mut().random();
                let res = if fuse { a.mul_add(b, c).value } else { ((a * b).value + c).value };
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }
            "fmuladdf64" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let c = this.read_scalar(c)?.to_f64()?;
                let fuse: bool = this.machine.float_nondet && this.machine.rng.get_mut().random();
                let res = if fuse { a.mul_add(b, c).value } else { ((a * b).value + c).value };
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }

            "powf32" => {
                let [f1, f2] = check_intrinsic_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f32()?;
                let f2 = this.read_scalar(f2)?.to_f32()?;

                let res = fixed_float_value(intrinsic_name, &[f1, f2]).unwrap_or_else(|| {
                    // Using host floats (but it's fine, this operation does not have guaranteed precision).
                    let res = f1.to_host().powf(f2.to_host()).to_soft();

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    apply_random_float_error_ulp(
                        this, res, 2, // log2(4)
                    )
                });
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }
            "powf64" => {
                let [f1, f2] = check_intrinsic_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f64()?;
                let f2 = this.read_scalar(f2)?.to_f64()?;

                let res = fixed_float_value(intrinsic_name, &[f1, f2]).unwrap_or_else(|| {
                    // Using host floats (but it's fine, this operation does not have guaranteed precision).
                    let res = f1.to_host().powf(f2.to_host()).to_soft();

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    apply_random_float_error_ulp(
                        this, res, 2, // log2(4)
                    )
                });
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }

            "powif32" => {
                let [f, i] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                let i = this.read_scalar(i)?.to_i32()?;

                let res = fixed_powi_float_value(f, i).unwrap_or_else(|| {
                    // Using host floats (but it's fine, this operation does not have guaranteed precision).
                    let res = f.to_host().powi(i).to_soft();

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    apply_random_float_error_ulp(
                        this, res, 2, // log2(4)
                    )
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "powif64" => {
                let [f, i] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                let i = this.read_scalar(i)?.to_i32()?;

                let res = fixed_powi_float_value(f, i).unwrap_or_else(|| {
                    // Using host floats (but it's fine, this operation does not have guaranteed precision).
                    let res = f.to_host().powi(i).to_soft();

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    apply_random_float_error_ulp(
                        this, res, 2, // log2(4)
                    )
                });
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

/// Applies a random ULP floating point error to `val` and returns the new value.
/// So if you want an X ULP error, `ulp_exponent` should be log2(X).
///
/// Will fail if `val` is not a floating point number.
fn apply_random_float_error_to_imm<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    val: ImmTy<'tcx>,
    ulp_exponent: u32,
) -> InterpResult<'tcx, ImmTy<'tcx>> {
    let scalar = val.to_scalar_int()?;
    let res: ScalarInt = match val.layout.ty.kind() {
        ty::Float(FloatTy::F16) =>
            apply_random_float_error_ulp(ecx, scalar.to_f16(), ulp_exponent).into(),
        ty::Float(FloatTy::F32) =>
            apply_random_float_error_ulp(ecx, scalar.to_f32(), ulp_exponent).into(),
        ty::Float(FloatTy::F64) =>
            apply_random_float_error_ulp(ecx, scalar.to_f64(), ulp_exponent).into(),
        ty::Float(FloatTy::F128) =>
            apply_random_float_error_ulp(ecx, scalar.to_f128(), ulp_exponent).into(),
        _ => bug!("intrinsic called with non-float input type"),
    };

    interp_ok(ImmTy::from_scalar_int(res, val.layout))
}

/// For the intrinsics:
/// - sinf32, sinf64
/// - cosf32, cosf64
/// - expf32, expf64, exp2f32, exp2f64
/// - logf32, logf64, log2f32, log2f64, log10f32, log10f64
/// - powf32, powf64
///
/// Returns `Some(output)` if the `intrinsic` results in a defined fixed `output` specified in the C standard
/// (specifically, C23 annex F.10)  when given `args` as arguments. Outputs that are unaffected by a relative error
/// (such as INF and zero) are not handled here, they are assumed to be handled by the underlying
/// implementation. Returns `None` if no specific value is guaranteed.
fn fixed_float_value<S: Semantics>(
    intrinsic_name: &str,
    args: &[IeeeFloat<S>],
) -> Option<IeeeFloat<S>> {
    let one = IeeeFloat::<S>::one();
    match (intrinsic_name, args) {
        // cos(+- 0) = 1
        ("cosf32" | "cosf64", [input]) if input.is_zero() => Some(one),

        // e^0 = 1
        ("expf32" | "expf64" | "exp2f32" | "exp2f64", [input]) if input.is_zero() => Some(one),

        // 1^y = 1 for any y, even a NaN.
        ("powf32" | "powf64", [base, _]) if *base == one => Some(one),

        // (-1)^(±INF) = 1
        ("powf32" | "powf64", [base, exp]) if *base == -one && exp.is_infinite() => Some(one),

        // FIXME(#4286): The C ecosystem is inconsistent with handling sNaN's, some return 1 others propogate
        // the NaN. We should return either 1 or the NaN non-deterministically here.
        // But for now, just handle them all the same.
        // x^(±0) = 1 for any x, even a NaN
        ("powf32" | "powf64", [_, exp]) if exp.is_zero() => Some(one),

        // There are a lot of cases for fixed outputs according to the C Standard, but these are mainly INF or zero
        // which are not affected by the applied error.
        _ => None,
    }
}

/// Returns `Some(output)` if `powi` (called `pown` in C) results in a fixed value specified in the C standard
/// (specifically, C23 annex F.10.4.6) when doing `base^exp`. Otherwise, returns `None`.
fn fixed_powi_float_value<S: Semantics>(base: IeeeFloat<S>, exp: i32) -> Option<IeeeFloat<S>> {
    match (base.category(), exp) {
        // x^0 = 1, if x is not a Signaling NaN
        // FIXME(#4286): The C ecosystem is inconsistent with handling sNaN's, some return 1 others propogate
        // the NaN. We should return either 1 or the NaN non-deterministically here.
        // But for now, just handle them all the same.
        (_, 0) => Some(IeeeFloat::<S>::one()),

        _ => None,
    }
}

/// Given an floating-point operation and a floating-point value, clamps the result to the output
/// range of the given operation.
fn clamp_float_value<S: Semantics>(intrinsic_name: &str, val: IeeeFloat<S>) -> IeeeFloat<S> {
    match intrinsic_name {
        // sin and cos: [-1, 1]
        "sinf32" | "cosf32" | "sinf64" | "cosf64" =>
            val.clamp(IeeeFloat::<S>::one().neg(), IeeeFloat::<S>::one()),
        // exp: [0, +INF]
        "expf32" | "exp2f32" | "expf64" | "exp2f64" =>
            IeeeFloat::<S>::maximum(val, IeeeFloat::<S>::ZERO),
        _ => val,
    }
}
