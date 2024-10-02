#![warn(clippy::arithmetic_side_effects)]

mod atomic;
mod simd;

use rand::Rng;
use rustc_apfloat::{Float, Round};
use rustc_middle::mir;
use rustc_middle::ty::{self, FloatTy};
use rustc_span::{Symbol, sym};
use rustc_target::abi::Size;

use self::atomic::EvalContextExt as _;
use self::helpers::{ToHost, ToSoft, check_arg_count};
use self::simd::EvalContextExt as _;
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        let this = self.eval_context_mut();

        // See if the core engine can handle this intrinsic.
        if this.eval_intrinsic(instance, args, dest, ret)? {
            return interp_ok(None);
        }
        let intrinsic_name = this.tcx.item_name(instance.def_id());
        let intrinsic_name = intrinsic_name.as_str();

        match this.emulate_intrinsic_by_name(intrinsic_name, instance.args, args, dest, ret)? {
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
            return this.emulate_atomic_intrinsic(name, args, dest);
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
                let [place] = check_arg_count(args)?;
                let place = this.deref_pointer(place)?;
                this.copy_op(&place, dest)?;
            }
            "volatile_store" => {
                let [place, dest] = check_arg_count(args)?;
                let place = this.deref_pointer(place)?;
                this.copy_op(dest, &place)?;
            }

            "volatile_set_memory" => {
                let [ptr, val_byte, count] = check_arg_count(args)?;
                this.write_bytes_intrinsic(ptr, val_byte, count, "volatile_set_memory")?;
            }

            // Memory model / provenance manipulation
            "ptr_mask" => {
                let [ptr, mask] = check_arg_count(args)?;

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
                let [_arg] = check_arg_count(args)?;
                // FIXME: should we check for validity here? It's tricky because we do not have a
                // place. Codegen does not seem to set any attributes like `noundef` for intrinsic
                // calls, so we don't *have* to do anything.
                let branch: bool = this.machine.rng.get_mut().gen();
                this.write_scalar(Scalar::from_bool(branch), dest)?;
            }

            // Floating-point operations
            "fabsf32" => {
                let [f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                // This is a "bitwise" operation, so there's no NaN non-determinism.
                this.write_scalar(Scalar::from_f32(f.abs()), dest)?;
            }
            "fabsf64" => {
                let [f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                // This is a "bitwise" operation, so there's no NaN non-determinism.
                this.write_scalar(Scalar::from_f64(f.abs()), dest)?;
            }

            "floorf32" | "ceilf32" | "truncf32" | "roundf32" | "rintf32" => {
                let [f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                let mode = match intrinsic_name {
                    "floorf32" => Round::TowardNegative,
                    "ceilf32" => Round::TowardPositive,
                    "truncf32" => Round::TowardZero,
                    "roundf32" => Round::NearestTiesToAway,
                    "rintf32" => Round::NearestTiesToEven,
                    _ => bug!(),
                };
                let res = f.round_to_integral(mode).value;
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "floorf64" | "ceilf64" | "truncf64" | "roundf64" | "rintf64" => {
                let [f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                let mode = match intrinsic_name {
                    "floorf64" => Round::TowardNegative,
                    "ceilf64" => Round::TowardPositive,
                    "truncf64" => Round::TowardZero,
                    "roundf64" => Round::NearestTiesToAway,
                    "rintf64" => Round::NearestTiesToEven,
                    _ => bug!(),
                };
                let res = f.round_to_integral(mode).value;
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            #[rustfmt::skip]
            | "sinf32"
            | "cosf32"
            | "sqrtf32"
            | "expf32"
            | "exp2f32"
            | "logf32"
            | "log10f32"
            | "log2f32"
            => {
                let [f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                // Using host floats (but it's fine, these operations do not have guaranteed precision).
                let f_host = f.to_host();
                let res = match intrinsic_name {
                    "sinf32" => f_host.sin(),
                    "cosf32" => f_host.cos(),
                    "sqrtf32" => f_host.sqrt(), // FIXME Using host floats, this should use full-precision soft-floats
                    "expf32" => f_host.exp(),
                    "exp2f32" => f_host.exp2(),
                    "logf32" => f_host.ln(),
                    "log10f32" => f_host.log10(),
                    "log2f32" => f_host.log2(),
                    _ => bug!(),
                };
                let res = res.to_soft();
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            #[rustfmt::skip]
            | "sinf64"
            | "cosf64"
            | "sqrtf64"
            | "expf64"
            | "exp2f64"
            | "logf64"
            | "log10f64"
            | "log2f64"
            => {
                let [f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                // Using host floats (but it's fine, these operations do not have guaranteed precision).
                let f_host = f.to_host();
                let res = match intrinsic_name {
                    "sinf64" => f_host.sin(),
                    "cosf64" => f_host.cos(),
                    "sqrtf64" => f_host.sqrt(), // FIXME Using host floats, this should use full-precision soft-floats
                    "expf64" => f_host.exp(),
                    "exp2f64" => f_host.exp2(),
                    "logf64" => f_host.ln(),
                    "log10f64" => f_host.log10(),
                    "log2f64" => f_host.log2(),
                    _ => bug!(),
                };
                let res = res.to_soft();
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            "minnumf32" | "maxnumf32" | "copysignf32" => {
                let [a, b] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let res = match intrinsic_name {
                    "minnumf32" => this.adjust_nan(a.min(b), &[a, b]),
                    "maxnumf32" => this.adjust_nan(a.max(b), &[a, b]),
                    "copysignf32" => a.copy_sign(b), // bitwise, no NaN adjustments
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_f32(res), dest)?;
            }
            "minnumf64" | "maxnumf64" | "copysignf64" => {
                let [a, b] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let res = match intrinsic_name {
                    "minnumf64" => this.adjust_nan(a.min(b), &[a, b]),
                    "maxnumf64" => this.adjust_nan(a.max(b), &[a, b]),
                    "copysignf64" => a.copy_sign(b), // bitwise, no NaN adjustments
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "fmaf32" => {
                let [a, b, c] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let c = this.read_scalar(c)?.to_f32()?;
                // FIXME: Using host floats, to work around https://github.com/rust-lang/rustc_apfloat/issues/11
                let res = a.to_host().mul_add(b.to_host(), c.to_host()).to_soft();
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }
            "fmaf64" => {
                let [a, b, c] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let c = this.read_scalar(c)?.to_f64()?;
                // FIXME: Using host floats, to work around https://github.com/rust-lang/rustc_apfloat/issues/11
                let res = a.to_host().mul_add(b.to_host(), c.to_host()).to_soft();
                let res = this.adjust_nan(res, &[a, b, c]);
                this.write_scalar(res, dest)?;
            }

            "powf32" => {
                let [f1, f2] = check_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f32()?;
                let f2 = this.read_scalar(f2)?.to_f32()?;
                // Using host floats (but it's fine, this operation does not have guaranteed precision).
                let res = f1.to_host().powf(f2.to_host()).to_soft();
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }
            "powf64" => {
                let [f1, f2] = check_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f64()?;
                let f2 = this.read_scalar(f2)?.to_f64()?;
                // Using host floats (but it's fine, this operation does not have guaranteed precision).
                let res = f1.to_host().powf(f2.to_host()).to_soft();
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }

            "powif32" => {
                let [f, i] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                let i = this.read_scalar(i)?.to_i32()?;
                // Using host floats (but it's fine, this operation does not have guaranteed precision).
                let res = f.to_host().powi(i).to_soft();
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "powif64" => {
                let [f, i] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                let i = this.read_scalar(i)?.to_i32()?;
                // Using host floats (but it's fine, this operation does not have guaranteed precision).
                let res = f.to_host().powi(i).to_soft();
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            #[rustfmt::skip]
            | "fadd_algebraic"
            | "fsub_algebraic"
            | "fmul_algebraic"
            | "fdiv_algebraic"
            | "frem_algebraic"
            => {
                let [a, b] = check_arg_count(args)?;
                let a = this.read_immediate(a)?;
                let b = this.read_immediate(b)?;
                let op = match intrinsic_name {
                    "fadd_algebraic" => mir::BinOp::Add,
                    "fsub_algebraic" => mir::BinOp::Sub,
                    "fmul_algebraic" => mir::BinOp::Mul,
                    "fdiv_algebraic" => mir::BinOp::Div,
                    "frem_algebraic" => mir::BinOp::Rem,
                    _ => bug!(),
                };
                let res = this.binary_op(op, &a, &b)?;
                // `binary_op` already called `generate_nan` if necessary.
                this.write_immediate(*res, dest)?;
            }

            #[rustfmt::skip]
            | "fadd_fast"
            | "fsub_fast"
            | "fmul_fast"
            | "fdiv_fast"
            | "frem_fast"
            => {
                let [a, b] = check_arg_count(args)?;
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
                if !float_finite(&res)? {
                    throw_ub_format!("`{intrinsic_name}` intrinsic produced non-finite value as result");
                }
                // This cannot be a NaN so we also don't have to apply any non-determinism.
                // (Also, `binary_op` already called `generate_nan` if needed.)
                this.write_immediate(*res, dest)?;
            }

            "float_to_int_unchecked" => {
                let [val] = check_arg_count(args)?;
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
                let [] = check_arg_count(args)?;
                // normally this would raise a SIGTRAP, which aborts if no debugger is connected
                throw_machine_stop!(TerminationInfo::Abort(format!("trace/breakpoint trap")))
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
