mod atomic;
mod simd;

use std::iter;

use rand::Rng;
use rustc_apfloat::{Float, Round};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::{
    mir,
    ty::{self, FloatTy},
};
use rustc_target::abi::Size;

use crate::*;
use atomic::EvalContextExt as _;
use helpers::{check_arg_count, ToHost, ToSoft};
use simd::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Provenance>],
        dest: &MPlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
        _unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // See if the core engine can handle this intrinsic.
        if this.emulate_intrinsic(instance, args, dest, ret)? {
            return Ok(());
        }
        let intrinsic_name = this.tcx.item_name(instance.def_id());
        let intrinsic_name = intrinsic_name.as_str();

        // Handle intrinsics without return place.
        match intrinsic_name {
            "abort" => {
                throw_machine_stop!(TerminationInfo::Abort(
                    "the program aborted execution".to_owned()
                ))
            }
            _ => {}
        }

        // All remaining supported intrinsics have a return place.
        let ret = match ret {
            None => throw_unsup_format!("unimplemented (diverging) intrinsic: `{intrinsic_name}`"),
            Some(p) => p,
        };

        // Some intrinsics are special and need the "ret".
        match intrinsic_name {
            "catch_unwind" => return this.handle_catch_unwind(args, dest, ret),
            _ => {}
        }

        // The rest jumps to `ret` immediately.
        this.emulate_intrinsic_by_name(intrinsic_name, instance.args, args, dest)?;

        trace!("{:?}", this.dump_place(&dest.clone().into()));
        this.go_to_block(ret);
        Ok(())
    }

    /// Emulates a Miri-supported intrinsic (not supported by the core engine).
    fn emulate_intrinsic_by_name(
        &mut self,
        intrinsic_name: &str,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx, Provenance>],
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        if let Some(name) = intrinsic_name.strip_prefix("atomic_") {
            return this.emulate_atomic_intrinsic(name, args, dest);
        }
        if let Some(name) = intrinsic_name.strip_prefix("simd_") {
            return this.emulate_simd_intrinsic(name, generic_args, args, dest);
        }

        match intrinsic_name {
            // Miri overwriting CTFE intrinsics.
            "ptr_guaranteed_cmp" => {
                let [left, right] = check_arg_count(args)?;
                let left = this.read_immediate(left)?;
                let right = this.read_immediate(right)?;
                let val = this.wrapping_binary_op(mir::BinOp::Eq, &left, &right)?;
                // We're type punning a bool as an u8 here.
                this.write_scalar(val.to_scalar(), dest)?;
            }
            "const_allocate" => {
                // For now, for compatibility with the run-time implementation of this, we just return null.
                // See <https://github.com/rust-lang/rust/issues/93935>.
                this.write_null(dest)?;
            }
            "const_deallocate" => {
                // complete NOP
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

            "write_bytes" | "volatile_set_memory" => {
                let [ptr, val_byte, count] = check_arg_count(args)?;
                let ty = ptr.layout.ty.builtin_deref(true).unwrap().ty;
                let ty_layout = this.layout_of(ty)?;
                let val_byte = this.read_scalar(val_byte)?.to_u8()?;
                let ptr = this.read_pointer(ptr)?;
                let count = this.read_target_usize(count)?;
                // `checked_mul` enforces a too small bound (the correct one would probably be target_isize_max),
                // but no actual allocation can be big enough for the difference to be noticeable.
                let byte_count = ty_layout.size.checked_mul(count, this).ok_or_else(|| {
                    err_ub_format!("overflow computing total size of `{intrinsic_name}`")
                })?;
                this.write_bytes_ptr(ptr, iter::repeat(val_byte).take(byte_count.bytes_usize()))?;
            }

            // Memory model / provenance manipulation
            "ptr_mask" => {
                let [ptr, mask] = check_arg_count(args)?;

                let ptr = this.read_pointer(ptr)?;
                let mask = this.read_target_usize(mask)?;

                let masked_addr = Size::from_bytes(ptr.addr().bytes() & mask);

                this.write_pointer(Pointer::new(ptr.provenance, masked_addr), dest)?;
            }
            "retag_box_to_raw" => {
                let [ptr] = check_arg_count(args)?;
                let alloc_ty = generic_args[1].expect_ty();

                let val = this.read_immediate(ptr)?;
                let new_val = if this.machine.borrow_tracker.is_some() {
                    this.retag_box_to_raw(&val, alloc_ty)?
                } else {
                    val
                };
                this.write_immediate(*new_val, dest)?;
            }

            // We want to return either `true` or `false` at random, or else something like
            // ```
            // if !is_val_statically_known(0) { unreachable_unchecked(); }
            // ```
            // Would not be considered UB, or the other way around (`is_val_statically_known(0)`).
            "is_val_statically_known" => {
                let [arg] = check_arg_count(args)?;
                this.validate_operand(arg)?;
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
                // FIXME: Using host floats.
                let f_host = f.to_host();
                let res = match intrinsic_name {
                    "sinf32" => f_host.sin(),
                    "cosf32" => f_host.cos(),
                    "sqrtf32" => f_host.sqrt(),
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
                // FIXME: Using host floats.
                let f_host = f.to_host();
                let res = match intrinsic_name {
                    "sinf64" => f_host.sin(),
                    "cosf64" => f_host.cos(),
                    "sqrtf64" => f_host.sqrt(),
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
                let float_finite = |x: &ImmTy<'tcx, _>| -> InterpResult<'tcx, bool> {
                    let ty::Float(fty) = x.layout.ty.kind() else {
                        bug!("float_finite: non-float input type {}", x.layout.ty)
                    };
                    Ok(match fty {
                        FloatTy::F16 => unimplemented!("f16_f128"),
                        FloatTy::F32 => x.to_scalar().to_f32()?.is_finite(),
                        FloatTy::F64 => x.to_scalar().to_f64()?.is_finite(),
                        FloatTy::F128 => unimplemented!("f16_f128"),
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
                let res = this.wrapping_binary_op(op, &a, &b)?;
                if !float_finite(&res)? {
                    throw_ub_format!("`{intrinsic_name}` intrinsic produced non-finite value as result");
                }
                // This cannot be a NaN so we also don't have to apply any non-determinism.
                // (Also, `wrapping_binary_op` already called `generate_nan` if needed.)
                this.write_immediate(*res, dest)?;
            }

            #[rustfmt::skip]
            | "minnumf32"
            | "maxnumf32"
            | "copysignf32"
            => {
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

            #[rustfmt::skip]
            | "minnumf64"
            | "maxnumf64"
            | "copysignf64"
            => {
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
                // FIXME: Using host floats.
                let res = f1.to_host().powf(f2.to_host()).to_soft();
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }

            "powf64" => {
                let [f1, f2] = check_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f64()?;
                let f2 = this.read_scalar(f2)?.to_f64()?;
                // FIXME: Using host floats.
                let res = f1.to_host().powf(f2.to_host()).to_soft();
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }

            "powif32" => {
                let [f, i] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                let i = this.read_scalar(i)?.to_i32()?;
                // FIXME: Using host floats.
                let res = f.to_host().powi(i).to_soft();
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            "powif64" => {
                let [f, i] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                let i = this.read_scalar(i)?.to_i32()?;
                // FIXME: Using host floats.
                let res = f.to_host().powi(i).to_soft();
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
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

            name => throw_unsup_format!("unimplemented intrinsic: `{name}`"),
        }

        Ok(())
    }
}
