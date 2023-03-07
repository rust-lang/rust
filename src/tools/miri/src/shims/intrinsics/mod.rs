mod atomic;
mod simd;

use std::iter;

use log::trace;

use rustc_apfloat::{Float, Round};
use rustc_middle::ty::layout::{IntegerExt, LayoutOf};
use rustc_middle::{
    mir,
    ty::{self, FloatTy, Ty},
};
use rustc_target::abi::{Integer, Size};

use crate::*;
use atomic::EvalContextExt as _;
use helpers::check_arg_count;
use simd::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
        _unwind: StackPopUnwind,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // See if the core engine can handle this intrinsic.
        if this.emulate_intrinsic(instance, args, dest, ret)? {
            return Ok(());
        }

        // All remaining supported intrinsics have a return place.
        let intrinsic_name = this.tcx.item_name(instance.def_id());
        let intrinsic_name = intrinsic_name.as_str();
        let ret = match ret {
            None => throw_unsup_format!("unimplemented (diverging) intrinsic: `{intrinsic_name}`"),
            Some(p) => p,
        };

        // Some intrinsics are special and need the "ret".
        match intrinsic_name {
            "try" => return this.handle_try(args, dest, ret),
            _ => {}
        }

        // The rest jumps to `ret` immediately.
        this.emulate_intrinsic_by_name(intrinsic_name, args, dest)?;

        trace!("{:?}", this.dump_place(**dest));
        this.go_to_block(ret);
        Ok(())
    }

    /// Emulates a Miri-supported intrinsic (not supported by the core engine).
    fn emulate_intrinsic_by_name(
        &mut self,
        intrinsic_name: &str,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        if let Some(name) = intrinsic_name.strip_prefix("atomic_") {
            return this.emulate_atomic_intrinsic(name, args, dest);
        }
        if let Some(name) = intrinsic_name.strip_prefix("simd_") {
            return this.emulate_simd_intrinsic(name, args, dest);
        }

        match intrinsic_name {
            // Miri overwriting CTFE intrinsics.
            "ptr_guaranteed_cmp" => {
                let [left, right] = check_arg_count(args)?;
                let left = this.read_immediate(left)?;
                let right = this.read_immediate(right)?;
                let (val, _overflowed, _ty) =
                    this.overflowing_binary_op(mir::BinOp::Eq, &left, &right)?;
                // We're type punning a bool as an u8 here.
                this.write_scalar(val, dest)?;
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
                let place = this.deref_operand(place)?;
                this.copy_op(&place.into(), dest, /*allow_transmute*/ false)?;
            }
            "volatile_store" => {
                let [place, dest] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                this.copy_op(dest, &place.into(), /*allow_transmute*/ false)?;
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

            "ptr_mask" => {
                let [ptr, mask] = check_arg_count(args)?;

                let ptr = this.read_pointer(ptr)?;
                let mask = this.read_target_usize(mask)?;

                let masked_addr = Size::from_bytes(ptr.addr().bytes() & mask);

                this.write_pointer(Pointer::new(ptr.provenance, masked_addr), dest)?;
            }

            // Floating-point operations
            "fabsf32" => {
                let [f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                // Can be implemented in soft-floats.
                this.write_scalar(Scalar::from_f32(f.abs()), dest)?;
            }
            "fabsf64" => {
                let [f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                // Can be implemented in soft-floats.
                this.write_scalar(Scalar::from_f64(f.abs()), dest)?;
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
            | "floorf32"
            | "ceilf32"
            | "truncf32"
            | "roundf32"
            => {
                let [f] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let f = match intrinsic_name {
                    "sinf32" => f.sin(),
                    "cosf32" => f.cos(),
                    "sqrtf32" => f.sqrt(),
                    "expf32" => f.exp(),
                    "exp2f32" => f.exp2(),
                    "logf32" => f.ln(),
                    "log10f32" => f.log10(),
                    "log2f32" => f.log2(),
                    "floorf32" => f.floor(),
                    "ceilf32" => f.ceil(),
                    "truncf32" => f.trunc(),
                    "roundf32" => f.round(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(f.to_bits()), dest)?;
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
            | "floorf64"
            | "ceilf64"
            | "truncf64"
            | "roundf64"
            => {
                let [f] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let f = match intrinsic_name {
                    "sinf64" => f.sin(),
                    "cosf64" => f.cos(),
                    "sqrtf64" => f.sqrt(),
                    "expf64" => f.exp(),
                    "exp2f64" => f.exp2(),
                    "logf64" => f.ln(),
                    "log10f64" => f.log10(),
                    "log2f64" => f.log2(),
                    "floorf64" => f.floor(),
                    "ceilf64" => f.ceil(),
                    "truncf64" => f.trunc(),
                    "roundf64" => f.round(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(f.to_bits()), dest)?;
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
                    Ok(match x.layout.ty.kind() {
                        ty::Float(FloatTy::F32) => x.to_scalar().to_f32()?.is_finite(),
                        ty::Float(FloatTy::F64) => x.to_scalar().to_f64()?.is_finite(),
                        _ => bug!(
                            "`{intrinsic_name}` called with non-float input type {ty:?}",
                            ty = x.layout.ty,
                        ),
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
                this.binop_ignore_overflow(op, &a, &b, dest)?;
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
                    "minnumf32" => a.min(b),
                    "maxnumf32" => a.max(b),
                    "copysignf32" => a.copy_sign(b),
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
                    "minnumf64" => a.min(b),
                    "maxnumf64" => a.max(b),
                    "copysignf64" => a.copy_sign(b),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "powf32" => {
                let [f, f2] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(f2)?.to_u32()?);
                let res = f.powf(f2);
                this.write_scalar(Scalar::from_u32(res.to_bits()), dest)?;
            }

            "powf64" => {
                let [f, f2] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(f2)?.to_u64()?);
                let res = f.powf(f2);
                this.write_scalar(Scalar::from_u64(res.to_bits()), dest)?;
            }

            "fmaf32" => {
                let [a, b, c] = check_arg_count(args)?;
                // FIXME: Using host floats, to work around https://github.com/rust-lang/miri/issues/2468.
                let a = f32::from_bits(this.read_scalar(a)?.to_u32()?);
                let b = f32::from_bits(this.read_scalar(b)?.to_u32()?);
                let c = f32::from_bits(this.read_scalar(c)?.to_u32()?);
                let res = a.mul_add(b, c);
                this.write_scalar(Scalar::from_u32(res.to_bits()), dest)?;
            }

            "fmaf64" => {
                let [a, b, c] = check_arg_count(args)?;
                // FIXME: Using host floats, to work around https://github.com/rust-lang/miri/issues/2468.
                let a = f64::from_bits(this.read_scalar(a)?.to_u64()?);
                let b = f64::from_bits(this.read_scalar(b)?.to_u64()?);
                let c = f64::from_bits(this.read_scalar(c)?.to_u64()?);
                let res = a.mul_add(b, c);
                this.write_scalar(Scalar::from_u64(res.to_bits()), dest)?;
            }

            "powif32" => {
                let [f, i] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let i = this.read_scalar(i)?.to_i32()?;
                let res = f.powi(i);
                this.write_scalar(Scalar::from_u32(res.to_bits()), dest)?;
            }

            "powif64" => {
                let [f, i] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let i = this.read_scalar(i)?.to_i32()?;
                let res = f.powi(i);
                this.write_scalar(Scalar::from_u64(res.to_bits()), dest)?;
            }

            "float_to_int_unchecked" => {
                let [val] = check_arg_count(args)?;
                let val = this.read_immediate(val)?;

                let res = match val.layout.ty.kind() {
                    ty::Float(FloatTy::F32) =>
                        this.float_to_int_unchecked(val.to_scalar().to_f32()?, dest.layout.ty)?,
                    ty::Float(FloatTy::F64) =>
                        this.float_to_int_unchecked(val.to_scalar().to_f64()?, dest.layout.ty)?,
                    _ =>
                        span_bug!(
                            this.cur_span(),
                            "`float_to_int_unchecked` called with non-float input type {:?}",
                            val.layout.ty
                        ),
                };

                this.write_scalar(res, dest)?;
            }

            // Other
            "breakpoint" => {
                let [] = check_arg_count(args)?;
                // normally this would raise a SIGTRAP, which aborts if no debugger is connected
                throw_machine_stop!(TerminationInfo::Abort(format!("Trace/breakpoint trap")))
            }

            name => throw_unsup_format!("unimplemented intrinsic: `{name}`"),
        }

        Ok(())
    }

    fn float_to_int_unchecked<F>(
        &self,
        f: F,
        dest_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, Scalar<Provenance>>
    where
        F: Float + Into<Scalar<Provenance>>,
    {
        let this = self.eval_context_ref();

        // Step 1: cut off the fractional part of `f`. The result of this is
        // guaranteed to be precisely representable in IEEE floats.
        let f = f.round_to_integral(Round::TowardZero).value;

        // Step 2: Cast the truncated float to the target integer type and see if we lose any information in this step.
        Ok(match dest_ty.kind() {
            // Unsigned
            ty::Uint(t) => {
                let size = Integer::from_uint_ty(this, *t).size();
                let res = f.to_u128(size.bits_usize());
                if res.status.is_empty() {
                    // No status flags means there was no further rounding or other loss of precision.
                    Scalar::from_uint(res.value, size)
                } else {
                    // `f` was not representable in this integer type.
                    throw_ub_format!(
                        "`float_to_int_unchecked` intrinsic called on {f} which cannot be represented in target type `{dest_ty:?}`",
                    );
                }
            }
            // Signed
            ty::Int(t) => {
                let size = Integer::from_int_ty(this, *t).size();
                let res = f.to_i128(size.bits_usize());
                if res.status.is_empty() {
                    // No status flags means there was no further rounding or other loss of precision.
                    Scalar::from_int(res.value, size)
                } else {
                    // `f` was not representable in this integer type.
                    throw_ub_format!(
                        "`float_to_int_unchecked` intrinsic called on {f} which cannot be represented in target type `{dest_ty:?}`",
                    );
                }
            }
            // Nothing else
            _ =>
                span_bug!(
                    this.cur_span(),
                    "`float_to_int_unchecked` called with non-int output type {dest_ty:?}"
                ),
        })
    }
}
