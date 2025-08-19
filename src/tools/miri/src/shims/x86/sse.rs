use rustc_abi::CanonAbi;
use rustc_apfloat::ieee::Single;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::{
    FloatBinOp, FloatUnaryOp, bin_op_simd_float_all, bin_op_simd_float_first, unary_op_ps,
    unary_op_ss,
};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_sse_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "sse")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sse.").unwrap();
        // All these intrinsics operate on 128-bit (f32x4) SIMD vectors unless stated otherwise.
        // Many intrinsic names are sufixed with "ps" (packed single) or "ss" (scalar single),
        // where single means single precision floating point (f32). "ps" means thet the operation
        // is performed on each element of the vector, while "ss" means that the operation is
        // performed only on the first element, copying the remaining elements from the input
        // vector (for binary operations, from the left-hand side).
        match unprefixed_name {
            // Used to implement _mm_{min,max}_ss functions.
            // Performs the operations on the first component of `left` and
            // `right` and copies the remaining components from `left`.
            "min.ss" | "max.ss" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "min.ss" => FloatBinOp::Min,
                    "max.ss" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_simd_float_first::<Single>(this, which, left, right, dest)?;
            }
            // Used to implement _mm_min_ps and _mm_max_ps functions.
            // Note that the semantics are a bit different from Rust simd_min
            // and simd_max intrinsics regarding handling of NaN and -0.0: Rust
            // matches the IEEE min/max operations, while x86 has different
            // semantics.
            "min.ps" | "max.ps" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "min.ps" => FloatBinOp::Min,
                    "max.ps" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_simd_float_all::<Single>(this, which, left, right, dest)?;
            }
            // Used to implement _mm_{rcp,rsqrt}_ss functions.
            // Performs the operations on the first component of `op` and
            // copies the remaining components from `op`.
            "rcp.ss" | "rsqrt.ss" => {
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "rcp.ss" => FloatUnaryOp::Rcp,
                    "rsqrt.ss" => FloatUnaryOp::Rsqrt,
                    _ => unreachable!(),
                };

                unary_op_ss(this, which, op, dest)?;
            }
            // Used to implement _mm_{sqrt,rcp,rsqrt}_ps functions.
            // Performs the operations on all components of `op`.
            "rcp.ps" | "rsqrt.ps" => {
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "rcp.ps" => FloatUnaryOp::Rcp,
                    "rsqrt.ps" => FloatUnaryOp::Rsqrt,
                    _ => unreachable!(),
                };

                unary_op_ps(this, which, op, dest)?;
            }
            // Used to implement the _mm_cmp*_ss functions.
            // Performs a comparison operation on the first component of `left`
            // and `right`, returning 0 if false or `u32::MAX` if true. The remaining
            // components are copied from `left`.
            // _mm_cmp_ss is actually an AVX function where the operation is specified
            // by a const parameter.
            // _mm_cmp{eq,lt,le,gt,ge,neq,nlt,nle,ngt,nge,ord,unord}_ss are SSE functions
            // with hard-coded operations.
            "cmp.ss" => {
                let [left, right, imm] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which =
                    FloatBinOp::cmp_from_imm(this, this.read_scalar(imm)?.to_i8()?, link_name)?;

                bin_op_simd_float_first::<Single>(this, which, left, right, dest)?;
            }
            // Used to implement the _mm_cmp*_ps functions.
            // Performs a comparison operation on each component of `left`
            // and `right`. For each component, returns 0 if false or u32::MAX
            // if true.
            // _mm_cmp_ps is actually an AVX function where the operation is specified
            // by a const parameter.
            // _mm_cmp{eq,lt,le,gt,ge,neq,nlt,nle,ngt,nge,ord,unord}_ps are SSE functions
            // with hard-coded operations.
            "cmp.ps" => {
                let [left, right, imm] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which =
                    FloatBinOp::cmp_from_imm(this, this.read_scalar(imm)?.to_i8()?, link_name)?;

                bin_op_simd_float_all::<Single>(this, which, left, right, dest)?;
            }
            // Used to implement _mm_{,u}comi{eq,lt,le,gt,ge,neq}_ss functions.
            // Compares the first component of `left` and `right` and returns
            // a scalar value (0 or 1).
            "comieq.ss" | "comilt.ss" | "comile.ss" | "comigt.ss" | "comige.ss" | "comineq.ss"
            | "ucomieq.ss" | "ucomilt.ss" | "ucomile.ss" | "ucomigt.ss" | "ucomige.ss"
            | "ucomineq.ss" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;

                assert_eq!(left_len, right_len);

                let left = this.read_scalar(&this.project_index(&left, 0)?)?.to_f32()?;
                let right = this.read_scalar(&this.project_index(&right, 0)?)?.to_f32()?;
                // The difference between the com* and ucom* variants is signaling
                // of exceptions when either argument is a quiet NaN. We do not
                // support accessing the SSE status register from miri (or from Rust,
                // for that matter), so we treat both variants equally.
                let res = match unprefixed_name {
                    "comieq.ss" | "ucomieq.ss" => left == right,
                    "comilt.ss" | "ucomilt.ss" => left < right,
                    "comile.ss" | "ucomile.ss" => left <= right,
                    "comigt.ss" | "ucomigt.ss" => left > right,
                    "comige.ss" | "ucomige.ss" => left >= right,
                    "comineq.ss" | "ucomineq.ss" => left != right,
                    _ => unreachable!(),
                };
                this.write_scalar(Scalar::from_i32(i32::from(res)), dest)?;
            }
            // Use to implement the _mm_cvtss_si32, _mm_cvttss_si32,
            // _mm_cvtss_si64 and _mm_cvttss_si64 functions.
            // Converts the first component of `op` from f32 to i32/i64.
            "cvtss2si" | "cvttss2si" | "cvtss2si64" | "cvttss2si64" => {
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let (op, _) = this.project_to_simd(op)?;

                let op = this.read_immediate(&this.project_index(&op, 0)?)?;

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtss2si
                    "cvtss2si" | "cvtss2si64" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttss2si
                    "cvttss2si" | "cvttss2si64" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                let res = this.float_to_int_checked(&op, dest.layout, rnd)?.unwrap_or_else(|| {
                    // Fallback to minimum according to SSE semantics.
                    ImmTy::from_int(dest.layout.size.signed_int_min(), dest.layout)
                });

                this.write_immediate(*res, dest)?;
            }
            // Used to implement the _mm_cvtsi32_ss and _mm_cvtsi64_ss functions.
            // Converts `right` from i32/i64 to f32. Returns a SIMD vector with
            // the result in the first component and the remaining components
            // are copied from `left`.
            // https://www.felixcloutier.com/x86/cvtsi2ss
            "cvtsi2ss" | "cvtsi642ss" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, left_len);

                let right = this.read_immediate(right)?;
                let dest0 = this.project_index(&dest, 0)?;
                let res0 = this.int_to_int_or_float(&right, dest0.layout)?;
                this.write_immediate(*res0, &dest0)?;

                for i in 1..dest_len {
                    this.copy_op(&this.project_index(&left, i)?, &this.project_index(&dest, i)?)?;
                }
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
