use rustc_abi::CanonAbi;
use rustc_middle::mir::BinOp;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_aarch64_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.aarch64.").unwrap();
        match unprefixed_name {
            // Used to implement the vpmaxq_u8 function.
            // Computes the maximum of adjacent pairs; the first half of the output is produced from the
            // `left` input, the second half of the output from the `right` input.
            // https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxq_u8
            "neon.umaxp.v16i8" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, lane_count) = this.project_to_simd(dest)?;
                assert_eq!(left_len, right_len);
                assert_eq!(lane_count, left_len);

                for lane_idx in 0..lane_count {
                    let src = if lane_idx < (lane_count / 2) { &left } else { &right };
                    let src_idx = lane_idx.strict_rem(lane_count / 2);

                    let lhs_lane =
                        this.read_immediate(&this.project_index(src, src_idx.strict_mul(2))?)?;
                    let rhs_lane = this.read_immediate(
                        &this.project_index(src, src_idx.strict_mul(2).strict_add(1))?,
                    )?;

                    // Compute `if lhs > rhs { lhs } else { rhs }`, i.e., `max`.
                    let res_lane = if this
                        .binary_op(BinOp::Gt, &lhs_lane, &rhs_lane)?
                        .to_scalar()
                        .to_bool()?
                    {
                        lhs_lane
                    } else {
                        rhs_lane
                    };

                    let dest = this.project_index(&dest, lane_idx)?;
                    this.write_immediate(*res_lane, &dest)?;
                }
            }
            // Vector table lookup: each index selects a byte from the 16-byte table, out-of-range -> 0.
            // Used to implement vtbl1_u8 function.
            // LLVM does not have a portable shuffle that takes non-const indices
            // so we need to implement this ourselves.
            // https://developer.arm.com/architectures/instruction-sets/intrinsics/vtbl1_u8
            "neon.tbl1.v16i8" => {
                let [table, indices] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (table, table_len) = this.project_to_simd(table)?;
                let (indices, idx_len) = this.project_to_simd(indices)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;
                assert_eq!(table_len, 16);
                assert_eq!(idx_len, dest_len);

                for i in 0..dest_len {
                    let idx = this.read_immediate(&this.project_index(&indices, i)?)?;
                    let idx_u = idx.to_scalar().to_u8()?;
                    let val = if u64::from(idx_u) < table_len {
                        let t = this.read_immediate(&this.project_index(&table, idx_u.into())?)?;
                        t.to_scalar()
                    } else {
                        Scalar::from_u8(0)
                    };
                    this.write_scalar(val, &this.project_index(&dest, i)?)?;
                }
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
