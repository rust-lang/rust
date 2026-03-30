use rustc_abi::CanonAbi;
use rustc_middle::mir::BinOp;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::shims::math::compute_crc32;
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

            // Wrapping pairwise addition.
            //
            // Concatenates the two input vectors and adds adjacent elements. For input vectors `v`
            // and `w` this computes `[v0 + v1, v2 + v3, ..., w0 + w1, w2 + w3, ...]`, using
            // wrapping addition for `+`.
            //
            // Used by `vpadd_{s8, u8, s16, u16, s32, u32}`.
            name if name.starts_with("neon.addp.") => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(left_len, dest_len);

                assert_eq!(left.layout, right.layout);
                assert_eq!(left.layout, dest.layout);

                assert!(dest_len.is_multiple_of(2));
                let half_len = dest_len.strict_div(2);

                for lane_idx in 0..dest_len {
                    // The left and right vectors are concatenated.
                    let (src, src_pair_idx) = if lane_idx < half_len {
                        (&left, lane_idx)
                    } else {
                        (&right, lane_idx.strict_sub(half_len))
                    };
                    // Convert "pair index" into "index of first element of the pair".
                    let i = src_pair_idx.strict_mul(2);

                    let lhs = this.read_immediate(&this.project_index(src, i)?)?;
                    let rhs = this.read_immediate(&this.project_index(src, i.strict_add(1))?)?;

                    // Wrapping addition on the element type.
                    let sum = this.binary_op(BinOp::Add, &lhs, &rhs)?;

                    let dst_lane = this.project_index(&dest, lane_idx)?;
                    this.write_immediate(*sum, &dst_lane)?;
                }
            }

            // Widening pairwise addition.
            //
            // Takes a single input vector, and an output vector with half as many lanes and double
            // the element width. Takes adjacent pairs of elements, widens both, and then adds them
            // together.
            //
            // Used by `vpaddl_{u8, u16, u32}` and `vpaddlq_{u8, u16, u32}`.
            name if name.starts_with("neon.uaddlp.") => {
                let [src] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (src, src_len) = this.project_to_simd(src)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                // Operates pairwise, so src has twice as many lanes.
                assert_eq!(src_len, dest_len.strict_mul(2));

                let src_elem_size = src.layout.field(this, 0).size;
                let dest_elem_size = dest.layout.field(this, 0).size;

                // Widens, so dest elements must be exactly twice as wide.
                assert_eq!(dest_elem_size.bytes(), src_elem_size.bytes().strict_mul(2));

                for dest_idx in 0..dest_len {
                    let src_idx = dest_idx.strict_mul(2);

                    let a_scalar = this.read_scalar(&this.project_index(&src, src_idx)?)?;
                    let b_scalar =
                        this.read_scalar(&this.project_index(&src, src_idx.strict_add(1))?)?;

                    let a_val = a_scalar.to_uint(src_elem_size)?;
                    let b_val = b_scalar.to_uint(src_elem_size)?;

                    // Use addition on u128 to simulate widening addition for the destination type.
                    // This cannot wrap since the element type is at most u64.
                    let sum = a_val.strict_add(b_val);

                    let dst_lane = this.project_index(&dest, dest_idx)?;
                    this.write_scalar(Scalar::from_uint(sum, dest_elem_size), &dst_lane)?;
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
            // Used to implement the __crc32{b,h,w,x} and __crc32c{b,h,w,x} functions.
            // Polynomial 0x04C11DB7 (standard CRC-32):
            // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/CRC32B--CRC32H--CRC32W--CRC32X--CRC32-checksum-
            // Polynomial 0x1EDC6F41 (CRC-32C / Castagnoli):
            // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/CRC32CB--CRC32CH--CRC32CW--CRC32CX--CRC32C-checksum-
            "crc32b" | "crc32h" | "crc32w" | "crc32x" | "crc32cb" | "crc32ch" | "crc32cw"
            | "crc32cx" => {
                this.expect_target_feature_for_intrinsic(link_name, "crc")?;
                // The polynomial constants below include the leading 1 bit
                // (e.g. 0x104C11DB7 instead of 0x04C11DB7) which the ARM docs
                // omit but the polynomial division algorithm requires.
                let (bit_size, polynomial): (u32, u128) = match unprefixed_name {
                    "crc32b" => (8, 0x104C11DB7),
                    "crc32h" => (16, 0x104C11DB7),
                    "crc32w" => (32, 0x104C11DB7),
                    "crc32x" => (64, 0x104C11DB7),
                    "crc32cb" => (8, 0x11EDC6F41),
                    "crc32ch" => (16, 0x11EDC6F41),
                    "crc32cw" => (32, 0x11EDC6F41),
                    "crc32cx" => (64, 0x11EDC6F41),
                    _ => unreachable!(),
                };

                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let left = this.read_scalar(left)?;
                let right = this.read_scalar(right)?;

                // The CRC accumulator is always u32. The data argument is u32 for
                // b/h/w variants and u64 for the x variant, per the LLVM intrinsic
                // definitions (all b/h/w take i32, only x takes i64).
                // https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAArch64.td
                // If the higher bits are non-zero, `compute_crc32` will panic. We should probably
                // raise a proper error instead, but outside stdarch nobody can trigger this anyway.
                let crc = left.to_u32()?;
                let data =
                    if bit_size == 64 { right.to_u64()? } else { u64::from(right.to_u32()?) };

                let result = compute_crc32(crc, data, bit_size, polynomial);
                this.write_scalar(Scalar::from_u32(result), dest)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
