use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_bmi_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.bmi.").unwrap();

        // The intrinsics are suffixed with the bit size of their operands.
        let (is_64_bit, unprefixed_name) = if unprefixed_name.ends_with("64") {
            (true, unprefixed_name.strip_suffix(".64").unwrap_or(""))
        } else {
            (false, unprefixed_name.strip_suffix(".32").unwrap_or(""))
        };

        // All intrinsics of the "bmi" namespace belong to the "bmi2" ISA extension.
        // The exception is "bextr", which belongs to "bmi1".
        let target_feature = if unprefixed_name == "bextr" { "bmi1" } else { "bmi2" };
        this.expect_target_feature_for_intrinsic(link_name, target_feature)?;

        if is_64_bit && this.tcx.sess.target.arch != "x86_64" {
            return interp_ok(EmulateItemResult::NotSupported);
        }

        let [left, right] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
        let left = this.read_scalar(left)?;
        let right = this.read_scalar(right)?;

        let left = if is_64_bit { left.to_u64()? } else { u64::from(left.to_u32()?) };
        let right = if is_64_bit { right.to_u64()? } else { u64::from(right.to_u32()?) };

        let result = match unprefixed_name {
            // Extract a contigous range of bits from an unsigned integer.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_bextr_u32
            "bextr" => {
                let start = u32::try_from(right & 0xff).unwrap();
                let len = u32::try_from((right >> 8) & 0xff).unwrap();
                let shifted = left.checked_shr(start).unwrap_or(0);
                // Keep the `len` lowest bits of `shifted`, or all bits if `len` is too big.
                if len >= 64 { shifted } else { shifted & 1u64.wrapping_shl(len).wrapping_sub(1) }
            }
            // Create a copy of an unsigned integer with bits above a certain index cleared.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_bzhi_u32
            "bzhi" => {
                let index = u32::try_from(right & 0xff).unwrap();
                // Keep the `index` lowest bits of `left`, or all bits if `index` is too big.
                if index >= 64 { left } else { left & 1u64.wrapping_shl(index).wrapping_sub(1) }
            }
            // Extract bit values of an unsigned integer at positions marked by a mask.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_pext_u32
            "pext" => {
                let mut mask = right;
                let mut i = 0u32;
                let mut result = 0;
                // Iterate over the mask one 1-bit at a time, from
                // the least significant bit to the most significant bit.
                while mask != 0 {
                    // Extract the bit marked by the mask's least significant set bit
                    // and put it at position `i` of the result.
                    result |= u64::from(left & (1 << mask.trailing_zeros()) != 0) << i;
                    i = i.wrapping_add(1);
                    // Clear the least significant set bit.
                    mask &= mask.wrapping_sub(1);
                }
                result
            }
            // Deposit bit values of an unsigned integer to positions marked by a mask.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_pdep_u32
            "pdep" => {
                let mut mask = right;
                let mut set = left;
                let mut result = 0;
                // Iterate over the mask one 1-bit at a time, from
                // the least significant bit to the most significant bit.
                while mask != 0 {
                    // Put rightmost bit of `set` at the position of the current `mask` bit.
                    result |= (set & 1) << mask.trailing_zeros();
                    // Go to next bit of `set`.
                    set >>= 1;
                    // Clear the least significant set bit.
                    mask &= mask.wrapping_sub(1);
                }
                result
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        };

        let result = if is_64_bit {
            Scalar::from_u64(result)
        } else {
            Scalar::from_u32(u32::try_from(result).unwrap())
        };
        this.write_scalar(result, dest)?;

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
