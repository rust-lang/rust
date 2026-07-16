use rustc_abi::Size;
use rustc_span::Symbol;

use crate::shims::math::compute_crc32;
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_loongarch_intrinsic(
        &mut self,
        link_name: Symbol,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.loongarch.").unwrap();
        match unprefixed_name {
            // Used to implement the crc.w.{b,h,w,d}.w and crcc.w.{b,h,w,d}.w functions.
            // https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html#crc-check-instructions
            // These are only available on LA64, not on LA32, and are part of
            // the LA64 1.0 baseline and therefore always available and don't
            // require a target feature to be enabled.
            "crc.w.b.w" | "crc.w.h.w" | "crc.w.w.w" | "crc.w.d.w" | "crcc.w.b.w" | "crcc.w.h.w"
            | "crcc.w.w.w" | "crcc.w.d.w"
                if this.tcx.pointer_size().bits() == 64 =>
            {
                // The polynomial constants below include the leading 1 bit
                // (e.g. 0x104C11DB7 instead of 0x04C11DB7) which the the
                // polynomial division algorithm requires.
                // Note that Loongson's documentation mentions the numbers
                // 0xEDB88320 for IEEE802.3 and 0x82F63B78 for Castagnoli,
                // which is because their docs put the least significant bit
                // first.
                let (bit_size, polynomial): (u32, u128) = match unprefixed_name {
                    "crc.w.b.w" => (8, 0x104C11DB7),
                    "crc.w.h.w" => (16, 0x104C11DB7),
                    "crc.w.w.w" => (32, 0x104C11DB7),
                    "crc.w.d.w" => (64, 0x104C11DB7),
                    "crcc.w.b.w" => (8, 0x11EDC6F41),
                    "crcc.w.h.w" => (16, 0x11EDC6F41),
                    "crcc.w.w.w" => (32, 0x11EDC6F41),
                    "crcc.w.d.w" => (64, 0x11EDC6F41),
                    _ => unreachable!(),
                };

                let [data, crc] = this.check_shim_sig_unadjusted(link_name, args)?;
                let data = this.read_scalar(data)?;
                let crc = this.read_scalar(crc)?;

                // The CRC accumulator is always i32. The data argument is i32 for
                // b/h/w variants and i64 for the d variant, per the LLVM intrinsic
                // definitions.
                // https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsLoongArch.td
                // LoongArch CRC b/h/w instructions ignore any bits above `bit_size`.
                // https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html#crc-check-instructions
                // Miri's `compute_crc32` requires all higher bits to be zero and may
                // panic otherwise, so we explicitly mask them off here to reproduce the
                // hardware behavior.
                let crc = crc.to_u32()?;
                let data = if bit_size == 64 {
                    data.to_u64()?
                } else {
                    Size::from_bits(bit_size).truncate(data.to_u32()?.into()).try_into().unwrap()
                };

                let result = compute_crc32(crc, data, bit_size, polynomial);
                this.write_scalar(Scalar::from_u32(result), dest)?;
            }
            _ => return interp_ok(false),
        }
        interp_ok(true)
    }
}
