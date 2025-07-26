use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_gfni_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.").unwrap();

        this.expect_target_feature_for_intrinsic(link_name, "gfni")?;
        if unprefixed_name.ends_with(".256") {
            this.expect_target_feature_for_intrinsic(link_name, "avx")?;
        } else if unprefixed_name.ends_with(".512") {
            this.expect_target_feature_for_intrinsic(link_name, "avx512f")?;
        }

        match unprefixed_name {
            // Used to implement the `_mm{, 256, 512}_gf2p8affine_epi64_epi8` functions.
            // See `affine_transform` for details.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=gf2p8affine_
            "vgf2p8affineqb.128" | "vgf2p8affineqb.256" | "vgf2p8affineqb.512" => {
                let [left, right, imm8] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                affine_transform(this, left, right, imm8, dest, /* inverse */ false)?;
            }
            // Used to implement the `_mm{, 256, 512}_gf2p8affineinv_epi64_epi8` functions.
            // See `affine_transform` for details.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=gf2p8affineinv
            "vgf2p8affineinvqb.128" | "vgf2p8affineinvqb.256" | "vgf2p8affineinvqb.512" => {
                let [left, right, imm8] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                affine_transform(this, left, right, imm8, dest, /* inverse */ true)?;
            }
            // Used to implement the `_mm{, 256, 512}_gf2p8mul_epi8` functions.
            // Multiplies packed 8-bit integers in `left` and `right` in the finite field GF(2^8)
            // and store the results in `dst`. The field GF(2^8) is represented in
            // polynomial representation with the reduction polynomial x^8 + x^4 + x^3 + x + 1.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=gf2p8mul
            "vgf2p8mulb.128" | "vgf2p8mulb.256" | "vgf2p8mulb.512" => {
                let [left, right] = this.check_shim(abi, CanonAbi::C, link_name, args)?;
                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(dest_len, right_len);

                for i in 0..dest_len {
                    let left = this.read_scalar(&this.project_index(&left, i)?)?.to_u8()?;
                    let right = this.read_scalar(&this.project_index(&right, i)?)?.to_u8()?;
                    let dest = this.project_index(&dest, i)?;
                    this.write_scalar(Scalar::from_u8(gf2p8_mul(left, right)), &dest)?;
                }
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}

/// Calculates the affine transformation `right * left + imm8` inside the finite field GF(2^8).
/// `right` is an 8x8 bit matrix, `left` and `imm8` are bit vectors.
/// If `inverse` is set, then the inverse transformation with respect to the reduction polynomial
/// x^8 + x^4 + x^3 + x + 1 is performed instead.
fn affine_transform<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    left: &OpTy<'tcx>,
    right: &OpTy<'tcx>,
    imm8: &OpTy<'tcx>,
    dest: &MPlaceTy<'tcx>,
    inverse: bool,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = ecx.project_to_simd(left)?;
    let (right, right_len) = ecx.project_to_simd(right)?;
    let (dest, dest_len) = ecx.project_to_simd(dest)?;

    assert_eq!(dest_len, right_len);
    assert_eq!(dest_len, left_len);

    let imm8 = ecx.read_scalar(imm8)?.to_u8()?;

    // Each 8x8 bit matrix gets multiplied with eight bit vectors.
    // Therefore, the iteration is done in chunks of eight.
    for i in (0..dest_len).step_by(8) {
        // Get the bit matrix.
        let mut matrix = [0u8; 8];
        for j in 0..8 {
            matrix[usize::try_from(j).unwrap()] =
                ecx.read_scalar(&ecx.project_index(&right, i.wrapping_add(j))?)?.to_u8()?;
        }

        // Multiply the matrix with the vector and perform the addition.
        for j in 0..8 {
            let index = i.wrapping_add(j);
            let left = ecx.read_scalar(&ecx.project_index(&left, index)?)?.to_u8()?;
            let left = if inverse { TABLE[usize::from(left)] } else { left };

            let mut res = 0;

            // Do the matrix multiplication.
            for bit in 0u8..8 {
                let mut b = matrix[usize::from(bit)] & left;

                // Calculate the parity bit.
                b = (b & 0b1111) ^ (b >> 4);
                b = (b & 0b11) ^ (b >> 2);
                b = (b & 0b1) ^ (b >> 1);

                res |= b << 7u8.wrapping_sub(bit);
            }

            // Perform the addition.
            res ^= imm8;

            let dest = ecx.project_index(&dest, index)?;
            ecx.write_scalar(Scalar::from_u8(res), &dest)?;
        }
    }

    interp_ok(())
}

/// A lookup table for computing the inverse byte for the inverse affine transformation.
// This is a evaluated at compile time. Trait based conversion is not available.
/// See <https://www.corsix.org/content/galois-field-instructions-2021-cpus> for the
/// definition of `gf_inv` which was used for the creation of this table.
static TABLE: [u8; 256] = {
    let mut array = [0; 256];

    let mut i = 1;
    while i < 256 {
        #[expect(clippy::as_conversions)] // no `try_from` in const...
        let mut x = i as u8;
        let mut y = gf2p8_mul(x, x);
        x = y;
        let mut j = 2;
        while j < 8 {
            x = gf2p8_mul(x, x);
            y = gf2p8_mul(x, y);
            j += 1;
        }
        array[i] = y;
        i += 1;
    }

    array
};

/// Multiplies packed 8-bit integers in `left` and `right` in the finite field GF(2^8)
/// and store the results in `dst`. The field GF(2^8) is represented in
/// polynomial representation with the reduction polynomial x^8 + x^4 + x^3 + x + 1.
/// See <https://www.corsix.org/content/galois-field-instructions-2021-cpus> for details.
// This is a const function. Trait based conversion is not available.
#[expect(clippy::as_conversions)]
const fn gf2p8_mul(left: u8, right: u8) -> u8 {
    // This implementation is based on the `gf2p8mul_byte` definition found inside the Intel intrinsics guide.
    // See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=gf2p8mul
    // for more information.

    const POLYNOMIAL: u32 = 0x11b;

    let left = left as u32;
    let right = right as u32;

    let mut result = 0u32;

    let mut i = 0u32;
    while i < 8 {
        if left & (1 << i) != 0 {
            result ^= right << i;
        }
        i = i.wrapping_add(1);
    }

    let mut i = 14u32;
    while i >= 8 {
        if result & (1 << i) != 0 {
            result ^= POLYNOMIAL << i.wrapping_sub(8);
        }
        i = i.wrapping_sub(1);
    }

    result as u8
}
