use rustc_middle::ty::Ty;
use rustc_span::Symbol;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_aesni_intrinsic(
        &mut self,
        link_name: Symbol,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "aes")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.aesni.").unwrap();

        match unprefixed_name {
            // Used to implement the _mm_aesdec_si128, _mm256_aesdec_epi128
            // and _mm512_aesdec_epi128 functions.
            // Performs one round of an AES decryption on each 128-bit word of
            // `state` with the corresponding 128-bit key of `key`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdec_si128
            "aesdec" | "aesdec.256" | "aesdec.512" => {
                let [state, key] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;
                aes_round(this, state, key, dest, |state, key| {
                    let key = aes::Block::from(key.to_le_bytes());
                    let mut state = aes::Block::from(state.to_le_bytes());
                    // `aes::hazmat::equiv_inv_cipher_round` documentation states that
                    // it performs the same operation as the x86 aesdec instruction.
                    aes::hazmat::equiv_inv_cipher_round(&mut state, &key);
                    u128::from_le_bytes(state.into())
                })?;
            }
            // Used to implement the _mm_aesdeclast_si128, _mm256_aesdeclast_epi128
            // and _mm512_aesdeclast_epi128 functions.
            // Performs last round of an AES decryption on each 128-bit word of
            // `state` with the corresponding 128-bit key of `key`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdeclast_si128
            "aesdeclast" | "aesdeclast.256" | "aesdeclast.512" => {
                let [state, key] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;

                aes_round(this, state, key, dest, |state, key| {
                    let mut state = aes::Block::from(state.to_le_bytes());
                    // `aes::hazmat::equiv_inv_cipher_round` does the following operations:
                    // state = InvShiftRows(state)
                    // state = InvSubBytes(state)
                    // state = InvMixColumns(state)
                    // state = state ^ key
                    // But we need to skip the InvMixColumns.
                    // First, use a zeroed key to skip the XOR.
                    aes::hazmat::equiv_inv_cipher_round(&mut state, &aes::Block::from([0; 16]));
                    // Then, undo the InvMixColumns with MixColumns.
                    aes::hazmat::mix_columns(&mut state);
                    // Finally, do the XOR.
                    u128::from_le_bytes(state.into()) ^ key
                })?;
            }
            // Used to implement the _mm_aesenc_si128, _mm256_aesenc_epi128
            // and _mm512_aesenc_epi128 functions.
            // Performs one round of an AES encryption on each 128-bit word of
            // `state` with the corresponding 128-bit key of `key`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesenc_si128
            "aesenc" | "aesenc.256" | "aesenc.512" => {
                let [state, key] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;
                aes_round(this, state, key, dest, |state, key| {
                    let key = aes::Block::from(key.to_le_bytes());
                    let mut state = aes::Block::from(state.to_le_bytes());
                    // `aes::hazmat::cipher_round` documentation states that
                    // it performs the same operation as the x86 aesenc instruction.
                    aes::hazmat::cipher_round(&mut state, &key);
                    u128::from_le_bytes(state.into())
                })?;
            }
            // Used to implement the _mm_aesenclast_si128, _mm256_aesenclast_epi128
            // and _mm512_aesenclast_epi128 functions.
            // Performs last round of an AES encryption on each 128-bit word of
            // `state` with the corresponding 128-bit key of `key`.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesenclast_si128
            "aesenclast" | "aesenclast.256" | "aesenclast.512" => {
                let [state, key] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;
                aes_round(this, state, key, dest, |state, key| {
                    let mut state = aes::Block::from(state.to_le_bytes());
                    // `aes::hazmat::cipher_round` does the following operations:
                    // state = ShiftRows(state)
                    // state = SubBytes(state)
                    // state = MixColumns(state)
                    // state = state ^ key
                    // But we need to skip the MixColumns.
                    // First, use a zeroed key to skip the XOR.
                    aes::hazmat::cipher_round(&mut state, &aes::Block::from([0; 16]));
                    // Then, undo the MixColumns with InvMixColumns.
                    aes::hazmat::inv_mix_columns(&mut state);
                    // Finally, do the XOR.
                    u128::from_le_bytes(state.into()) ^ key
                })?;
            }
            // Used to implement the _mm_aesimc_si128 function.
            // Performs the AES InvMixColumns operation on `op`
            "aesimc" => {
                let [op] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;
                // Transmute to `u128`
                let op = op.transmute(this.machine.layouts.u128, this)?;
                let dest = dest.transmute(this.machine.layouts.u128, this)?;

                let state = this.read_scalar(&op)?.to_u128()?;
                let mut state = aes::Block::from(state.to_le_bytes());
                aes::hazmat::inv_mix_columns(&mut state);

                this.write_scalar(Scalar::from_u128(u128::from_le_bytes(state.into())), &dest)?;
            }
            // Used to implement the _mm_aeskeygenassist_si128 function.
            // Assists in expanding the AES cipher key.
            "aeskeygenassist" => {
                let [ckey, rcon] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;
                // Transmute `__m128i` to `u128`.
                let ckey = ckey.transmute(this.machine.layouts.u128, this)?;
                let dest = dest.transmute(this.machine.layouts.u128, this)?;

                let rcon = this.read_scalar(rcon)?.to_u8()?;
                let ckey = this.read_scalar(&ckey)?.to_u128()?;

                let res = aeskeygenassist(ckey, rcon);

                this.write_scalar(Scalar::from_u128(res), &dest)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}

// Performs an AES round (given by `f`) on each 128-bit word of
// `state` with the corresponding 128-bit key of `key`.
fn aes_round<'tcx>(
    ecx: &mut crate::MiriInterpCx<'tcx>,
    state: &OpTy<'tcx>,
    key: &OpTy<'tcx>,
    dest: &MPlaceTy<'tcx>,
    f: impl Fn(u128, u128) -> u128,
) -> InterpResult<'tcx, ()> {
    assert_eq!(dest.layout.size, state.layout.size);
    assert_eq!(dest.layout.size, key.layout.size);

    // Transmute arguments to arrays of `u128`.
    assert_eq!(dest.layout.size.bytes() % 16, 0);
    let len = dest.layout.size.bytes() / 16;

    let u128_array_layout = ecx.layout_of(Ty::new_array(ecx.tcx.tcx, ecx.tcx.types.u128, len))?;

    let state = state.transmute(u128_array_layout, ecx)?;
    let key = key.transmute(u128_array_layout, ecx)?;
    let dest = dest.transmute(u128_array_layout, ecx)?;

    for i in 0..len {
        let state = ecx.read_scalar(&ecx.project_index(&state, i)?)?.to_u128()?;
        let key = ecx.read_scalar(&ecx.project_index(&key, i)?)?.to_u128()?;
        let dest = ecx.project_index(&dest, i)?;

        let res = f(state, key);

        ecx.write_scalar(Scalar::from_u128(res), &dest)?;
    }

    interp_ok(())
}

/// AES Key Generation Assist
///
/// From [Intel Intrinsics Guide][1]:
/// ```text
/// X3[31:0] := a[127:96]
/// X2[31:0] := a[95:64]
/// X1[31:0] := a[63:32]
/// X0[31:0] := a[31:0]
/// RCON[31:0] := ZeroExtend32(imm8[7:0])
/// dst[31:0] := SubWord(X1)
/// dst[63:32] := RotWord(SubWord(X1)) XOR RCON
/// dst[95:64] := SubWord(X3)
/// dst[127:96] := RotWord(SubWord(X3)) XOR RCON
/// ```
///
/// [1]: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aeskeygenassist_si128
#[expect(clippy::as_conversions, reason = "deliberately truncating")]
fn aeskeygenassist(a: u128, rcon: u8) -> u128 {
    use crate::intrinsics::math::aes::sub_word;

    let rcon = u32::from(rcon);
    // TODO: use `truncate` method on stabilization
    let x1 = (a >> 32) as u32;
    let x3 = (a >> 96) as u32;

    let x0 = sub_word(x1);
    let x1 = x0.rotate_right(8) ^ rcon;
    let x2 = sub_word(x3);
    let x3 = x2.rotate_right(8) ^ rcon;

    (u128::from(x3) << 96) | (u128::from(x2) << 64) | (u128::from(x1) << 32) | u128::from(x0)
}
