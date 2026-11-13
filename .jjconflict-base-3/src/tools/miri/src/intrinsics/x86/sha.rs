//! Implements sha256 SIMD instructions of x86 targets
//!
//! The functions that actually compute SHA256 were copied from [RustCrypto's sha256 module].
//!
//! [RustCrypto's sha256 module]: https://github.com/RustCrypto/hashes/blob/6be8466247e936c415d8aafb848697f39894a386/sha2/src/sha256/soft.rs

use rustc_span::Symbol;

use crate::intrinsics::math::sha256;
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_sha_intrinsic(
        &mut self,
        link_name: Symbol,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "sha")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sha").unwrap();

        fn read<'c>(ecx: &mut MiriInterpCx<'c>, vec: &OpTy<'c>) -> InterpResult<'c, [u32; 4]> {
            let mut res = [0; 4];
            // We reverse the order because x86 is little endian but the copied implementation uses
            // big endian.
            for (i, dst) in res.iter_mut().rev().enumerate() {
                let projected = &ecx.project_index(vec, i.try_into().unwrap())?;
                *dst = ecx.read_scalar(projected)?.to_u32()?
            }
            interp_ok(res)
        }

        fn write<'c>(
            ecx: &mut MiriInterpCx<'c>,
            dest: &MPlaceTy<'c>,
            val: [u32; 4],
        ) -> InterpResult<'c, ()> {
            // We reverse the order because x86 is little endian but the copied implementation uses
            // big endian.
            for (i, part) in val.into_iter().rev().enumerate() {
                let projected = &ecx.project_index(dest, i.to_u64())?;
                ecx.write_scalar(Scalar::from_u32(part), projected)?;
            }
            interp_ok(())
        }

        match unprefixed_name {
            // Used to implement the _mm_sha256rnds2_epu32 function.
            "256rnds2" => {
                let [a, b, k] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;

                let (a, a_len) = this.project_to_simd(a)?;
                let (b, b_len) = this.project_to_simd(b)?;
                let (k, k_len) = this.project_to_simd(k)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(a_len, 4);
                assert_eq!(b_len, 4);
                assert_eq!(k_len, 4);
                assert_eq!(dest_len, 4);

                let a = read(this, &a)?;
                let b = read(this, &b)?;
                let k = read(this, &k)?;

                let result = sha256_digest_round_x2(a, b, k);
                write(this, &dest, result)?;
            }
            // Used to implement the _mm_sha256msg1_epu32 function.
            "256msg1" => {
                let [a, b] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;

                let (a, a_len) = this.project_to_simd(a)?;
                let (b, b_len) = this.project_to_simd(b)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(a_len, 4);
                assert_eq!(b_len, 4);
                assert_eq!(dest_len, 4);

                let a = read(this, &a)?;
                let b = read(this, &b)?;

                let result = sha256msg1(a, b);
                write(this, &dest, result)?;
            }
            // Used to implement the _mm_sha256msg2_epu32 function.
            "256msg2" => {
                let [a, b] = this.check_shim_sig_llvm_intrinsic(link_name, args)?;

                let (a, a_len) = this.project_to_simd(a)?;
                let (b, b_len) = this.project_to_simd(b)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(a_len, 4);
                assert_eq!(b_len, 4);
                assert_eq!(dest_len, 4);

                let a = read(this, &a)?;
                let b = read(this, &b)?;

                let result = sha256msg2(a, b);
                write(this, &dest, result)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}

fn sha256load(v2: [u32; 4], v3: [u32; 4]) -> [u32; 4] {
    [v3[3], v2[0], v2[1], v2[2]]
}

fn sha256_digest_round_x2(cdgh: [u32; 4], abef: [u32; 4], wk: [u32; 4]) -> [u32; 4] {
    // `sha256rnds2`: two rounds on the abef/cdgh permutation
    // Ref: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_sha256rnds2
    let [_, _, wk1, wk0] = wk;
    let [a0, b0, e0, f0] = abef;
    let [c0, d0, g0, h0] = cdgh;

    let state = sha256::round([a0, b0, c0, d0, e0, f0, g0, h0], wk0);
    let state = sha256::round(state, wk1);

    [state[0], state[1], state[4], state[5]]
}

fn sha256msg1(v0: [u32; 4], v1: [u32; 4]) -> [u32; 4] {
    let x = sha256load(v0, v1);
    [
        v0[0].wrapping_add(sha256::sigma0(x[0])),
        v0[1].wrapping_add(sha256::sigma0(x[1])),
        v0[2].wrapping_add(sha256::sigma0(x[2])),
        v0[3].wrapping_add(sha256::sigma0(x[3])),
    ]
}

fn sha256msg2(v4: [u32; 4], v3: [u32; 4]) -> [u32; 4] {
    let [x3, x2, x1, x0] = v4;
    let [w15, w14, _, _] = v3;

    let w16 = x0.wrapping_add(sha256::sigma1(w14));
    let w17 = x1.wrapping_add(sha256::sigma1(w15));
    let w18 = x2.wrapping_add(sha256::sigma1(w16));
    let w19 = x3.wrapping_add(sha256::sigma1(w17));

    [w19, w18, w17, w16]
}
