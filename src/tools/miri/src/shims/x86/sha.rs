//! Implements sha256 SIMD instructions of x86 targets
//!
//! The functions that actually compute SHA256 were copied from [RustCrypto's sha256 module].
//!
//! [RustCrypto's sha256 module]: https://github.com/RustCrypto/hashes/blob/6be8466247e936c415d8aafb848697f39894a386/sha2/src/sha256/soft.rs

use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_sha_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "sha")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sha").unwrap();

        fn read<'c>(ecx: &mut MiriInterpCx<'c>, reg: &OpTy<'c>) -> InterpResult<'c, [u32; 4]> {
            let mut res = [0; 4];
            // We reverse the order because x86 is little endian but the copied implementation uses
            // big endian.
            for (i, dst) in res.iter_mut().rev().enumerate() {
                let projected = &ecx.project_index(reg, i.try_into().unwrap())?;
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
                let [a, b, k] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (a_reg, a_len) = this.project_to_simd(a)?;
                let (b_reg, b_len) = this.project_to_simd(b)?;
                let (k_reg, k_len) = this.project_to_simd(k)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(a_len, 4);
                assert_eq!(b_len, 4);
                assert_eq!(k_len, 4);
                assert_eq!(dest_len, 4);

                let a = read(this, &a_reg)?;
                let b = read(this, &b_reg)?;
                let k = read(this, &k_reg)?;

                let result = sha256_digest_round_x2(a, b, k);
                write(this, &dest, result)?;
            }
            // Used to implement the _mm_sha256msg1_epu32 function.
            "256msg1" => {
                let [a, b] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (a_reg, a_len) = this.project_to_simd(a)?;
                let (b_reg, b_len) = this.project_to_simd(b)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(a_len, 4);
                assert_eq!(b_len, 4);
                assert_eq!(dest_len, 4);

                let a = read(this, &a_reg)?;
                let b = read(this, &b_reg)?;

                let result = sha256msg1(a, b);
                write(this, &dest, result)?;
            }
            // Used to implement the _mm_sha256msg2_epu32 function.
            "256msg2" => {
                let [a, b] = this.check_shim(abi, CanonAbi::C, link_name, args)?;

                let (a_reg, a_len) = this.project_to_simd(a)?;
                let (b_reg, b_len) = this.project_to_simd(b)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(a_len, 4);
                assert_eq!(b_len, 4);
                assert_eq!(dest_len, 4);

                let a = read(this, &a_reg)?;
                let b = read(this, &b_reg)?;

                let result = sha256msg2(a, b);
                write(this, &dest, result)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}

#[inline(always)]
fn shr(v: [u32; 4], o: u32) -> [u32; 4] {
    [v[0] >> o, v[1] >> o, v[2] >> o, v[3] >> o]
}

#[inline(always)]
fn shl(v: [u32; 4], o: u32) -> [u32; 4] {
    [v[0] << o, v[1] << o, v[2] << o, v[3] << o]
}

#[inline(always)]
fn or(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
}

#[inline(always)]
fn xor(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}

#[inline(always)]
fn add(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [
        a[0].wrapping_add(b[0]),
        a[1].wrapping_add(b[1]),
        a[2].wrapping_add(b[2]),
        a[3].wrapping_add(b[3]),
    ]
}

fn sha256load(v2: [u32; 4], v3: [u32; 4]) -> [u32; 4] {
    [v3[3], v2[0], v2[1], v2[2]]
}

fn sha256_digest_round_x2(cdgh: [u32; 4], abef: [u32; 4], wk: [u32; 4]) -> [u32; 4] {
    macro_rules! big_sigma0 {
        ($a:expr) => {
            ($a.rotate_right(2) ^ $a.rotate_right(13) ^ $a.rotate_right(22))
        };
    }
    macro_rules! big_sigma1 {
        ($a:expr) => {
            ($a.rotate_right(6) ^ $a.rotate_right(11) ^ $a.rotate_right(25))
        };
    }
    macro_rules! bool3ary_202 {
        ($a:expr, $b:expr, $c:expr) => {
            $c ^ ($a & ($b ^ $c))
        };
    } // Choose, MD5F, SHA1C
    macro_rules! bool3ary_232 {
        ($a:expr, $b:expr, $c:expr) => {
            ($a & $b) ^ ($a & $c) ^ ($b & $c)
        };
    } // Majority, SHA1M

    let [_, _, wk1, wk0] = wk;
    let [a0, b0, e0, f0] = abef;
    let [c0, d0, g0, h0] = cdgh;

    // a round
    let x0 =
        big_sigma1!(e0).wrapping_add(bool3ary_202!(e0, f0, g0)).wrapping_add(wk0).wrapping_add(h0);
    let y0 = big_sigma0!(a0).wrapping_add(bool3ary_232!(a0, b0, c0));
    let (a1, b1, c1, d1, e1, f1, g1, h1) =
        (x0.wrapping_add(y0), a0, b0, c0, x0.wrapping_add(d0), e0, f0, g0);

    // a round
    let x1 =
        big_sigma1!(e1).wrapping_add(bool3ary_202!(e1, f1, g1)).wrapping_add(wk1).wrapping_add(h1);
    let y1 = big_sigma0!(a1).wrapping_add(bool3ary_232!(a1, b1, c1));
    let (a2, b2, _, _, e2, f2, _, _) =
        (x1.wrapping_add(y1), a1, b1, c1, x1.wrapping_add(d1), e1, f1, g1);

    [a2, b2, e2, f2]
}

fn sha256msg1(v0: [u32; 4], v1: [u32; 4]) -> [u32; 4] {
    // sigma 0 on vectors
    #[inline]
    fn sigma0x4(x: [u32; 4]) -> [u32; 4] {
        let t1 = or(shr(x, 7), shl(x, 25));
        let t2 = or(shr(x, 18), shl(x, 14));
        let t3 = shr(x, 3);
        xor(xor(t1, t2), t3)
    }

    add(v0, sigma0x4(sha256load(v0, v1)))
}

fn sha256msg2(v4: [u32; 4], v3: [u32; 4]) -> [u32; 4] {
    macro_rules! sigma1 {
        ($a:expr) => {
            $a.rotate_right(17) ^ $a.rotate_right(19) ^ ($a >> 10)
        };
    }

    let [x3, x2, x1, x0] = v4;
    let [w15, w14, _, _] = v3;

    let w16 = x0.wrapping_add(sigma1!(w14));
    let w17 = x1.wrapping_add(sigma1!(w15));
    let w18 = x2.wrapping_add(sigma1!(w16));
    let w19 = x3.wrapping_add(sigma1!(w17));

    [w19, w18, w17, w16]
}
