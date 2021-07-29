use crate::core_arch::arm_shared::{uint32x4_t, uint8x16_t};

#[allow(improper_ctypes)]
extern "C" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.aese")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.aese")]
    fn vaeseq_u8_(data: uint8x16_t, key: uint8x16_t) -> uint8x16_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.aesd")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.aesd")]
    fn vaesdq_u8_(data: uint8x16_t, key: uint8x16_t) -> uint8x16_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.aesmc")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.aesmc")]
    fn vaesmcq_u8_(data: uint8x16_t) -> uint8x16_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.aesimc")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.aesimc")]
    fn vaesimcq_u8_(data: uint8x16_t) -> uint8x16_t;

    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha1h")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha1h")]
    fn vsha1h_u32_(hash_e: u32) -> u32;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha1su0")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha1su0")]
    fn vsha1su0q_u32_(w0_3: uint32x4_t, w4_7: uint32x4_t, w8_11: uint32x4_t) -> uint32x4_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha1su1")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha1su1")]
    fn vsha1su1q_u32_(tw0_3: uint32x4_t, w12_15: uint32x4_t) -> uint32x4_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha1c")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha1c")]
    fn vsha1cq_u32_(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha1p")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha1p")]
    fn vsha1pq_u32_(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha1m")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha1m")]
    fn vsha1mq_u32_(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t;

    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha256h")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha256h")]
    fn vsha256hq_u32_(hash_abcd: uint32x4_t, hash_efgh: uint32x4_t, wk: uint32x4_t) -> uint32x4_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha256h2")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha256h2")]
    fn vsha256h2q_u32_(hash_efgh: uint32x4_t, hash_abcd: uint32x4_t, wk: uint32x4_t) -> uint32x4_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha256su0")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha256su0")]
    fn vsha256su0q_u32_(w0_3: uint32x4_t, w4_7: uint32x4_t) -> uint32x4_t;
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.crypto.sha256su1")]
    #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.sha256su1")]
    fn vsha256su1q_u32_(tw0_3: uint32x4_t, w8_11: uint32x4_t, w12_15: uint32x4_t) -> uint32x4_t;
}

#[cfg(test)]
use stdarch_test::assert_instr;

// TODO: Use AES for ARM when the minimum LLVM version includes b8baa2a9132498ea286dbb0d03f005760ecc6fdb

/// AES single round encryption.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "aes"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(aese))]
pub unsafe fn vaeseq_u8(data: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
    vaeseq_u8_(data, key)
}

/// AES single round decryption.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "aes"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(aesd))]
pub unsafe fn vaesdq_u8(data: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
    vaesdq_u8_(data, key)
}

/// AES mix columns.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "aes"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(aesmc))]
pub unsafe fn vaesmcq_u8(data: uint8x16_t) -> uint8x16_t {
    vaesmcq_u8_(data)
}

/// AES inverse mix columns.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "aes"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(aesimc))]
pub unsafe fn vaesimcq_u8(data: uint8x16_t) -> uint8x16_t {
    vaesimcq_u8_(data)
}

/// SHA1 fixed rotate.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha1h))]
pub unsafe fn vsha1h_u32(hash_e: u32) -> u32 {
    vsha1h_u32_(hash_e)
}

/// SHA1 hash update accelerator, choose.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha1c))]
pub unsafe fn vsha1cq_u32(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t {
    vsha1cq_u32_(hash_abcd, hash_e, wk)
}

/// SHA1 hash update accelerator, majority.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha1m))]
pub unsafe fn vsha1mq_u32(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t {
    vsha1mq_u32_(hash_abcd, hash_e, wk)
}

/// SHA1 hash update accelerator, parity.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha1p))]
pub unsafe fn vsha1pq_u32(hash_abcd: uint32x4_t, hash_e: u32, wk: uint32x4_t) -> uint32x4_t {
    vsha1pq_u32_(hash_abcd, hash_e, wk)
}

/// SHA1 schedule update accelerator, first part.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha1su0))]
pub unsafe fn vsha1su0q_u32(w0_3: uint32x4_t, w4_7: uint32x4_t, w8_11: uint32x4_t) -> uint32x4_t {
    vsha1su0q_u32_(w0_3, w4_7, w8_11)
}

/// SHA1 schedule update accelerator, second part.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha1su1))]
pub unsafe fn vsha1su1q_u32(tw0_3: uint32x4_t, w12_15: uint32x4_t) -> uint32x4_t {
    vsha1su1q_u32_(tw0_3, w12_15)
}

/// SHA256 hash update accelerator.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha256h))]
pub unsafe fn vsha256hq_u32(
    hash_abcd: uint32x4_t,
    hash_efgh: uint32x4_t,
    wk: uint32x4_t,
) -> uint32x4_t {
    vsha256hq_u32_(hash_abcd, hash_efgh, wk)
}

/// SHA256 hash update accelerator, upper part.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha256h2))]
pub unsafe fn vsha256h2q_u32(
    hash_efgh: uint32x4_t,
    hash_abcd: uint32x4_t,
    wk: uint32x4_t,
) -> uint32x4_t {
    vsha256h2q_u32_(hash_efgh, hash_abcd, wk)
}

/// SHA256 schedule update accelerator, first part.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha256su0))]
pub unsafe fn vsha256su0q_u32(w0_3: uint32x4_t, w4_7: uint32x4_t) -> uint32x4_t {
    vsha256su0q_u32_(w0_3, w4_7)
}

/// SHA256 schedule update accelerator, second part.
#[inline]
#[cfg_attr(not(target_arch = "arm"), target_feature(enable = "sha2"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(test, assert_instr(sha256su1))]
pub unsafe fn vsha256su1q_u32(
    tw0_3: uint32x4_t,
    w8_11: uint32x4_t,
    w12_15: uint32x4_t,
) -> uint32x4_t {
    vsha256su1q_u32_(tw0_3, w8_11, w12_15)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_arch::{arm_shared::*, simd::*};
    use std::mem;
    use stdarch_test::simd_test;

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "aes"))]
    unsafe fn test_vaeseq_u8() {
        let data = mem::transmute(u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8));
        let key = mem::transmute(u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
        let r: u8x16 = mem::transmute(vaeseq_u8(data, key));
        assert_eq!(
            r,
            u8x16::new(
                124, 123, 124, 118, 124, 123, 124, 197, 124, 123, 124, 118, 124, 123, 124, 197
            )
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "aes"))]
    unsafe fn test_vaesdq_u8() {
        let data = mem::transmute(u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8));
        let key = mem::transmute(u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
        let r: u8x16 = mem::transmute(vaesdq_u8(data, key));
        assert_eq!(
            r,
            u8x16::new(9, 213, 9, 251, 9, 213, 9, 56, 9, 213, 9, 251, 9, 213, 9, 56)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "aes"))]
    unsafe fn test_vaesmcq_u8() {
        let data = mem::transmute(u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8));
        let r: u8x16 = mem::transmute(vaesmcq_u8(data));
        assert_eq!(
            r,
            u8x16::new(3, 4, 9, 10, 15, 8, 21, 30, 3, 4, 9, 10, 15, 8, 21, 30)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "aes"))]
    unsafe fn test_vaesimcq_u8() {
        let data = mem::transmute(u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8));
        let r: u8x16 = mem::transmute(vaesimcq_u8(data));
        assert_eq!(
            r,
            u8x16::new(43, 60, 33, 50, 103, 80, 125, 70, 43, 60, 33, 50, 103, 80, 125, 70)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha1h_u32() {
        assert_eq!(vsha1h_u32(0x1234), 0x048d);
        assert_eq!(vsha1h_u32(0x5678), 0x159e);
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha1su0q_u32() {
        let r: u32x4 = mem::transmute(vsha1su0q_u32(
            mem::transmute(u32x4::new(0x1234_u32, 0x5678_u32, 0x9abc_u32, 0xdef0_u32)),
            mem::transmute(u32x4::new(0x1234_u32, 0x5678_u32, 0x9abc_u32, 0xdef0_u32)),
            mem::transmute(u32x4::new(0x1234_u32, 0x5678_u32, 0x9abc_u32, 0xdef0_u32)),
        ));
        assert_eq!(r, u32x4::new(0x9abc, 0xdef0, 0x1234, 0x5678));
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha1su1q_u32() {
        let r: u32x4 = mem::transmute(vsha1su1q_u32(
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
        ));
        assert_eq!(
            r,
            u32x4::new(0x00008898, 0x00019988, 0x00008898, 0x0000acd0)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha1cq_u32() {
        let r: u32x4 = mem::transmute(vsha1cq_u32(
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            0x1234,
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
        ));
        assert_eq!(
            r,
            u32x4::new(0x8a32cbd8, 0x0c518a96, 0x0018a081, 0x0000c168)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha1pq_u32() {
        let r: u32x4 = mem::transmute(vsha1pq_u32(
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            0x1234,
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
        ));
        assert_eq!(
            r,
            u32x4::new(0x469f0ba3, 0x0a326147, 0x80145d7f, 0x00009f47)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha1mq_u32() {
        let r: u32x4 = mem::transmute(vsha1mq_u32(
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            0x1234,
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
        ));
        assert_eq!(
            r,
            u32x4::new(0xaa39693b, 0x0d51bf84, 0x001aa109, 0x0000d278)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha256hq_u32() {
        let r: u32x4 = mem::transmute(vsha256hq_u32(
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
        ));
        assert_eq!(
            r,
            u32x4::new(0x05e9aaa8, 0xec5f4c02, 0x20a1ea61, 0x28738cef)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha256h2q_u32() {
        let r: u32x4 = mem::transmute(vsha256h2q_u32(
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
        ));
        assert_eq!(
            r,
            u32x4::new(0x3745362e, 0x2fb51d00, 0xbd4c529b, 0x968b8516)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha256su0q_u32() {
        let r: u32x4 = mem::transmute(vsha256su0q_u32(
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
        ));
        assert_eq!(
            r,
            u32x4::new(0xe59e1c97, 0x5eaf68da, 0xd7bcb51f, 0x6c8de152)
        );
    }

    #[cfg_attr(target_arch = "arm", simd_test(enable = "crypto"))]
    #[cfg_attr(not(target_arch = "arm"), simd_test(enable = "sha2"))]
    unsafe fn test_vsha256su1q_u32() {
        let r: u32x4 = mem::transmute(vsha256su1q_u32(
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
            mem::transmute(u32x4::new(0x1234, 0x5678, 0x9abc, 0xdef0)),
        ));
        assert_eq!(
            r,
            u32x4::new(0x5e09e8d2, 0x74a6f16b, 0xc966606b, 0xa686ee9f)
        );
    }
}
