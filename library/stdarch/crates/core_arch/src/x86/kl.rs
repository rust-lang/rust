//! AES Key Locker Intrinsics
//!
//! The Intrinsics here correspond to those in the `keylockerintrin.h` C header.

use crate::core_arch::x86::__m128i;
use crate::ptr;

#[cfg(test)]
use stdarch_test::assert_instr;

#[repr(C, packed)]
struct EncodeKey128Output(u32, __m128i, __m128i, __m128i, __m128i, __m128i, __m128i);

#[repr(C, packed)]
struct EncodeKey256Output(
    u32,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
);

#[repr(C, packed)]
struct AesOutput(u8, __m128i);

#[repr(C, packed)]
struct WideAesOutput(
    u8,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
);

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.loadiwkey"]
    fn loadiwkey(integrity_key: __m128i, key_lo: __m128i, key_hi: __m128i, control: u32);

    #[link_name = "llvm.x86.encodekey128"]
    fn encodekey128(key_metadata: u32, key: __m128i) -> EncodeKey128Output;
    #[link_name = "llvm.x86.encodekey256"]
    fn encodekey256(key_metadata: u32, key_lo: __m128i, key_hi: __m128i) -> EncodeKey256Output;

    #[link_name = "llvm.x86.aesenc128kl"]
    fn aesenc128kl(data: __m128i, handle: *const u8) -> AesOutput;
    #[link_name = "llvm.x86.aesdec128kl"]
    fn aesdec128kl(data: __m128i, handle: *const u8) -> AesOutput;
    #[link_name = "llvm.x86.aesenc256kl"]
    fn aesenc256kl(data: __m128i, handle: *const u8) -> AesOutput;
    #[link_name = "llvm.x86.aesdec256kl"]
    fn aesdec256kl(data: __m128i, handle: *const u8) -> AesOutput;

    #[link_name = "llvm.x86.aesencwide128kl"]
    fn aesencwide128kl(
        handle: *const u8,
        i0: __m128i,
        i1: __m128i,
        i2: __m128i,
        i3: __m128i,
        i4: __m128i,
        i5: __m128i,
        i6: __m128i,
        i7: __m128i,
    ) -> WideAesOutput;
    #[link_name = "llvm.x86.aesdecwide128kl"]
    fn aesdecwide128kl(
        handle: *const u8,
        i0: __m128i,
        i1: __m128i,
        i2: __m128i,
        i3: __m128i,
        i4: __m128i,
        i5: __m128i,
        i6: __m128i,
        i7: __m128i,
    ) -> WideAesOutput;
    #[link_name = "llvm.x86.aesencwide256kl"]
    fn aesencwide256kl(
        handle: *const u8,
        i0: __m128i,
        i1: __m128i,
        i2: __m128i,
        i3: __m128i,
        i4: __m128i,
        i5: __m128i,
        i6: __m128i,
        i7: __m128i,
    ) -> WideAesOutput;
    #[link_name = "llvm.x86.aesdecwide256kl"]
    fn aesdecwide256kl(
        handle: *const u8,
        i0: __m128i,
        i1: __m128i,
        i2: __m128i,
        i3: __m128i,
        i4: __m128i,
        i5: __m128i,
        i6: __m128i,
        i7: __m128i,
    ) -> WideAesOutput;
}

/// Load internal wrapping key (IWKey). The 32-bit unsigned integer `control` specifies IWKey's KeySource
/// and whether backing up the key is permitted. IWKey's 256-bit encryption key is loaded from `key_lo`
/// and `key_hi`.
///
///  - `control[0]`: NoBackup bit. If set, the IWKey cannot be backed up.
///  - `control[1:4]`: KeySource bits. These bits specify the encoding method of the IWKey. The only
///    allowed values are `0` (AES GCM SIV wrapping algorithm with the specified key) and `1` (AES GCM
///    SIV wrapping algorithm with random keys enforced by hardware). After calling `_mm_loadiwkey` with
///    KeySource set to `1`, software must check `ZF` to ensure that the key was loaded successfully.
///    Using any other value may result in a General Protection Exception.
///  - `control[5:31]`: Reserved for future use, must be set to `0`.
///
/// Note that setting the NoBackup bit and using the KeySource value `1` requires hardware support. These
/// permissions can be found by calling `__cpuid(0x19)` and checking the `ECX[0:1]` bits. Failing to follow
/// these restrictions may result in a General Protection Exception.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadiwkey)
#[inline]
#[target_feature(enable = "kl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(loadiwkey))]
pub unsafe fn _mm_loadiwkey(
    control: u32,
    integrity_key: __m128i,
    key_lo: __m128i,
    key_hi: __m128i,
) {
    loadiwkey(integrity_key, key_lo, key_hi, control);
}

/// Wrap a 128-bit AES key into a 384-bit key handle and stores it in `handle`. Returns the `control`
/// parameter used to create the IWKey.
///
///  - `key_params[0]`: If set, this key can only be used by the Kernel.
///  - `key_params[1]`: If set, this key can not be used to encrypt.
///  - `key_params[2]`: If set, this key can not be used to decrypt.
///  - `key_params[31:3]`: Reserved for future use, must be set to `0`.
///
/// Note that these restrictions need hardware support, and the supported restrictions can be found by
/// calling `__cpuid(0x19)` and checking the `EAX[0:2]` bits. Failing to follow these restrictions may
/// result in a General Protection Exception.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_encodekey128_u32)
#[inline]
#[target_feature(enable = "kl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(encodekey128))]
pub unsafe fn _mm_encodekey128_u32(key_params: u32, key: __m128i, handle: *mut u8) -> u32 {
    let EncodeKey128Output(control, key0, key1, key2, _, _, _) = encodekey128(key_params, key);
    ptr::write_unaligned(handle.cast(), [key0, key1, key2]);
    control
}

/// Wrap a 256-bit AES key into a 512-bit key handle and stores it in `handle`. Returns the `control`
/// parameter used to create the IWKey.
///
///  - `key_params[0]`: If set, this key can only be used by the Kernel.
///  - `key_params[1]`: If set, this key can not be used to encrypt.
///  - `key_params[2]`: If set, this key can not be used to decrypt.
///  - `key_params[31:3]`: Reserved for future use, must be set to `0`.
///
/// Note that these restrictions need hardware support, and the supported restrictions can be found by
/// calling `__cpuid(0x19)` and checking the `EAX[0:2]` bits. Failing to follow these restrictions may
/// result in a General Protection Exception.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_encodekey256_u32)
#[inline]
#[target_feature(enable = "kl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(encodekey256))]
pub unsafe fn _mm_encodekey256_u32(
    key_params: u32,
    key_lo: __m128i,
    key_hi: __m128i,
    handle: *mut u8,
) -> u32 {
    let EncodeKey256Output(control, key0, key1, key2, key3, _, _, _) =
        encodekey256(key_params, key_lo, key_hi);
    ptr::write_unaligned(handle.cast(), [key0, key1, key2, key3]);
    control
}

/// Encrypt 10 rounds of unsigned 8-bit integers in `input` using 128-bit AES key specified in the
/// 384-bit key handle `handle`. Store the resulting unsigned 8-bit integers into the corresponding
/// elements of `output`. Returns `0` if the operation was successful, and `1` if the operation failed
/// due to a handle violation.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesenc128kl_u8)
#[inline]
#[target_feature(enable = "kl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(aesenc128kl))]
pub unsafe fn _mm_aesenc128kl_u8(output: *mut __m128i, input: __m128i, handle: *const u8) -> u8 {
    let AesOutput(status, result) = aesenc128kl(input, handle);
    *output = result;
    status
}

/// Decrypt 10 rounds of unsigned 8-bit integers in `input` using 128-bit AES key specified in the
/// 384-bit key handle `handle`. Store the resulting unsigned 8-bit integers into the corresponding
/// elements of `output`. Returns `0` if the operation was successful, and `1` if the operation failed
/// due to a handle violation.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdec128kl_u8)
#[inline]
#[target_feature(enable = "kl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(aesdec128kl))]
pub unsafe fn _mm_aesdec128kl_u8(output: *mut __m128i, input: __m128i, handle: *const u8) -> u8 {
    let AesOutput(status, result) = aesdec128kl(input, handle);
    *output = result;
    status
}

/// Encrypt 14 rounds of unsigned 8-bit integers in `input` using 256-bit AES key specified in the
/// 512-bit key handle `handle`. Store the resulting unsigned 8-bit integers into the corresponding
/// elements of `output`. Returns `0` if the operation was successful, and `1` if the operation failed
/// due to a handle violation.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesenc256kl_u8)
#[inline]
#[target_feature(enable = "kl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(aesenc256kl))]
pub unsafe fn _mm_aesenc256kl_u8(output: *mut __m128i, input: __m128i, handle: *const u8) -> u8 {
    let AesOutput(status, result) = aesenc256kl(input, handle);
    *output = result;
    status
}

/// Decrypt 14 rounds of unsigned 8-bit integers in `input` using 256-bit AES key specified in the
/// 512-bit key handle `handle`. Store the resulting unsigned 8-bit integers into the corresponding
/// elements of `output`. Returns `0` if the operation was successful, and `1` if the operation failed
/// due to a handle violation.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdec256kl_u8)
#[inline]
#[target_feature(enable = "kl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(aesdec256kl))]
pub unsafe fn _mm_aesdec256kl_u8(output: *mut __m128i, input: __m128i, handle: *const u8) -> u8 {
    let AesOutput(status, result) = aesdec256kl(input, handle);
    *output = result;
    status
}

/// Encrypt 10 rounds of 8 groups of unsigned 8-bit integers in `input` using 128-bit AES key specified
/// in the 384-bit key handle `handle`. Store the resulting unsigned 8-bit integers into the corresponding
/// elements of `output`. Returns `0` if the operation was successful, and `1` if the operation failed
/// due to a handle violation.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesencwide128kl_u8)
#[inline]
#[target_feature(enable = "widekl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(aesencwide128kl))]
pub unsafe fn _mm_aesencwide128kl_u8(
    output: *mut __m128i,
    input: *const __m128i,
    handle: *const u8,
) -> u8 {
    let input = &*ptr::slice_from_raw_parts(input, 8);
    let WideAesOutput(status, out0, out1, out2, out3, out4, out5, out6, out7) = aesencwide128kl(
        handle, input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
    );
    *output.cast() = [out0, out1, out2, out3, out4, out5, out6, out7];
    status
}

/// Decrypt 10 rounds of 8 groups of unsigned 8-bit integers in `input` using 128-bit AES key specified
/// in the 384-bit key handle `handle`. Store the resulting unsigned 8-bit integers into the corresponding
/// elements of `output`. Returns `0` if the operation was successful, and `1` if the operation failed
/// due to a handle violation.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdecwide128kl_u8)
#[inline]
#[target_feature(enable = "widekl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(aesdecwide128kl))]
pub unsafe fn _mm_aesdecwide128kl_u8(
    output: *mut __m128i,
    input: *const __m128i,
    handle: *const u8,
) -> u8 {
    let input = &*ptr::slice_from_raw_parts(input, 8);
    let WideAesOutput(status, out0, out1, out2, out3, out4, out5, out6, out7) = aesdecwide128kl(
        handle, input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
    );
    *output.cast() = [out0, out1, out2, out3, out4, out5, out6, out7];
    status
}

/// Encrypt 14 rounds of 8 groups of unsigned 8-bit integers in `input` using 256-bit AES key specified
/// in the 512-bit key handle `handle`. Store the resulting unsigned 8-bit integers into the corresponding
/// elements of `output`. Returns `0` if the operation was successful, and `1` if the operation failed
/// due to a handle violation.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesencwide256kl_u8)
#[inline]
#[target_feature(enable = "widekl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(aesencwide256kl))]
pub unsafe fn _mm_aesencwide256kl_u8(
    output: *mut __m128i,
    input: *const __m128i,
    handle: *const u8,
) -> u8 {
    let input = &*ptr::slice_from_raw_parts(input, 8);
    let WideAesOutput(status, out0, out1, out2, out3, out4, out5, out6, out7) = aesencwide256kl(
        handle, input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
    );
    *output.cast() = [out0, out1, out2, out3, out4, out5, out6, out7];
    status
}

/// Decrypt 14 rounds of 8 groups of unsigned 8-bit integers in `input` using 256-bit AES key specified
/// in the 512-bit key handle `handle`. Store the resulting unsigned 8-bit integers into the corresponding
/// elements of `output`. Returns `0` if the operation was successful, and `1` if the operation failed
/// due to a handle violation.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_aesdecwide256kl_u8)
#[inline]
#[target_feature(enable = "widekl")]
#[unstable(feature = "keylocker_x86", issue = "134813")]
#[cfg_attr(test, assert_instr(aesdecwide256kl))]
pub unsafe fn _mm_aesdecwide256kl_u8(
    output: *mut __m128i,
    input: *const __m128i,
    handle: *const u8,
) -> u8 {
    let input = &*ptr::slice_from_raw_parts(input, 8);
    let WideAesOutput(status, out0, out1, out2, out3, out4, out5, out6, out7) = aesdecwide256kl(
        handle, input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
    );
    *output.cast() = [out0, out1, out2, out3, out4, out5, out6, out7];
    status
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[target_feature(enable = "kl")]
    unsafe fn encodekey128() -> [u8; 48] {
        let mut handle = [0; 48];
        let _ = _mm_encodekey128_u32(0, _mm_setzero_si128(), handle.as_mut_ptr());
        handle
    }

    #[target_feature(enable = "kl")]
    unsafe fn encodekey256() -> [u8; 64] {
        let mut handle = [0; 64];
        let _ = _mm_encodekey256_u32(
            0,
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            handle.as_mut_ptr(),
        );
        handle
    }

    #[simd_test(enable = "kl")]
    unsafe fn test_mm_encodekey128_u32() {
        encodekey128();
    }

    #[simd_test(enable = "kl")]
    unsafe fn test_mm_encodekey256_u32() {
        encodekey256();
    }

    #[simd_test(enable = "kl")]
    unsafe fn test_mm_aesenc128kl_u8() {
        let mut buffer = _mm_setzero_si128();
        let key = encodekey128();

        for _ in 0..100 {
            let status = _mm_aesenc128kl_u8(&mut buffer, buffer, key.as_ptr());
            assert_eq!(status, 0);
        }
        for _ in 0..100 {
            let status = _mm_aesdec128kl_u8(&mut buffer, buffer, key.as_ptr());
            assert_eq!(status, 0);
        }

        assert_eq_m128i(buffer, _mm_setzero_si128());
    }

    #[simd_test(enable = "kl")]
    unsafe fn test_mm_aesdec128kl_u8() {
        let mut buffer = _mm_setzero_si128();
        let key = encodekey128();

        for _ in 0..100 {
            let status = _mm_aesdec128kl_u8(&mut buffer, buffer, key.as_ptr());
            assert_eq!(status, 0);
        }
        for _ in 0..100 {
            let status = _mm_aesenc128kl_u8(&mut buffer, buffer, key.as_ptr());
            assert_eq!(status, 0);
        }

        assert_eq_m128i(buffer, _mm_setzero_si128());
    }

    #[simd_test(enable = "kl")]
    unsafe fn test_mm_aesenc256kl_u8() {
        let mut buffer = _mm_setzero_si128();
        let key = encodekey256();

        for _ in 0..100 {
            let status = _mm_aesenc256kl_u8(&mut buffer, buffer, key.as_ptr());
            assert_eq!(status, 0);
        }
        for _ in 0..100 {
            let status = _mm_aesdec256kl_u8(&mut buffer, buffer, key.as_ptr());
            assert_eq!(status, 0);
        }

        assert_eq_m128i(buffer, _mm_setzero_si128());
    }

    #[simd_test(enable = "kl")]
    unsafe fn test_mm_aesdec256kl_u8() {
        let mut buffer = _mm_setzero_si128();
        let key = encodekey256();

        for _ in 0..100 {
            let status = _mm_aesdec256kl_u8(&mut buffer, buffer, key.as_ptr());
            assert_eq!(status, 0);
        }
        for _ in 0..100 {
            let status = _mm_aesenc256kl_u8(&mut buffer, buffer, key.as_ptr());
            assert_eq!(status, 0);
        }

        assert_eq_m128i(buffer, _mm_setzero_si128());
    }

    #[simd_test(enable = "widekl")]
    unsafe fn test_mm_aesencwide128kl_u8() {
        let mut buffer = [_mm_setzero_si128(); 8];
        let key = encodekey128();

        for _ in 0..100 {
            let status = _mm_aesencwide128kl_u8(buffer.as_mut_ptr(), buffer.as_ptr(), key.as_ptr());
            assert_eq!(status, 0);
        }
        for _ in 0..100 {
            let status = _mm_aesdecwide128kl_u8(buffer.as_mut_ptr(), buffer.as_ptr(), key.as_ptr());
            assert_eq!(status, 0);
        }

        for elem in buffer {
            assert_eq_m128i(elem, _mm_setzero_si128());
        }
    }

    #[simd_test(enable = "widekl")]
    unsafe fn test_mm_aesdecwide128kl_u8() {
        let mut buffer = [_mm_setzero_si128(); 8];
        let key = encodekey128();

        for _ in 0..100 {
            let status = _mm_aesdecwide128kl_u8(buffer.as_mut_ptr(), buffer.as_ptr(), key.as_ptr());
            assert_eq!(status, 0);
        }
        for _ in 0..100 {
            let status = _mm_aesencwide128kl_u8(buffer.as_mut_ptr(), buffer.as_ptr(), key.as_ptr());
            assert_eq!(status, 0);
        }

        for elem in buffer {
            assert_eq_m128i(elem, _mm_setzero_si128());
        }
    }

    #[simd_test(enable = "widekl")]
    unsafe fn test_mm_aesencwide256kl_u8() {
        let mut buffer = [_mm_setzero_si128(); 8];
        let key = encodekey256();

        for _ in 0..100 {
            let status = _mm_aesencwide256kl_u8(buffer.as_mut_ptr(), buffer.as_ptr(), key.as_ptr());
            assert_eq!(status, 0);
        }
        for _ in 0..100 {
            let status = _mm_aesdecwide256kl_u8(buffer.as_mut_ptr(), buffer.as_ptr(), key.as_ptr());
            assert_eq!(status, 0);
        }

        for elem in buffer {
            assert_eq_m128i(elem, _mm_setzero_si128());
        }
    }

    #[simd_test(enable = "widekl")]
    unsafe fn test_mm_aesdecwide256kl_u8() {
        let mut buffer = [_mm_setzero_si128(); 8];
        let key = encodekey256();

        for _ in 0..100 {
            let status = _mm_aesdecwide256kl_u8(buffer.as_mut_ptr(), buffer.as_ptr(), key.as_ptr());
            assert_eq!(status, 0);
        }
        for _ in 0..100 {
            let status = _mm_aesencwide256kl_u8(buffer.as_mut_ptr(), buffer.as_ptr(), key.as_ptr());
            assert_eq!(status, 0);
        }

        for elem in buffer {
            assert_eq_m128i(elem, _mm_setzero_si128());
        }
    }
}
