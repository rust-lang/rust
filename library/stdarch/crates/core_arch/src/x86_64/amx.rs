#[cfg(test)]
use stdarch_test::assert_instr;

/// Load tile configuration from a 64-byte memory location specified by mem_addr.
/// The tile configuration format is specified below, and includes the tile type pallette,
/// the number of bytes per row, and the number of rows. If the specified pallette_id is zero,
/// that signifies the init state for both the tile config and the tile data, and the tiles are zeroed.
/// Any invalid configurations will result in #GP fault.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_loadconfig&ig_expand=6875)
#[inline]
#[target_feature(enable = "amx-tile")]
#[cfg_attr(test, assert_instr(ldtilecfg))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_loadconfig(mem_addr: *const u8) {
    ldtilecfg(mem_addr);
}

/// Stores the current tile configuration to a 64-byte memory location specified by mem_addr.
/// The tile configuration format is specified below, and includes the tile type pallette,
/// the number of bytes per row, and the number of rows. If tiles are not configured, all zeroes will be stored to memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_storeconfig&ig_expand=6879)
#[inline]
#[target_feature(enable = "amx-tile")]
#[cfg_attr(test, assert_instr(sttilecfg))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_storeconfig(mem_addr: *mut u8) {
    sttilecfg(mem_addr);
}

/// Load tile rows from memory specifieid by base address and stride into destination tile dst using the tile configuration previously configured via _tile_loadconfig.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_loadd&ig_expand=6877)
#[inline]
#[rustc_legacy_const_generics(0)]
#[target_feature(enable = "amx-tile")]
#[cfg_attr(test, assert_instr(tileloadd, DST = 0))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_loadd<const DST: i32>(base: *const u8, stride: usize) {
    static_assert_uimm_bits!(DST, 3);
    tileloadd64(DST as i8, base, stride);
}

/// Release the tile configuration to return to the init state, which releases all storage it currently holds.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_release&ig_expand=6878)
#[inline]
#[target_feature(enable = "amx-tile")]
#[cfg_attr(test, assert_instr(tilerelease))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_release() {
    tilerelease();
}

/// Store the tile specified by src to memory specifieid by base address and stride using the tile configuration previously configured via _tile_loadconfig.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_stored&ig_expand=6881)
#[inline]
#[rustc_legacy_const_generics(0)]
#[target_feature(enable = "amx-tile")]
#[cfg_attr(test, assert_instr(tilestored, DST = 0))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_stored<const DST: i32>(base: *mut u8, stride: usize) {
    static_assert_uimm_bits!(DST, 3);
    tilestored64(DST as i8, base, stride);
}

/// Load tile rows from memory specifieid by base address and stride into destination tile dst using the tile configuration
/// previously configured via _tile_loadconfig. This intrinsic provides a hint to the implementation that the data will
/// likely not be reused in the near future and the data caching can be optimized accordingly.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_stream_loadd&ig_expand=6883)
#[inline]
#[rustc_legacy_const_generics(0)]
#[target_feature(enable = "amx-tile")]
#[cfg_attr(test, assert_instr(tileloaddt1, DST = 0))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_stream_loadd<const DST: i32>(base: *const u8, stride: usize) {
    static_assert_uimm_bits!(DST, 3);
    tileloaddt164(DST as i8, base, stride);
}

/// Zero the tile specified by tdest.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_zero&ig_expand=6885)
#[inline]
#[rustc_legacy_const_generics(0)]
#[target_feature(enable = "amx-tile")]
#[cfg_attr(test, assert_instr(tilezero, DST = 0))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_zero<const DST: i32>() {
    static_assert_uimm_bits!(DST, 3);
    tilezero(DST as i8);
}

/// Compute dot-product of BF16 (16-bit) floating-point pairs in tiles a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
/// with elements in dst, and store the 32-bit result back to tile dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_dpbf16ps&ig_expand=6864)
#[inline]
#[rustc_legacy_const_generics(0, 1, 2)]
#[target_feature(enable = "amx-bf16")]
#[cfg_attr(test, assert_instr(tdpbf16ps, DST = 0, A = 1, B = 2))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_dpbf16ps<const DST: i32, const A: i32, const B: i32>() {
    static_assert_uimm_bits!(DST, 3);
    static_assert_uimm_bits!(A, 3);
    static_assert_uimm_bits!(B, 3);
    tdpbf16ps(DST as i8, A as i8, B as i8);
}

/// Compute dot-product of bytes in tiles with a source/destination accumulator.
/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding
/// signed 8-bit integers in b, producing 4 intermediate 32-bit results.
/// Sum these 4 results with the corresponding 32-bit integer in dst, and store the 32-bit result back to tile dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_dpbssd&ig_expand=6866)
#[inline]
#[rustc_legacy_const_generics(0, 1, 2)]
#[target_feature(enable = "amx-int8")]
#[cfg_attr(test, assert_instr(tdpbssd, DST = 0, A = 1, B = 2))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_dpbssd<const DST: i32, const A: i32, const B: i32>() {
    static_assert_uimm_bits!(DST, 3);
    static_assert_uimm_bits!(A, 3);
    static_assert_uimm_bits!(B, 3);
    tdpbssd(DST as i8, A as i8, B as i8);
}

/// Compute dot-product of bytes in tiles with a source/destination accumulator.
/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding
/// unsigned 8-bit integers in b, producing 4 intermediate 32-bit results.
/// Sum these 4 results with the corresponding 32-bit integer in dst, and store the 32-bit result back to tile dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_dpbsud&ig_expand=6868)
#[inline]
#[rustc_legacy_const_generics(0, 1, 2)]
#[target_feature(enable = "amx-int8")]
#[cfg_attr(test, assert_instr(tdpbsud, DST = 0, A = 1, B = 2))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_dpbsud<const DST: i32, const A: i32, const B: i32>() {
    static_assert_uimm_bits!(DST, 3);
    static_assert_uimm_bits!(A, 3);
    static_assert_uimm_bits!(B, 3);
    tdpbsud(DST as i8, A as i8, B as i8);
}

/// Compute dot-product of bytes in tiles with a source/destination accumulator.
/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding
/// signed 8-bit integers in b, producing 4 intermediate 32-bit results.
/// Sum these 4 results with the corresponding 32-bit integer in dst, and store the 32-bit result back to tile dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_dpbusd&ig_expand=6870)
#[inline]
#[rustc_legacy_const_generics(0, 1, 2)]
#[target_feature(enable = "amx-int8")]
#[cfg_attr(test, assert_instr(tdpbusd, DST = 0, A = 1, B = 2))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_dpbusd<const DST: i32, const A: i32, const B: i32>() {
    static_assert_uimm_bits!(DST, 3);
    static_assert_uimm_bits!(A, 3);
    static_assert_uimm_bits!(B, 3);
    tdpbusd(DST as i8, A as i8, B as i8);
}

/// Compute dot-product of bytes in tiles with a source/destination accumulator.
/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding
/// unsigned 8-bit integers in b, producing 4 intermediate 32-bit results.
/// Sum these 4 results with the corresponding 32-bit integer in dst, and store the 32-bit result back to tile dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_dpbuud&ig_expand=6872)
#[inline]
#[rustc_legacy_const_generics(0, 1, 2)]
#[target_feature(enable = "amx-int8")]
#[cfg_attr(test, assert_instr(tdpbuud, DST = 0, A = 1, B = 2))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_dpbuud<const DST: i32, const A: i32, const B: i32>() {
    static_assert_uimm_bits!(DST, 3);
    static_assert_uimm_bits!(A, 3);
    static_assert_uimm_bits!(B, 3);
    tdpbuud(DST as i8, A as i8, B as i8);
}

/// Compute dot-product of FP16 (16-bit) floating-point pairs in tiles a and b,
/// accumulating the intermediate single-precision (32-bit) floating-point elements
///  with elements in dst, and store the 32-bit result back to tile dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_dpfp16ps&ig_expand=6874)
#[inline]
#[rustc_legacy_const_generics(0, 1, 2)]
#[target_feature(enable = "amx-fp16")]
#[cfg_attr(test, assert_instr(tdpfp16ps, DST = 0, A = 1, B = 2))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_dpfp16ps<const DST: i32, const A: i32, const B: i32>() {
    static_assert_uimm_bits!(DST, 3);
    static_assert_uimm_bits!(A, 3);
    static_assert_uimm_bits!(B, 3);
    tdpfp16ps(DST as i8, A as i8, B as i8);
}

/// Perform matrix multiplication of two tiles containing complex elements and accumulate the results into a packed single precision tile.
/// Each dword element in input tiles a and b is interpreted as a complex number with FP16 real part and FP16 imaginary part.
/// Calculates the imaginary part of the result. For each possible combination of (row of a, column of b),
/// it performs a set of multiplication and accumulations on all corresponding complex numbers (one from a and one from b).
/// The imaginary part of the a element is multiplied with the real part of the corresponding b element, and the real part of
/// the a element is multiplied with the imaginary part of the corresponding b elements. The two accumulated results are added,
/// and then accumulated into the corresponding row and column of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_cmmimfp16ps&ig_expand=6860)
#[inline]
#[rustc_legacy_const_generics(0, 1, 2)]
#[target_feature(enable = "amx-complex")]
#[cfg_attr(test, assert_instr(tcmmimfp16ps, DST = 0, A = 1, B = 2))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_cmmimfp16ps<const DST: i32, const A: i32, const B: i32>() {
    static_assert_uimm_bits!(DST, 3);
    static_assert_uimm_bits!(A, 3);
    static_assert_uimm_bits!(B, 3);
    tcmmimfp16ps(DST as i8, A as i8, B as i8);
}

/// Perform matrix multiplication of two tiles containing complex elements and accumulate the results into a packed single precision tile.
/// Each dword element in input tiles a and b is interpreted as a complex number with FP16 real part and FP16 imaginary part.
/// Calculates the real part of the result. For each possible combination of (row of a, column of b),
/// it performs a set of multiplication and accumulations on all corresponding complex numbers (one from a and one from b).
/// The real part of the a element is multiplied with the real part of the corresponding b element, and the negated imaginary part of
/// the a element is multiplied with the imaginary part of the corresponding b elements.
/// The two accumulated results are added, and then accumulated into the corresponding row and column of dst.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_tile_cmmrlfp16ps&ig_expand=6862)
#[inline]
#[rustc_legacy_const_generics(0, 1, 2)]
#[target_feature(enable = "amx-complex")]
#[cfg_attr(test, assert_instr(tcmmrlfp16ps, DST = 0, A = 1, B = 2))]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub unsafe fn _tile_cmmrlfp16ps<const DST: i32, const A: i32, const B: i32>() {
    static_assert_uimm_bits!(DST, 3);
    static_assert_uimm_bits!(A, 3);
    static_assert_uimm_bits!(B, 3);
    tcmmrlfp16ps(DST as i8, A as i8, B as i8);
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.ldtilecfg"]
    fn ldtilecfg(mem_addr: *const u8);
    #[link_name = "llvm.x86.sttilecfg"]
    fn sttilecfg(mem_addr: *mut u8);
    #[link_name = "llvm.x86.tileloadd64"]
    fn tileloadd64(dst: i8, base: *const u8, stride: usize);
    #[link_name = "llvm.x86.tileloaddt164"]
    fn tileloaddt164(dst: i8, base: *const u8, stride: usize);
    #[link_name = "llvm.x86.tilerelease"]
    fn tilerelease();
    #[link_name = "llvm.x86.tilestored64"]
    fn tilestored64(dst: i8, base: *mut u8, stride: usize);
    #[link_name = "llvm.x86.tilezero"]
    fn tilezero(dst: i8);
    #[link_name = "llvm.x86.tdpbf16ps"]
    fn tdpbf16ps(dst: i8, a: i8, b: i8);
    #[link_name = "llvm.x86.tdpbuud"]
    fn tdpbuud(dst: i8, a: i8, b: i8);
    #[link_name = "llvm.x86.tdpbusd"]
    fn tdpbusd(dst: i8, a: i8, b: i8);
    #[link_name = "llvm.x86.tdpbsud"]
    fn tdpbsud(dst: i8, a: i8, b: i8);
    #[link_name = "llvm.x86.tdpbssd"]
    fn tdpbssd(dst: i8, a: i8, b: i8);
    #[link_name = "llvm.x86.tdpfp16ps"]
    fn tdpfp16ps(dst: i8, a: i8, b: i8);
    #[link_name = "llvm.x86.tcmmimfp16ps"]
    fn tcmmimfp16ps(dst: i8, a: i8, b: i8);
    #[link_name = "llvm.x86.tcmmrlfp16ps"]
    fn tcmmrlfp16ps(dst: i8, a: i8, b: i8);
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::_mm_cvtness_sbh;
    use crate::core_arch::x86_64::*;
    use core::mem::transmute;
    use stdarch_test::simd_test;
    #[cfg(target_os = "linux")]
    use syscalls::{Sysno, syscall};

    #[allow(non_camel_case_types)]
    #[repr(packed)]
    #[derive(Copy, Clone, Default, Debug, PartialEq)]
    struct __tilecfg {
        /// 0 `or` 1
        palette: u8,
        start_row: u8,
        /// reserved, must be zero
        reserved_a0: [u8; 14],
        /// number of bytes of one row in each tile
        colsb: [u16; 8],
        /// reserved, must be zero
        reserved_b0: [u16; 8],
        /// number of rows in each tile
        rows: [u8; 8],
        /// reserved, must be zero
        reserved_c0: [u8; 8],
    }

    impl __tilecfg {
        fn new(palette: u8, start_row: u8, colsb: [u16; 8], rows: [u8; 8]) -> Self {
            Self {
                palette,
                start_row,
                reserved_a0: [0u8; 14],
                colsb,
                reserved_b0: [0u16; 8],
                rows,
                reserved_c0: [0u8; 8],
            }
        }

        const fn as_ptr(&self) -> *const u8 {
            self as *const Self as *const u8
        }

        fn as_mut_ptr(&mut self) -> *mut u8 {
            self as *mut Self as *mut u8
        }
    }

    #[cfg(not(target_os = "linux"))]
    #[target_feature(enable = "amx-tile")]
    fn _init_amx() {}

    #[cfg(target_os = "linux")]
    #[target_feature(enable = "amx-tile")]
    #[inline]
    unsafe fn _init_amx() {
        let mut ret: usize;
        let mut xfeatures: usize = 0;
        ret = syscall!(Sysno::arch_prctl, 0x1022, &mut xfeatures as *mut usize)
            .expect("arch_prctl ARCH_GET_XCOMP_PERM syscall failed");
        if ret != 0 {
            panic!("Failed to get XFEATURES");
        } else {
            match 0b11 & (xfeatures >> 17) {
                0 => panic!("AMX is not available"),
                1 => {
                    ret = syscall!(Sysno::arch_prctl, 0x1023, 18)
                        .expect("arch_prctl ARCH_REQ_XCOMP_PERM syscall failed");
                    if ret != 0 {
                        panic!("Failed to enable AMX");
                    }
                }
                3 => {}
                _ => unreachable!(),
            }
        }
    }

    #[simd_test(enable = "amx-tile")]
    unsafe fn test_tile_loadconfig() {
        let config = __tilecfg::default();
        _tile_loadconfig(config.as_ptr());
        _tile_release();
    }

    #[simd_test(enable = "amx-tile")]
    unsafe fn test_tile_storeconfig() {
        let config = __tilecfg::new(1, 0, [32; 8], [8; 8]);
        _tile_loadconfig(config.as_ptr());
        let mut _config = __tilecfg::default();
        _tile_storeconfig(_config.as_mut_ptr());
        _tile_release();
        assert_eq!(config, _config);
    }

    #[simd_test(enable = "amx-tile")]
    unsafe fn test_tile_zero() {
        _init_amx();
        let mut config = __tilecfg::default();
        config.palette = 1;
        config.colsb[0] = 64;
        config.rows[0] = 16;
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        let mut out = [[1_i8; 64]; 16];
        _tile_stored::<0>(&mut out as *mut [i8; 64] as *mut u8, 64);
        _tile_release();
        assert_eq!(out, [[0; 64]; 16]);
    }

    #[simd_test(enable = "amx-tile")]
    unsafe fn test_tile_stored() {
        _init_amx();
        let mut config = __tilecfg::default();
        config.palette = 1;
        config.colsb[0] = 64;
        config.rows[0] = 16;
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        let mut out = [[1_i8; 64]; 16];
        _tile_stored::<0>(&mut out as *mut [i8; 64] as *mut u8, 64);
        _tile_release();
        assert_eq!(out, [[0; 64]; 16]);
    }

    #[simd_test(enable = "amx-tile")]
    unsafe fn test_tile_loadd() {
        _init_amx();
        let mut config = __tilecfg::default();
        config.palette = 1;
        config.colsb[0] = 64;
        config.rows[0] = 16;
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        let mat = [1_i8; 1024];
        _tile_loadd::<0>(&mat as *const i8 as *const u8, 64);
        let mut out = [[0_i8; 64]; 16];
        _tile_stored::<0>(&mut out as *mut [i8; 64] as *mut u8, 64);
        _tile_release();
        assert_eq!(out, [[1; 64]; 16]);
    }

    #[simd_test(enable = "amx-tile")]
    unsafe fn test_tile_stream_loadd() {
        _init_amx();
        let mut config = __tilecfg::default();
        config.palette = 1;
        config.colsb[0] = 64;
        config.rows[0] = 16;
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        let mat = [1_i8; 1024];
        _tile_stream_loadd::<0>(&mat as *const i8 as *const u8, 64);
        let mut out = [[0_i8; 64]; 16];
        _tile_stored::<0>(&mut out as *mut [i8; 64] as *mut u8, 64);
        _tile_release();
        assert_eq!(out, [[1; 64]; 16]);
    }

    #[simd_test(enable = "amx-tile")]
    unsafe fn test_tile_release() {
        _tile_release();
    }

    #[simd_test(enable = "amx-bf16,avx512f")]
    unsafe fn test_tile_dpbf16ps() {
        _init_amx();
        let bf16_1: u16 = _mm_cvtness_sbh(1.0).to_bits();
        let bf16_2: u16 = _mm_cvtness_sbh(2.0).to_bits();
        let ones: [u8; 1024] = transmute([bf16_1; 512]);
        let twos: [u8; 1024] = transmute([bf16_2; 512]);
        let mut res = [[0f32; 16]; 16];
        let mut config = __tilecfg::default();
        config.palette = 1;
        (0..=2).for_each(|i| {
            config.colsb[i] = 64;
            config.rows[i] = 16;
        });
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        _tile_loadd::<1>(&ones as *const u8, 64);
        _tile_loadd::<2>(&twos as *const u8, 64);
        _tile_dpbf16ps::<0, 1, 2>();
        _tile_stored::<0>(&mut res as *mut [f32; 16] as *mut u8, 64);
        _tile_release();
        assert_eq!(res, [[64f32; 16]; 16]);
    }

    #[simd_test(enable = "amx-int8")]
    unsafe fn test_tile_dpbssd() {
        _init_amx();
        let ones = [-1_i8; 1024];
        let twos = [-2_i8; 1024];
        let mut res = [[0_i32; 16]; 16];
        let mut config = __tilecfg::default();
        config.palette = 1;
        (0..=2).for_each(|i| {
            config.colsb[i] = 64;
            config.rows[i] = 16;
        });
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        _tile_loadd::<1>(&ones as *const i8 as *const u8, 64);
        _tile_loadd::<2>(&twos as *const i8 as *const u8, 64);
        _tile_dpbssd::<0, 1, 2>();
        _tile_stored::<0>(&mut res as *mut [i32; 16] as *mut u8, 64);
        _tile_release();
        assert_eq!(res, [[128_i32; 16]; 16]);
    }

    #[simd_test(enable = "amx-int8")]
    unsafe fn test_tile_dpbsud() {
        _init_amx();
        let ones = [-1_i8; 1024];
        let twos = [2_u8; 1024];
        let mut res = [[0_i32; 16]; 16];
        let mut config = __tilecfg::default();
        config.palette = 1;
        (0..=2).for_each(|i| {
            config.colsb[i] = 64;
            config.rows[i] = 16;
        });
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        _tile_loadd::<1>(&ones as *const i8 as *const u8, 64);
        _tile_loadd::<2>(&twos as *const u8, 64);
        _tile_dpbsud::<0, 1, 2>();
        _tile_stored::<0>(&mut res as *mut [i32; 16] as *mut u8, 64);
        _tile_release();
        assert_eq!(res, [[-128_i32; 16]; 16]);
    }

    #[simd_test(enable = "amx-int8")]
    unsafe fn test_tile_dpbusd() {
        _init_amx();
        let ones = [1_u8; 1024];
        let twos = [-2_i8; 1024];
        let mut res = [[0_i32; 16]; 16];
        let mut config = __tilecfg::default();
        config.palette = 1;
        (0..=2).for_each(|i| {
            config.colsb[i] = 64;
            config.rows[i] = 16;
        });
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        _tile_loadd::<1>(&ones as *const u8, 64);
        _tile_loadd::<2>(&twos as *const i8 as *const u8, 64);
        _tile_dpbusd::<0, 1, 2>();
        _tile_stored::<0>(&mut res as *mut [i32; 16] as *mut u8, 64);
        _tile_release();
        assert_eq!(res, [[-128_i32; 16]; 16]);
    }

    #[simd_test(enable = "amx-int8")]
    unsafe fn test_tile_dpbuud() {
        _init_amx();
        let ones = [1_u8; 1024];
        let twos = [2_u8; 1024];
        let mut res = [[0_i32; 16]; 16];
        let mut config = __tilecfg::default();
        config.palette = 1;
        (0..=2).for_each(|i| {
            config.colsb[i] = 64;
            config.rows[i] = 16;
        });
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        _tile_loadd::<1>(&ones as *const u8, 64);
        _tile_loadd::<2>(&twos as *const u8, 64);
        _tile_dpbuud::<0, 1, 2>();
        _tile_stored::<0>(&mut res as *mut [i32; 16] as *mut u8, 64);
        _tile_release();
        assert_eq!(res, [[128_i32; 16]; 16]);
    }

    #[simd_test(enable = "amx-fp16")]
    unsafe fn test_tile_dpfp16ps() {
        _init_amx();
        let ones = [1f16; 512];
        let twos = [2f16; 512];
        let mut res = [[0f32; 16]; 16];
        let mut config = __tilecfg::default();
        config.palette = 1;
        (0..=2).for_each(|i| {
            config.colsb[i] = 64;
            config.rows[i] = 16;
        });
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        _tile_loadd::<1>(&ones as *const f16 as *const u8, 64);
        _tile_loadd::<2>(&twos as *const f16 as *const u8, 64);
        _tile_dpfp16ps::<0, 1, 2>();
        _tile_stored::<0>(&mut res as *mut [f32; 16] as *mut u8, 64);
        _tile_release();
        assert_eq!(res, [[64f32; 16]; 16]);
    }

    #[simd_test(enable = "amx-complex")]
    unsafe fn test_tile_cmmimfp16ps() {
        _init_amx();
        let ones = [1f16; 512];
        let twos = [2f16; 512];
        let mut res = [[0f32; 16]; 16];
        let mut config = __tilecfg::default();
        config.palette = 1;
        (0..=2).for_each(|i| {
            config.colsb[i] = 64;
            config.rows[i] = 16;
        });
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        _tile_loadd::<1>(&ones as *const f16 as *const u8, 64);
        _tile_loadd::<2>(&twos as *const f16 as *const u8, 64);
        _tile_cmmimfp16ps::<0, 1, 2>();
        _tile_stored::<0>(&mut res as *mut [f32; 16] as *mut u8, 64);
        _tile_release();
        assert_eq!(res, [[64f32; 16]; 16]);
    }

    #[simd_test(enable = "amx-complex")]
    unsafe fn test_tile_cmmrlfp16ps() {
        _init_amx();
        let ones = [1f16; 512];
        let twos = [2f16; 512];
        let mut res = [[0f32; 16]; 16];
        let mut config = __tilecfg::default();
        config.palette = 1;
        (0..=2).for_each(|i| {
            config.colsb[i] = 64;
            config.rows[i] = 16;
        });
        _tile_loadconfig(config.as_ptr());
        _tile_zero::<0>();
        _tile_loadd::<1>(&ones as *const f16 as *const u8, 64);
        _tile_loadd::<2>(&twos as *const f16 as *const u8, 64);
        _tile_cmmrlfp16ps::<0, 1, 2>();
        _tile_stored::<0>(&mut res as *mut [f32; 16] as *mut u8, 64);
        _tile_release();
        assert_eq!(res, [[0f32; 16]; 16]);
    }
}
