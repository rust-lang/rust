// Compiler:
//
// Run-time:
//   status: 0

// The CRC32 intrinsics only compile when `#[target_feature(enable = "crc")]` reaches GCC as
// `target("+crc")`; a wrong lowering returns a different checksum.

#[cfg(target_arch = "aarch64")]
mod crc32 {
    use std::arch::aarch64::*;

    const IEEE_POLYNOMIAL: u64 = 0xEDB8_8320;
    const CASTAGNOLI_POLYNOMIAL: u64 = 0x82F6_3B78;

    fn software_crc32(crc: u32, data: u64, bits: u32, polynomial: u64) -> u32 {
        let mut state = crc as u64 ^ data;
        for _ in 0..bits {
            state = if state & 1 == 1 { (state >> 1) ^ polynomial } else { state >> 1 };
        }
        state as u32
    }

    #[target_feature(enable = "crc")]
    fn check(crc: u32, data: u64) {
        for (polynomial, byte, half, word, double) in [
            (
                IEEE_POLYNOMIAL,
                __crc32b(crc, data as u8),
                __crc32h(crc, data as u16),
                __crc32w(crc, data as u32),
                __crc32d(crc, data),
            ),
            (
                CASTAGNOLI_POLYNOMIAL,
                __crc32cb(crc, data as u8),
                __crc32ch(crc, data as u16),
                __crc32cw(crc, data as u32),
                __crc32cd(crc, data),
            ),
        ] {
            assert_eq!(byte, software_crc32(crc, data as u8 as u64, 8, polynomial));
            assert_eq!(half, software_crc32(crc, data as u16 as u64, 16, polynomial));
            assert_eq!(word, software_crc32(crc, data as u32 as u64, 32, polynomial));
            assert_eq!(double, software_crc32(crc, data, 64, polynomial));
        }
    }

    #[target_feature(enable = "crc")]
    fn check_values() {
        let input = b"123456789";
        let ieee = !input.iter().fold(!0, |crc, &byte| __crc32b(crc, byte));
        let castagnoli = !input.iter().fold(!0, |crc, &byte| __crc32cb(crc, byte));
        assert_eq!(ieee, 0xCBF4_3926);
        assert_eq!(castagnoli, 0xE306_9283);
    }

    pub fn run() {
        assert!(
            std::arch::is_aarch64_feature_detected!("crc"),
            "this test needs a CPU with the `crc` feature"
        );
        let inputs = [
            (0, 0),
            (!0, !0),
            (0x1234_5678, 0x0123_4567_89AB_CDEF),
            (0xDEAD_BEEF, 0xFEDC_BA98_7654_3210),
            (0x8000_0001, 0x8000_0000_0000_0001),
        ];
        for (crc, data) in inputs {
            unsafe { check(crc, data) };
        }
        unsafe { check_values() };
    }
}

fn main() {
    #[cfg(target_arch = "aarch64")]
    crc32::run();
}
