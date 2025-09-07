pub const NOTICE: &str = "\
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from a JSON specification, published under the same license as the
// `intrinsic-test` crate.\n";

pub const POLY128_OSTREAM_DECL: &str = r#"
#ifdef __aarch64__
std::ostream& operator<<(std::ostream& os, poly128_t value);
#endif
"#;

pub const POLY128_OSTREAM_DEF: &str = r#"
#ifdef __aarch64__
std::ostream& operator<<(std::ostream& os, poly128_t value) {
    std::stringstream temp;
    do {
      int n = value % 10;
      value /= 10;
      temp << n;
    } while (value != 0);
    std::string tempstr(temp.str());
    std::string res(tempstr.rbegin(), tempstr.rend());
    os << res;
    return os;
}
#endif
"#;

// Format f16 values (and vectors containing them) in a way that is consistent with C.
pub const F16_FORMATTING_DEF: &str = r#"
/// Used to continue `Debug`ging SIMD types as `MySimd(1, 2, 3, 4)`, as they
/// were before moving to array-based simd.
#[inline]
fn debug_simd_finish<T: core::fmt::Debug, const N: usize>(
    formatter: &mut core::fmt::Formatter<'_>,
    type_name: &str,
    array: &[T; N],
) -> core::fmt::Result {
    core::fmt::Formatter::debug_tuple_fields_finish(
        formatter,
        type_name,
        &core::array::from_fn::<&dyn core::fmt::Debug, N, _>(|i| &array[i]),
    )
}

#[repr(transparent)]
struct Hex<T>(T);

impl<T: DebugHexF16> core::fmt::Debug for Hex<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        <T as DebugHexF16>::fmt(&self.0, f)
    }
}

fn debug_f16<T: DebugHexF16>(x: T) -> impl core::fmt::Debug {
    Hex(x)
}

trait DebugHexF16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result;
}

impl DebugHexF16 for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:#06x?}", self.to_bits())
    }
}

impl DebugHexF16 for float16x4_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 4]>(*self) };
        debug_simd_finish(f, "float16x4_t", &array)
    }
}

impl DebugHexF16 for float16x8_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = unsafe { core::mem::transmute::<_, [Hex<f16>; 8]>(*self) };
        debug_simd_finish(f, "float16x8_t", &array)
    }
}

impl DebugHexF16 for float16x4x2_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        debug_simd_finish(f, "float16x4x2_t", &[Hex(self.0), Hex(self.1)])
    }
}
impl DebugHexF16 for float16x4x3_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        debug_simd_finish(f, "float16x4x3_t", &[Hex(self.0), Hex(self.1), Hex(self.2)])
    }
}
impl DebugHexF16 for float16x4x4_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        debug_simd_finish(f, "float16x4x4_t", &[Hex(self.0), Hex(self.1), Hex(self.2), Hex(self.3)])
    }
}

impl DebugHexF16 for float16x8x2_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        debug_simd_finish(f, "float16x8x2_t", &[Hex(self.0), Hex(self.1)])
    }
}
impl DebugHexF16 for float16x8x3_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        debug_simd_finish(f, "float16x8x3_t", &[Hex(self.0), Hex(self.1), Hex(self.2)])
    }
}
impl DebugHexF16 for float16x8x4_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        debug_simd_finish(f, "float16x8x4_t", &[Hex(self.0), Hex(self.1), Hex(self.2), Hex(self.3)])
    }
}
 "#;

pub const AARCH_CONFIGURATIONS: &str = r#"
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_neon_intrinsics))]
#![cfg_attr(target_arch = "arm", feature(stdarch_aarch32_crc32))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_fcma))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_dotprod))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_i8mm))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_sm4))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_ftts))]
#![feature(fmt_helpers_for_derive)]
#![feature(stdarch_neon_f16)]

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
use core::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use core::arch::arm::*;
"#;
