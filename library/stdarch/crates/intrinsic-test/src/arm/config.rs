pub const NOTICE: &str = "\
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from a JSON specification, published under the same license as the
// `intrinsic-test` crate.\n";

pub const PLATFORM_C_FORWARD_DECLARATIONS: &str = r#"
#ifdef __aarch64__
std::ostream& operator<<(std::ostream& os, poly128_t value);
#endif

std::ostream& operator<<(std::ostream& os, float16_t value);
std::ostream& operator<<(std::ostream& os, uint8_t value);

// T1 is the `To` type, T2 is the `From` type
template<typename T1, typename T2> T1 cast(T2 x) {
  static_assert(sizeof(T1) == sizeof(T2), "sizeof T1 and T2 must be the same");
  T1 ret{};
  memcpy(&ret, &x, sizeof(T1));
  return ret;
}
"#;

pub const PLATFORM_C_DEFINITIONS: &str = r#"
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

std::ostream& operator<<(std::ostream& os, float16_t value) {
    os << static_cast<float>(value);
    return os;
}

std::ostream& operator<<(std::ostream& os, uint8_t value) {
    os << (unsigned int) value;
    return os;
}
"#;

pub const PLATFORM_RUST_DEFINITIONS: &str = "";

pub const PLATFORM_RUST_CFGS: &str = r#"
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_neon_intrinsics))]
#![cfg_attr(target_arch = "arm", feature(stdarch_aarch32_crc32))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_fcma))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_dotprod))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_i8mm))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_sm4))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_ftts))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_aarch64_jscvt))]
#![feature(fmt_helpers_for_derive)]
#![feature(stdarch_neon_f16)]

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
use core_arch::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use core_arch::arch::arm::*;
"#;
