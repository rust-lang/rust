pub fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from a JSON specification, published under the same license as the
{line_prefix}`intrinsic-test` crate.\n
"
    )
}

pub const POLY128_OSTREAM_DEF: &str = 
r#"std::ostream& operator<<(std::ostream& os, poly128_t value) {
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
}"#;

pub const AARCH_CONFIGURATIONS: &str = r#"
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_neon_intrinsics))]
#![cfg_attr(target_arch = "arm", feature(stdarch_aarch32_crc32))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_fcma))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_dotprod))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_i8mm))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_sha3))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_sm4))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_ftts))]
#![feature(stdarch_neon_f16)]
"#;
