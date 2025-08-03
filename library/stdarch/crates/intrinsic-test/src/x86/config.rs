pub fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from an XML specification, published under the same license as the
{line_prefix}`intrinsic-test` crate.\n
"
    )
}

// Format f16 values (and vectors containing them) in a way that is consistent with C.
pub const F16_FORMATTING_DEF: &str = r#"
#[repr(transparent)]
struct Hex<T>(T);
 "#;

pub const X86_CONFIGURATIONS: &str = r#"
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx512_bf16))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx512_f16))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_rtm))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_rtm))]
#![cfg_attr(target_arch = "x86_64", feature(x86_amx_intrinsics))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512_f16))]
#![feature(fmt_helpers_for_derive)]
"#;
