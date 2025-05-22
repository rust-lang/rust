//@ only-x86_64
//
// gate-test-sse4a_target_feature
// gate-test-powerpc_target_feature
// gate-test-tbm_target_feature
// gate-test-arm_target_feature
// gate-test-hexagon_target_feature
// gate-test-mips_target_feature
// gate-test-wasm_target_feature
// gate-test-adx_target_feature
// gate-test-cmpxchg16b_target_feature
// gate-test-movbe_target_feature
// gate-test-rtm_target_feature
// gate-test-f16c_target_feature
// gate-test-riscv_target_feature
// gate-test-ermsb_target_feature
// gate-test-bpf_target_feature
// gate-test-aarch64_ver_target_feature
// gate-test-aarch64_unstable_target_feature
// gate-test-csky_target_feature
// gate-test-loongarch_target_feature
// gate-test-lahfsahf_target_feature
// gate-test-prfchw_target_feature
// gate-test-s390x_target_feature
// gate-test-sparc_target_feature
// gate-test-x87_target_feature
// gate-test-m68k_target_feature

#[target_feature(enable = "x87")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
