// ignore-arm
// ignore-aarch64
// ignore-wasm
// ignore-emscripten
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-riscv64
// ignore-sparc
// ignore-sparc64
// ignore-s390x
// gate-test-sse4a_target_feature
// gate-test-powerpc_target_feature
// gate-test-avx512_target_feature
// gate-test-tbm_target_feature
// gate-test-arm_target_feature
// gate-test-aarch64_target_feature
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

#[target_feature(enable = "avx512bw")]
//~^ ERROR: currently unstable
unsafe fn foo() {}

fn main() {}
