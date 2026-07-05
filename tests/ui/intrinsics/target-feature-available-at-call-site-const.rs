#![feature(core_intrinsics)]

const HAS_FMA: bool = std::intrinsics::simd::target_feature_available_at_call_site!("fma");
//~^ ERROR cannot call non-const intrinsic `target_feature_available_at_call_site` in constants

fn main() {
    let _ = HAS_FMA;
}
