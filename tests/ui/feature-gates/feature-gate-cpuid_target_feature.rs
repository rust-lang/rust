//@ only-x86_64

use std::arch::x86_64::*;

#[target_feature(enable = "cpuid")]
//~^ ERROR: currently unstable
fn annotation() {}

fn detection() -> bool {
    is_x86_feature_detected!("cpuid")
    //~^ ERROR: use of unstable library feature
}

fn intrinsic() -> CpuidResult {
    __cpuid(0)
    //~^ ERROR: requires unsafe function or block
}

fn main() {}
