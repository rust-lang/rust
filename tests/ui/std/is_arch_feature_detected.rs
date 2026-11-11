//! Check accepted syntax and errors of feature detection macros.

// Do separate revisions so we can always check another arch's macros
//@ revisions: AARCH64 X86
//@[AARCH64] only-aarch64
//@[X86] only-x86_64

use std::arch;

fn main() {
    // Correct uage
    #[cfg(target_arch = "aarch64")]
    let _ = arch::is_aarch64_feature_detected!("neon");
    #[cfg(target_arch = "x86_64")]
    let _ = arch::is_x86_feature_detected!("sse");

    // Invalid features get an error regardless of target
    let _ = arch::is_x86_feature_detected!("not-a-feature");
    //~^ ERROR: unknown x86 target feature
    let _ = arch::is_aarch64_feature_detected!("not-a-feature");
    //~^ ERROR: unknown aarch64 target feature

    // Valid features get an error if for the wrong target
    #[cfg(target_arch = "aarch64")]
    let _ = arch::is_x86_feature_detected!("sse");
    //[AARCH64]~^ ERROR: This macro cannot be used on the current target
    #[cfg(target_arch = "x86_64")]
    let _ = arch::is_aarch64_feature_detected!("neon");
    //[X86]~^ ERROR: This macro cannot be used on the current target

    // Check handling of deprecated features
    let _ = arch::is_x86_feature_detected!("avx512gfni");
    //[X86]~^ WARN: use of deprecated macro
    //[AARCH64]~^^ ERROR: This macro cannot be used on the current target

    // Check handling of features without runtime detection
    let _ = arch::is_aarch64_feature_detected!("ras");
    //~^ ERROR: feature cannot be detected at run-time

    // Check gating of the unstable macro.
    let _ = arch::is_mips_feature_detected!("msa");
    //~^ ERROR: use of unstable library feature `stdarch_mips_feature_detection`
    //~| ERROR: This macro cannot be used on the current target

    // Check gating of unstable features in stable macros. Note that if this moves too much, it
    // may be worth adding a `_demo-unstable` feature so we can test consistently.
    #[cfg(target_arch = "aarch64")]
    let _ = arch::is_aarch64_feature_detected!("wfxt");
    //[AARCH64]~^ ERROR: use of unstable library feature `stdarch_aarch64_feature_detection`
    #[cfg(target_arch = "x86_64")]
    let _ = arch::is_x86_feature_detected!("xop");
    //[X86]~^ ERROR: use of unstable library feature `xop_target_feature`
}
