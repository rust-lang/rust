//! Check accepted boolean syntax and errors of feature detection macros.

//@ revisions: AARCH64 X86
//@[AARCH64] only-aarch64
//@[X86] only-x86_64

#![feature(feature_detection_boolean_syntax)]

use std::arch;

fn main() {
    // Correct usage
    #[cfg(target_arch = "aarch64")]
    {
        let _ = arch::is_aarch64_feature_detected!("neon" && "crc");
        let _ = arch::is_aarch64_feature_detected!("neon" || "sve");
        let _ = arch::is_aarch64_feature_detected!(("neon" && "crc") || "sve");
        let _ = arch::is_aarch64_feature_detected!(("neon"));
        let _ = arch::is_aarch64_feature_detected!("neon" && "crc",);
    }
    #[cfg(target_arch = "x86_64")]
    {
        let _ = arch::is_x86_feature_detected!("sse" && "sse2");
        let _ = arch::is_x86_feature_detected!("avx2" || "sse4.2");
        let _ = arch::is_x86_feature_detected!(("avx2" && "bmi2") || "sse4.2");
        let _ = arch::is_x86_feature_detected!(("sse"));
        let _ = arch::is_x86_feature_detected!("sse" && "sse2",);
    }

    // Invalid features get an error regardless of target
    let _ = arch::is_x86_feature_detected!("sse" && "not-a-feature");
    //[AARCH64]~^ ERROR: This macro cannot be used on the current target
    //~^^ ERROR: unknown x86 target feature
    let _ = arch::is_aarch64_feature_detected!("neon" || "not-a-feature");
    //[X86]~^ ERROR: This macro cannot be used on the current target
    //~^^ ERROR: unknown aarch64 target feature

    // Valid features get an error if for the wrong target
    #[cfg(target_arch = "aarch64")]
    let _ = arch::is_x86_feature_detected!("sse" && "sse2");
    //[AARCH64]~^ ERROR: This macro cannot be used on the current target
    //[AARCH64]~| ERROR: This macro cannot be used on the current target
    #[cfg(target_arch = "x86_64")]
    let _ = arch::is_aarch64_feature_detected!("neon" && "sve");
    //[X86]~^ ERROR: This macro cannot be used on the current target
    //[X86]~| ERROR: This macro cannot be used on the current target

    // Check handling of deprecated features
    let _ = arch::is_x86_feature_detected!("sse" && "avx512gfni");
    //[X86]~^ WARN: use of deprecated macro
    //[AARCH64]~^^ ERROR: This macro cannot be used on the current target
    //[AARCH64]~| ERROR: This macro cannot be used on the current target

    // Check handling of features without runtime detection
    let _ = arch::is_aarch64_feature_detected!("neon" && "ras");
    //[X86]~^ ERROR: This macro cannot be used on the current target
    //~^^ ERROR: feature cannot be detected at run-time

    // Check gating of unstable features inside boolean expressions
    #[cfg(target_arch = "aarch64")]
    let _ = arch::is_aarch64_feature_detected!("neon" && "wfxt");
    //[AARCH64]~^ ERROR: use of unstable library feature `stdarch_aarch64_feature_detection`
    #[cfg(target_arch = "x86_64")]
    let _ = arch::is_x86_feature_detected!("sse" && "xop");
    //[X86]~^ ERROR: use of unstable library feature `xop_target_feature`

    // Check invalid boolean operator syntax
    let _ = arch::is_x86_feature_detected!("sse", "sse2");
    //~^ ERROR: expected `&&` or `||` between target features
    let _ = arch::is_x86_feature_detected!("sse" &&);
    //~^ ERROR: expected target feature after `&&`
    let _ = arch::is_x86_feature_detected!(&&"sse");
    //~^ ERROR: expected target feature before `&&`
    let _ = arch::is_x86_feature_detected!(());
    //~^ ERROR: expected target feature in parentheses
}
