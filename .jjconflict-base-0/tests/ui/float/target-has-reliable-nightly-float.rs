//@ run-pass
//@ compile-flags: --check-cfg=cfg(target_has_reliable_f16,target_has_reliable_f16_math,target_has_reliable_f128,target_has_reliable_f128_math)
// Verify that the feature gates and config work and are registered as known config
// options.

#![deny(unexpected_cfgs)]
#![feature(cfg_target_has_reliable_f16_f128)]

#[cfg(target_has_reliable_f16)]
pub fn has_f16() {}

#[cfg(target_has_reliable_f16_math)]
pub fn has_f16_math() {}

#[cfg(target_has_reliable_f128 )]
pub fn has_f128() {}

#[cfg(target_has_reliable_f128_math)]
pub fn has_f128_math() {}

fn main() {
    if cfg!(target_arch = "aarch64") &&
        cfg!(target_os = "linux") &&
        cfg!(not(target_env = "musl")) {
        // Aarch64+GNU+Linux is one target that has support for all features, so use it to spot
        // check that the compiler does indeed enable these gates.

        assert!(cfg!(target_has_reliable_f16));
        assert!(cfg!(target_has_reliable_f16_math));
        assert!(cfg!(target_has_reliable_f128));
        assert!(cfg!(target_has_reliable_f128_math));
    }
}
