//! Check that `cfg!(target_feature = "...")` correctly detects available CPU features,
//! specifically `sse2` on x86/x86_64 platforms, and correctly reports absent features.

//@ run-pass

#![allow(stable_features)]
#![feature(cfg_target_feature)]

use std::env;

fn main() {
    match env::var("TARGET") {
        Ok(s) => {
            // Skip this tests on i586-unknown-linux-gnu where sse2 is disabled
            if s.contains("i586") {
                return;
            }
        }
        Err(_) => return,
    }
    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        assert!(
            cfg!(target_feature = "sse2"),
            "SSE2 was not detected as available on an x86 platform"
        );
    }
    // check a negative case too -- certainly not enabled by default
    #[expect(unexpected_cfgs)]
    {
        assert!(
            cfg!(not(target_feature = "ferris_wheel")),
            "🎡 shouldn't be detected as available by default on any platform"
        )
    };
}
