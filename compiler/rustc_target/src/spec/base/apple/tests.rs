use super::parse_version;
use crate::spec::targets::{
    aarch64_apple_darwin, aarch64_apple_ios_sim, aarch64_apple_visionos_sim,
    aarch64_apple_watchos_sim, i686_apple_darwin, x86_64_apple_darwin, x86_64_apple_ios,
    x86_64_apple_tvos, x86_64_apple_watchos_sim,
};

#[test]
fn simulator_targets_set_abi() {
    let all_sim_targets = [
        x86_64_apple_ios::target(),
        x86_64_apple_tvos::target(),
        x86_64_apple_watchos_sim::target(),
        aarch64_apple_ios_sim::target(),
        // Note: There is currently no ARM64 tvOS simulator target
        aarch64_apple_watchos_sim::target(),
        aarch64_apple_visionos_sim::target(),
    ];

    for target in &all_sim_targets {
        assert_eq!(target.abi, "sim")
    }
}

#[test]
fn macos_link_environment_unmodified() {
    let all_macos_targets = [
        aarch64_apple_darwin::target(),
        i686_apple_darwin::target(),
        x86_64_apple_darwin::target(),
    ];

    for target in all_macos_targets {
        // macOS targets should only remove information for cross-compiling, but never
        // for the host.
        assert_eq!(target.link_env_remove, crate::spec::cvs![
            "IPHONEOS_DEPLOYMENT_TARGET",
            "TVOS_DEPLOYMENT_TARGET",
            "XROS_DEPLOYMENT_TARGET"
        ],);
    }
}

#[test]
fn test_parse_version() {
    assert_eq!(parse_version("10"), Ok((10, 0, 0)));
    assert_eq!(parse_version("10.12"), Ok((10, 12, 0)));
    assert_eq!(parse_version("10.12.6"), Ok((10, 12, 6)));
    assert_eq!(parse_version("9999.99.99"), Ok((9999, 99, 99)));
}
