#[test]
fn simulator_targets_set_abi() {
    let all_sim_targets = [
        super::x86_64_apple_ios::target(),
        super::x86_64_apple_tvos::target(),
        super::x86_64_apple_watchos_sim::target(),
        super::aarch64_apple_ios_sim::target(),
        // Note: There is currently no ARM64 tvOS simulator target
        super::aarch64_apple_watchos_sim::target(),
    ];

    for target in all_sim_targets {
        assert_eq!(target.abi, "sim")
    }
}

#[test]
fn macos_link_environment_unmodified() {
    let all_macos_targets = [
        super::aarch64_apple_darwin::target(),
        super::i686_apple_darwin::target(),
        super::x86_64_apple_darwin::target(),
    ];

    for target in all_macos_targets {
        // macOS targets should only remove information for cross-compiling, but never
        // for the host.
        assert_eq!(
            target.link_env_remove,
            crate::spec::cvs!["IPHONEOS_DEPLOYMENT_TARGET", "TVOS_DEPLOYMENT_TARGET"],
        );
    }
}
