use crate::spec::{
    aarch64_apple_ios_sim, aarch64_apple_watchos_sim, x86_64_apple_ios, x86_64_apple_tvos,
    x86_64_apple_watchos_sim,
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
    ];

    for target in all_sim_targets {
        assert_eq!(target.abi, "sim")
    }
}
