use crate::spec::{aarch64_apple_ios_sim, aarch64_apple_watchos_sim, x86_64_apple_ios};

#[test]
fn simulator_targets_set_abi() {
    let all_sim_targets = [
        x86_64_apple_ios::target(),
        aarch64_apple_ios_sim::target(),
        aarch64_apple_watchos_sim::target(),
        // TODO: x86_64-apple-tvos and x86_64-apple-watchos-sim
    ];

    for target in all_sim_targets {
        assert_eq!(target.abi, "sim")
    }
}
