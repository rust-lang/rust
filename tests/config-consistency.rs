#![feature(rustc_private)]

// This test checks that all lints defined in `clippy_config::conf` in `#[lints]`
// attributes exist as Clippy lints.
//
// This test is a no-op if run as part of the compiler test suite
// and will always succeed.

use std::collections::HashSet;

#[test]
fn config_consistency() {
    if option_env!("RUSTC_TEST_SUITE").is_some() {
        return;
    }

    let lint_names: HashSet<String> = clippy_lints::declared_lints::LINTS
        .iter()
        .map(|lint_info| lint_info.lint.name.strip_prefix("clippy::").unwrap().to_lowercase())
        .collect();
    for conf in clippy_config::get_configuration_metadata() {
        for lint in conf.lints {
            assert!(
                lint_names.contains(*lint),
                "Configuration option {} references lint `{lint}` which does not exist",
                conf.name
            );
        }
    }
}
