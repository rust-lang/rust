use super::UnstableFeatures;

#[test]
fn rustc_bootstrap_parsing() {
    let is_bootstrap = |env: &str, krate: Option<&str>| {
        matches!(
            UnstableFeatures::from_environment_value(krate, Ok(env.to_string())),
            UnstableFeatures::Cheat
        )
    };
    assert!(is_bootstrap("1", None));
    assert!(is_bootstrap("1", Some("x")));
    // RUSTC_BOOTSTRAP allows specifying a specific crate
    assert!(is_bootstrap("x", Some("x")));
    // RUSTC_BOOTSTRAP allows multiple comma-delimited crates
    assert!(is_bootstrap("x,y,z", Some("x")));
    assert!(is_bootstrap("x,y,z", Some("y")));
    // Crate that aren't specified do not get unstable features
    assert!(!is_bootstrap("x", Some("a")));
    assert!(!is_bootstrap("x,y,z", Some("a")));
    assert!(!is_bootstrap("x,y,z", None));

    // `RUSTC_BOOTSTRAP=0` is not recognized.
    assert!(!is_bootstrap("0", None));

    // `RUSTC_BOOTSTRAP=-1` is force-stable, no unstable features allowed.
    let is_force_stable = |krate: Option<&str>| {
        matches!(
            UnstableFeatures::from_environment_value(krate, Ok("-1".to_string())),
            UnstableFeatures::Disallow
        )
    };
    assert!(is_force_stable(None));
    // Does not support specifying any crate.
    assert!(is_force_stable(Some("x")));
    assert!(is_force_stable(Some("x,y,z")));

    // `RUSTC_BOOTSTRAP=-allow-unstable-compiler-flags` simulates the stable toolchain, only
    // allowing the specified unstable compiler flags to be passed.
    std::assert_eq!(
        UnstableFeatures::from_environment_value(
            None,
            Ok("-allow-unstable-compiler-flags=foo,bar".to_string())
        ),
        UnstableFeatures::AllowSelectedUnstableCompilerFlagsOnly {
            allowed_flags: vec!["foo".to_string(), "bar".to_string()]
        }
    );
}
