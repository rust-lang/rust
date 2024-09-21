use super::UnstableFeatures;

#[test]
fn rustc_bootstrap_parsing() {
    // FIXME(edition_2024, #130672): Audit this for safety. This is probably racey, per:
    // <https://github.com/rust-lang/rust/pull/129636#pullrequestreview-2314766092>.
    let is_bootstrap = |env, krate| {
        unsafe {
            std::env::set_var("RUSTC_BOOTSTRAP", env);
        }
        matches!(UnstableFeatures::from_environment(krate), UnstableFeatures::Cheat)
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

    // this is technically a breaking change, but there are no stability guarantees for RUSTC_BOOTSTRAP
    assert!(!is_bootstrap("0", None));
}
