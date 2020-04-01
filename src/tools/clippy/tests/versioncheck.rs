#[test]
fn check_that_clippy_lints_has_the_same_version_as_clippy() {
    let clippy_meta = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .expect("could not obtain cargo metadata");
    std::env::set_current_dir(std::env::current_dir().unwrap().join("clippy_lints")).unwrap();
    let clippy_lints_meta = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .expect("could not obtain cargo metadata");
    assert_eq!(clippy_lints_meta.packages[0].version, clippy_meta.packages[0].version);
    for package in &clippy_meta.packages[0].dependencies {
        if package.name == "clippy_lints" {
            assert!(package.req.matches(&clippy_lints_meta.packages[0].version));
            return;
        }
    }
}
