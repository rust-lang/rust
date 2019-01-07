use semver::VersionReq;

#[test]
fn check_that_clippy_lints_has_the_same_version_as_clippy() {
    let clippy_meta = cargo_metadata::metadata(None).expect("could not obtain cargo metadata");
    std::env::set_current_dir(std::env::current_dir().unwrap().join("clippy_lints")).unwrap();
    let clippy_lints_meta = cargo_metadata::metadata(None).expect("could not obtain cargo metadata");
    assert_eq!(clippy_lints_meta.packages[0].version, clippy_meta.packages[0].version);
    for package in &clippy_meta.packages[0].dependencies {
        if package.name == "clippy_lints" {
            assert_eq!(
                VersionReq::parse(&clippy_lints_meta.packages[0].version).unwrap(),
                package.req
            );
            return;
        }
    }
}
