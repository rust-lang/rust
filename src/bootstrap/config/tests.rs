use super::{Config, TomlConfig};
use std::path::Path;

fn toml(config: &str) -> impl '_ + Fn(&Path) -> TomlConfig {
    |&_| toml::from_str(config).unwrap()
}

fn parse(config: &str) -> Config {
    Config::parse_inner(&["check".to_owned(), "--config=/does/not/exist".to_owned()], toml(config))
}

#[test]
fn download_ci_llvm() {
    if crate::native::is_ci_llvm_modified(&parse("")) {
        eprintln!("Detected LLVM as non-available: running in CI and modified LLVM in this change");
        return;
    }

    let parse_llvm = |s| parse(s).llvm_from_ci;
    let if_available = parse_llvm("llvm.download-ci-llvm = \"if-available\"");

    assert!(parse_llvm("llvm.download-ci-llvm = true"));
    assert!(!parse_llvm("llvm.download-ci-llvm = false"));
    assert_eq!(parse_llvm(""), if_available);
    assert_eq!(parse_llvm("rust.channel = \"dev\""), if_available);
    assert!(!parse_llvm("rust.channel = \"stable\""));
    assert!(parse_llvm("build.build = \"x86_64-unknown-linux-gnu\""));
    assert!(parse_llvm(
        "llvm.assertions = true \r\n build.build = \"x86_64-unknown-linux-gnu\" \r\n llvm.download-ci-llvm = \"if-available\""
    ));
    assert!(!parse_llvm(
        "llvm.assertions = true \r\n build.build = \"aarch64-apple-darwin\" \r\n llvm.download-ci-llvm = \"if-available\""
    ));
}

// FIXME: add test for detecting `src` and `out`
