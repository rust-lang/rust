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
    let parse_llvm = |s| parse(s).llvm_from_ci;
    let if_available = parse_llvm("llvm.download-ci-llvm = \"if-available\"");

    assert!(parse_llvm("llvm.download-ci-llvm = true"));
    assert!(!parse_llvm("llvm.download-ci-llvm = false"));
    assert_eq!(parse_llvm(""), if_available);
    assert_eq!(parse_llvm("rust.channel = \"dev\""), if_available);
    assert!(!parse_llvm("rust.channel = \"stable\""));
}

// FIXME: add test for detecting `src` and `out`
