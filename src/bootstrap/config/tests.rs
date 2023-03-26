use super::{Config, TomlConfig};
use std::{env, path::Path};

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

#[test]
fn detect_src_and_out() {
    let cfg = parse("");

    // This will bring absolute form of `src/bootstrap` path
    let current_dir = std::env::current_dir().unwrap();

    // get `src` by moving into project root path
    let expected_src = current_dir.ancestors().nth(2).unwrap();

    assert_eq!(&cfg.src, expected_src);

    // This should bring output path of bootstrap in absolute form
    let cargo_target_dir = env::var_os("CARGO_TARGET_DIR")
        .expect("CARGO_TARGET_DIR must been provided for the test environment from bootstrap");

    // Move to `build` from `build/bootstrap`
    let expected_out = Path::new(&cargo_target_dir).parent().unwrap();
    assert_eq!(&cfg.out, expected_out);

    let args: Vec<String> = env::args().collect();

    // Another test for `out` as a sanity check
    //
    // This will bring something similar to:
    //     `{config_toml_place}/build/bootstrap/debug/deps/bootstrap-c7ee91d5661e2804`
    // `{config_toml_place}` can be anywhere, not just in the rust project directory.
    let dep = Path::new(args.first().unwrap());
    let expected_out = dep.ancestors().nth(4).unwrap();

    assert_eq!(&cfg.out, expected_out);
}
