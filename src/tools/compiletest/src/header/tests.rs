use std::path::Path;

use crate::common::{Config, Debugger};
use crate::header::{parse_normalization_string, EarlyProps};

#[test]
fn test_parse_normalization_string() {
    let mut s = "normalize-stderr-32bit: \"something (32 bits)\" -> \"something ($WORD bits)\".";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, Some("something (32 bits)".to_owned()));
    assert_eq!(s, " -> \"something ($WORD bits)\".");

    // Nothing to normalize (No quotes)
    let mut s = "normalize-stderr-32bit: something (32 bits) -> something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, None);
    assert_eq!(s, r#"normalize-stderr-32bit: something (32 bits) -> something ($WORD bits)."#);

    // Nothing to normalize (Only a single quote)
    let mut s = "normalize-stderr-32bit: \"something (32 bits) -> something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, None);
    assert_eq!(s, "normalize-stderr-32bit: \"something (32 bits) -> something ($WORD bits).");

    // Nothing to normalize (Three quotes)
    let mut s = "normalize-stderr-32bit: \"something (32 bits)\" -> \"something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, Some("something (32 bits)".to_owned()));
    assert_eq!(s, " -> \"something ($WORD bits).");

    // Nothing to normalize (No quotes, 16-bit)
    let mut s = "normalize-stderr-16bit: something (16 bits) -> something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, None);
    assert_eq!(s, r#"normalize-stderr-16bit: something (16 bits) -> something ($WORD bits)."#);
}

fn config() -> Config {
    let args = &[
        "compiletest",
        "--mode=ui",
        "--suite=ui",
        "--compile-lib-path=",
        "--run-lib-path=",
        "--rustc-path=",
        "--lldb-python=",
        "--docck-python=",
        "--src-base=",
        "--build-base=",
        "--stage-id=stage2",
        "--cc=c",
        "--cxx=c++",
        "--cflags=",
        "--llvm-components=",
        "--android-cross-path=",
        "--target=x86_64-unknown-linux-gnu",
    ];
    let args = args.iter().map(ToString::to_string).collect();
    crate::parse_config(args)
}

fn parse_rs(config: &Config, contents: &str) -> EarlyProps {
    let bytes = contents.as_bytes();
    EarlyProps::from_reader(config, Path::new("a.rs"), bytes)
}

fn parse_makefile(config: &Config, contents: &str) -> EarlyProps {
    let bytes = contents.as_bytes();
    EarlyProps::from_reader(config, Path::new("Makefile"), bytes)
}

#[test]
fn should_fail() {
    let config = config();

    assert!(!parse_rs(&config, "").should_fail);
    assert!(parse_rs(&config, "// should-fail").should_fail);
}

#[test]
fn revisions() {
    let config = config();

    assert_eq!(parse_rs(&config, "// revisions: a b c").revisions, vec!["a", "b", "c"],);
    assert_eq!(
        parse_makefile(&config, "# revisions: hello there").revisions,
        vec!["hello", "there"],
    );
}

#[test]
fn aux_build() {
    let config = config();

    assert_eq!(
        parse_rs(
            &config,
            r"
        // aux-build: a.rs
        // aux-build: b.rs
        "
        )
        .aux,
        vec!["a.rs", "b.rs"],
    );
}

#[test]
fn no_system_llvm() {
    let mut config = config();

    config.system_llvm = false;
    assert!(!parse_rs(&config, "// no-system-llvm").ignore);

    config.system_llvm = true;
    assert!(parse_rs(&config, "// no-system-llvm").ignore);
}

#[test]
fn llvm_version() {
    let mut config = config();

    config.llvm_version = Some(80102);
    assert!(parse_rs(&config, "// min-llvm-version: 9.0").ignore);

    config.llvm_version = Some(90001);
    assert!(parse_rs(&config, "// min-llvm-version: 9.2").ignore);

    config.llvm_version = Some(90301);
    assert!(!parse_rs(&config, "// min-llvm-version: 9.2").ignore);

    config.llvm_version = Some(100000);
    assert!(!parse_rs(&config, "// min-llvm-version: 9.0").ignore);
}

#[test]
fn ignore_target() {
    let mut config = config();
    config.target = "x86_64-unknown-linux-gnu".to_owned();

    assert!(parse_rs(&config, "// ignore-x86_64-unknown-linux-gnu").ignore);
    assert!(parse_rs(&config, "// ignore-x86_64").ignore);
    assert!(parse_rs(&config, "// ignore-linux").ignore);
    assert!(parse_rs(&config, "// ignore-gnu").ignore);
    assert!(parse_rs(&config, "// ignore-64bit").ignore);

    assert!(!parse_rs(&config, "// ignore-i686").ignore);
    assert!(!parse_rs(&config, "// ignore-windows").ignore);
    assert!(!parse_rs(&config, "// ignore-msvc").ignore);
    assert!(!parse_rs(&config, "// ignore-32bit").ignore);
}

#[test]
fn only_target() {
    let mut config = config();
    config.target = "x86_64-pc-windows-gnu".to_owned();

    assert!(parse_rs(&config, "// only-i686").ignore);
    assert!(parse_rs(&config, "// only-linux").ignore);
    assert!(parse_rs(&config, "// only-msvc").ignore);
    assert!(parse_rs(&config, "// only-32bit").ignore);

    assert!(!parse_rs(&config, "// only-x86_64-pc-windows-gnu").ignore);
    assert!(!parse_rs(&config, "// only-x86_64").ignore);
    assert!(!parse_rs(&config, "// only-windows").ignore);
    assert!(!parse_rs(&config, "// only-gnu").ignore);
    assert!(!parse_rs(&config, "// only-64bit").ignore);
}

#[test]
fn stage() {
    let mut config = config();
    config.stage_id = "stage1".to_owned();

    assert!(parse_rs(&config, "// ignore-stage1").ignore);
    assert!(!parse_rs(&config, "// ignore-stage2").ignore);
}

#[test]
fn cross_compile() {
    let mut config = config();
    config.host = "x86_64-apple-darwin".to_owned();
    config.target = "wasm32-unknown-unknown".to_owned();
    assert!(parse_rs(&config, "// ignore-cross-compile").ignore);

    config.target = config.host.clone();
    assert!(!parse_rs(&config, "// ignore-cross-compile").ignore);
}

#[test]
fn debugger() {
    let mut config = config();
    config.debugger = None;
    assert!(!parse_rs(&config, "// ignore-cdb").ignore);

    config.debugger = Some(Debugger::Cdb);
    assert!(parse_rs(&config, "// ignore-cdb").ignore);

    config.debugger = Some(Debugger::Gdb);
    assert!(parse_rs(&config, "// ignore-gdb").ignore);

    config.debugger = Some(Debugger::Lldb);
    assert!(parse_rs(&config, "// ignore-lldb").ignore);
}

#[test]
fn sanitizers() {
    let mut config = config();

    // Target that supports all sanitizers:
    config.target = "x86_64-unknown-linux-gnu".to_owned();
    assert!(!parse_rs(&config, "// needs-sanitizer-address").ignore);
    assert!(!parse_rs(&config, "// needs-sanitizer-leak").ignore);
    assert!(!parse_rs(&config, "// needs-sanitizer-memory").ignore);
    assert!(!parse_rs(&config, "// needs-sanitizer-thread").ignore);

    // Target that doesn't support sanitizers:
    config.target = "wasm32-unknown-emscripten".to_owned();
    assert!(parse_rs(&config, "// needs-sanitizer-address").ignore);
    assert!(parse_rs(&config, "// needs-sanitizer-leak").ignore);
    assert!(parse_rs(&config, "// needs-sanitizer-memory").ignore);
    assert!(parse_rs(&config, "// needs-sanitizer-thread").ignore);
}

#[test]
fn test_extract_version_range() {
    use super::{extract_llvm_version, extract_version_range};

    assert_eq!(extract_version_range("1.2.3 - 4.5.6", extract_llvm_version), Some((10203, 40506)));
    assert_eq!(extract_version_range("0   - 4.5.6", extract_llvm_version), Some((0, 40506)));
    assert_eq!(extract_version_range("1.2.3 -", extract_llvm_version), None);
    assert_eq!(extract_version_range("1.2.3 - ", extract_llvm_version), None);
    assert_eq!(extract_version_range("- 4.5.6", extract_llvm_version), None);
    assert_eq!(extract_version_range("-", extract_llvm_version), None);
    assert_eq!(extract_version_range(" - 4.5.6", extract_llvm_version), None);
    assert_eq!(extract_version_range("   - 4.5.6", extract_llvm_version), None);
    assert_eq!(extract_version_range("0  -", extract_llvm_version), None);
}
