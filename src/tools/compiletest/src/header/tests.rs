use std::path::Path;

use crate::common::{Config, Debugger};
use crate::header::{make_test_description, parse_normalization_string, EarlyProps};

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
        "--jsondocck-path=",
        "--src-base=",
        "--build-base=",
        "--stage-id=stage2",
        "--cc=c",
        "--cxx=c++",
        "--cflags=",
        "--cxxflags=",
        "--llvm-components=",
        "--android-cross-path=",
        "--target=x86_64-unknown-linux-gnu",
        "--channel=nightly",
    ];
    let args = args.iter().map(ToString::to_string).collect();
    crate::parse_config(args)
}

fn parse_rs(config: &Config, contents: &str) -> EarlyProps {
    let bytes = contents.as_bytes();
    EarlyProps::from_reader(config, Path::new("a.rs"), bytes)
}

fn check_ignore(config: &Config, contents: &str) -> bool {
    let tn = test::DynTestName(String::new());
    let p = Path::new("a.rs");
    let d = make_test_description(&config, tn, p, std::io::Cursor::new(contents), None);
    d.ignore
}

fn parse_makefile(config: &Config, contents: &str) -> EarlyProps {
    let bytes = contents.as_bytes();
    EarlyProps::from_reader(config, Path::new("Makefile"), bytes)
}

#[test]
fn should_fail() {
    let config = config();
    let tn = test::DynTestName(String::new());
    let p = Path::new("a.rs");

    let d = make_test_description(&config, tn.clone(), p, std::io::Cursor::new(""), None);
    assert_eq!(d.should_panic, test::ShouldPanic::No);
    let d = make_test_description(&config, tn, p, std::io::Cursor::new("// should-fail"), None);
    assert_eq!(d.should_panic, test::ShouldPanic::Yes);
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
    assert!(!check_ignore(&config, "// no-system-llvm"));

    config.system_llvm = true;
    assert!(check_ignore(&config, "// no-system-llvm"));
}

#[test]
fn llvm_version() {
    let mut config = config();

    config.llvm_version = Some(80102);
    assert!(check_ignore(&config, "// min-llvm-version: 9.0"));

    config.llvm_version = Some(90001);
    assert!(check_ignore(&config, "// min-llvm-version: 9.2"));

    config.llvm_version = Some(90301);
    assert!(!check_ignore(&config, "// min-llvm-version: 9.2"));

    config.llvm_version = Some(100000);
    assert!(!check_ignore(&config, "// min-llvm-version: 9.0"));
}

#[test]
fn ignore_target() {
    let mut config = config();
    config.target = "x86_64-unknown-linux-gnu".to_owned();

    assert!(check_ignore(&config, "// ignore-x86_64-unknown-linux-gnu"));
    assert!(check_ignore(&config, "// ignore-x86_64"));
    assert!(check_ignore(&config, "// ignore-linux"));
    assert!(check_ignore(&config, "// ignore-gnu"));
    assert!(check_ignore(&config, "// ignore-64bit"));

    assert!(!check_ignore(&config, "// ignore-i686"));
    assert!(!check_ignore(&config, "// ignore-windows"));
    assert!(!check_ignore(&config, "// ignore-msvc"));
    assert!(!check_ignore(&config, "// ignore-32bit"));
}

#[test]
fn only_target() {
    let mut config = config();
    config.target = "x86_64-pc-windows-gnu".to_owned();

    assert!(check_ignore(&config, "// only-x86"));
    assert!(check_ignore(&config, "// only-linux"));
    assert!(check_ignore(&config, "// only-msvc"));
    assert!(check_ignore(&config, "// only-32bit"));

    assert!(!check_ignore(&config, "// only-x86_64-pc-windows-gnu"));
    assert!(!check_ignore(&config, "// only-x86_64"));
    assert!(!check_ignore(&config, "// only-windows"));
    assert!(!check_ignore(&config, "// only-gnu"));
    assert!(!check_ignore(&config, "// only-64bit"));
}

#[test]
fn stage() {
    let mut config = config();
    config.stage_id = "stage1".to_owned();

    assert!(check_ignore(&config, "// ignore-stage1"));
    assert!(!check_ignore(&config, "// ignore-stage2"));
}

#[test]
fn cross_compile() {
    let mut config = config();
    config.host = "x86_64-apple-darwin".to_owned();
    config.target = "wasm32-unknown-unknown".to_owned();
    assert!(check_ignore(&config, "// ignore-cross-compile"));

    config.target = config.host.clone();
    assert!(!check_ignore(&config, "// ignore-cross-compile"));
}

#[test]
fn debugger() {
    let mut config = config();
    config.debugger = None;
    assert!(!check_ignore(&config, "// ignore-cdb"));

    config.debugger = Some(Debugger::Cdb);
    assert!(check_ignore(&config, "// ignore-cdb"));

    config.debugger = Some(Debugger::Gdb);
    assert!(check_ignore(&config, "// ignore-gdb"));

    config.debugger = Some(Debugger::Lldb);
    assert!(check_ignore(&config, "// ignore-lldb"));
}

#[test]
fn sanitizers() {
    let mut config = config();

    // Target that supports all sanitizers:
    config.target = "x86_64-unknown-linux-gnu".to_owned();
    assert!(!check_ignore(&config, "// needs-sanitizer-address"));
    assert!(!check_ignore(&config, "// needs-sanitizer-leak"));
    assert!(!check_ignore(&config, "// needs-sanitizer-memory"));
    assert!(!check_ignore(&config, "// needs-sanitizer-thread"));

    // Target that doesn't support sanitizers:
    config.target = "wasm32-unknown-emscripten".to_owned();
    assert!(check_ignore(&config, "// needs-sanitizer-address"));
    assert!(check_ignore(&config, "// needs-sanitizer-leak"));
    assert!(check_ignore(&config, "// needs-sanitizer-memory"));
    assert!(check_ignore(&config, "// needs-sanitizer-thread"));
}

#[test]
fn asm_support() {
    let mut config = config();

    config.target = "avr-unknown-gnu-atmega328".to_owned();
    assert!(check_ignore(&config, "// needs-asm-support"));

    config.target = "i686-unknown-netbsd".to_owned();
    assert!(!check_ignore(&config, "// needs-asm-support"));
}

#[test]
fn channel() {
    let mut config = config();
    config.channel = "beta".into();

    assert!(check_ignore(&config, "// ignore-beta"));
    assert!(check_ignore(&config, "// only-nightly"));
    assert!(check_ignore(&config, "// only-stable"));

    assert!(!check_ignore(&config, "// only-beta"));
    assert!(!check_ignore(&config, "// ignore-nightly"));
    assert!(!check_ignore(&config, "// ignore-stable"));
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

#[test]
#[should_panic(expected = "Duplicate revision: `rpass1` in line ` rpass1 rpass1`")]
fn test_duplicate_revisions() {
    let config = config();
    parse_rs(&config, "// revisions: rpass1 rpass1");
}
