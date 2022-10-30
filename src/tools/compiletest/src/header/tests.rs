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
        "--python=",
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
    let mut args: Vec<String> = args.iter().map(ToString::to_string).collect();
    args.push("--rustc-path".to_string());
    // This is a subtle/fragile thing. On rust-lang CI, there is no global
    // `rustc`, and Cargo doesn't offer a convenient way to get the path to
    // `rustc`. Fortunately bootstrap sets `RUSTC` for us, which is pointing
    // to the stage0 compiler.
    //
    // Otherwise, if you are running compiletests's tests manually, you
    // probably don't have `RUSTC` set, in which case this falls back to the
    // global rustc. If your global rustc is too far out of sync with stage0,
    // then this may cause confusing errors. Or if for some reason you don't
    // have rustc in PATH, that would also fail.
    args.push(std::env::var("RUSTC").unwrap_or_else(|_| {
        eprintln!(
            "warning: RUSTC not set, using global rustc (are you not running via bootstrap?)"
        );
        "rustc".to_string()
    }));
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
    let asms = [
        ("avr-unknown-gnu-atmega328", false),
        ("i686-unknown-netbsd", true),
        ("riscv32gc-unknown-linux-gnu", true),
        ("riscv64imac-unknown-none-elf", true),
        ("x86_64-unknown-linux-gnu", true),
        ("i686-unknown-netbsd", true),
    ];
    for (target, has_asm) in asms {
        let mut config = config();
        config.target = target.to_string();
        assert_eq!(config.has_asm_support(), has_asm);
        assert_eq!(check_ignore(&config, "// needs-asm-support"), !has_asm)
    }
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

#[test]
fn ignore_arch() {
    let archs = [
        ("x86_64-unknown-linux-gnu", "x86_64"),
        ("i686-unknown-linux-gnu", "x86"),
        ("nvptx64-nvidia-cuda", "nvptx64"),
        ("asmjs-unknown-emscripten", "wasm32"),
        ("asmjs-unknown-emscripten", "asmjs"),
        ("thumbv7m-none-eabi", "thumb"),
    ];
    for (target, arch) in archs {
        let mut config = config();
        config.target = target.to_string();
        assert!(config.matches_arch(arch), "{target} {arch}");
        assert!(check_ignore(&config, &format!("// ignore-{arch}")));
    }
}

#[test]
fn matches_os() {
    let oss = [
        ("x86_64-unknown-linux-gnu", "linux"),
        ("x86_64-fortanix-unknown-sgx", "unknown"),
        ("wasm32-unknown-unknown", "unknown"),
        ("x86_64-unknown-none", "none"),
    ];
    for (target, os) in oss {
        let mut config = config();
        config.target = target.to_string();
        assert!(config.matches_os(os), "{target} {os}");
        assert!(check_ignore(&config, &format!("// ignore-{os}")));
    }
}

#[test]
fn matches_env() {
    let envs = [
        ("x86_64-unknown-linux-gnu", "gnu"),
        ("x86_64-fortanix-unknown-sgx", "sgx"),
        ("arm-unknown-linux-musleabi", "musl"),
    ];
    for (target, env) in envs {
        let mut config = config();
        config.target = target.to_string();
        assert!(config.matches_env(env), "{target} {env}");
        assert!(check_ignore(&config, &format!("// ignore-{env}")));
    }
}

#[test]
fn matches_abi() {
    let abis = [
        ("aarch64-apple-ios-macabi", "macabi"),
        ("x86_64-unknown-linux-gnux32", "x32"),
        ("arm-unknown-linux-gnueabi", "eabi"),
    ];
    for (target, abi) in abis {
        let mut config = config();
        config.target = target.to_string();
        assert!(config.matches_abi(abi), "{target} {abi}");
        assert!(check_ignore(&config, &format!("// ignore-{abi}")));
    }
}

#[test]
fn is_big_endian() {
    let endians = [
        ("x86_64-unknown-linux-gnu", false),
        ("bpfeb-unknown-none", true),
        ("m68k-unknown-linux-gnu", true),
        ("aarch64_be-unknown-linux-gnu", true),
        ("powerpc64-unknown-linux-gnu", true),
    ];
    for (target, is_big) in endians {
        let mut config = config();
        config.target = target.to_string();
        assert_eq!(config.is_big_endian(), is_big, "{target} {is_big}");
        assert_eq!(check_ignore(&config, "// ignore-endian-big"), is_big);
    }
}

#[test]
fn pointer_width() {
    let widths = [
        ("x86_64-unknown-linux-gnu", 64),
        ("i686-unknown-linux-gnu", 32),
        ("arm64_32-apple-watchos", 32),
        ("msp430-none-elf", 16),
    ];
    for (target, width) in widths {
        let mut config = config();
        config.target = target.to_string();
        assert_eq!(config.get_pointer_width(), width, "{target} {width}");
        assert_eq!(check_ignore(&config, "// ignore-16bit"), width == 16);
        assert_eq!(check_ignore(&config, "// ignore-32bit"), width == 32);
        assert_eq!(check_ignore(&config, "// ignore-64bit"), width == 64);
    }
}

#[test]
fn wasm_special() {
    let ignores = [
        ("wasm32-unknown-unknown", "emscripten", true),
        ("wasm32-unknown-unknown", "wasm32", true),
        ("wasm32-unknown-unknown", "wasm32-bare", true),
        ("wasm32-unknown-unknown", "wasm64", false),
        ("asmjs-unknown-emscripten", "emscripten", true),
        ("asmjs-unknown-emscripten", "wasm32", true),
        ("asmjs-unknown-emscripten", "wasm32-bare", false),
        ("wasm32-unknown-emscripten", "emscripten", true),
        ("wasm32-unknown-emscripten", "wasm32", true),
        ("wasm32-unknown-emscripten", "wasm32-bare", false),
        ("wasm32-wasi", "emscripten", false),
        ("wasm32-wasi", "wasm32", true),
        ("wasm32-wasi", "wasm32-bare", false),
        ("wasm32-wasi", "wasi", true),
        ("wasm64-unknown-unknown", "emscripten", false),
        ("wasm64-unknown-unknown", "wasm32", false),
        ("wasm64-unknown-unknown", "wasm32-bare", false),
        ("wasm64-unknown-unknown", "wasm64", true),
    ];
    for (target, pattern, ignore) in ignores {
        let mut config = config();
        config.target = target.to_string();
        assert_eq!(
            check_ignore(&config, &format!("// ignore-{pattern}")),
            ignore,
            "{target} {pattern}"
        );
    }
}

#[test]
fn families() {
    let families = [
        ("x86_64-unknown-linux-gnu", "unix"),
        ("x86_64-pc-windows-gnu", "windows"),
        ("wasm32-unknown-unknown", "wasm"),
        ("wasm32-unknown-emscripten", "wasm"),
        ("wasm32-unknown-emscripten", "unix"),
    ];
    for (target, family) in families {
        let mut config = config();
        config.target = target.to_string();
        assert!(config.matches_family(family));
        let other = if family == "windows" { "unix" } else { "windows" };
        assert!(!config.matches_family(other));
        assert!(check_ignore(&config, &format!("// ignore-{family}")));
        assert!(!check_ignore(&config, &format!("// ignore-{other}")));
    }
}
