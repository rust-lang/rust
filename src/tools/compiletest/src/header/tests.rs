use std::io::Read;
use std::path::Path;

use super::iter_header;
use crate::common::{Config, Debugger, Mode};
use crate::header::{EarlyProps, HeadersCache, parse_normalize_rule};

fn make_test_description<R: Read>(
    config: &Config,
    name: test::TestName,
    path: &Path,
    src: R,
    revision: Option<&str>,
) -> test::TestDesc {
    let cache = HeadersCache::load(config);
    let mut poisoned = false;
    let test = crate::header::make_test_description(
        config,
        &cache,
        name,
        path,
        src,
        revision,
        &mut poisoned,
    );
    if poisoned {
        panic!("poisoned!");
    }
    test
}

#[test]
fn test_parse_normalize_rule() {
    let good_data = &[(
        r#"normalize-stderr-32bit: "something (32 bits)" -> "something ($WORD bits)""#,
        "something (32 bits)",
        "something ($WORD bits)",
    )];

    for &(input, expected_regex, expected_replacement) in good_data {
        let parsed = parse_normalize_rule(input);
        let parsed =
            parsed.as_ref().map(|(regex, replacement)| (regex.as_str(), replacement.as_str()));
        assert_eq!(parsed, Some((expected_regex, expected_replacement)));
    }

    let bad_data = &[
        r#"normalize-stderr-32bit "something (32 bits)" -> "something ($WORD bits)""#,
        r#"normalize-stderr-16bit: something (16 bits) -> something ($WORD bits)"#,
        r#"normalize-stderr-32bit: something (32 bits) -> something ($WORD bits)"#,
        r#"normalize-stderr-32bit: "something (32 bits) -> something ($WORD bits)"#,
        r#"normalize-stderr-32bit: "something (32 bits)" -> "something ($WORD bits)"#,
        r#"normalize-stderr-32bit: "something (32 bits)" -> "something ($WORD bits)"."#,
    ];

    for &input in bad_data {
        let parsed = parse_normalize_rule(input);
        assert_eq!(parsed, None);
    }
}

#[derive(Default)]
struct ConfigBuilder {
    mode: Option<String>,
    channel: Option<String>,
    host: Option<String>,
    target: Option<String>,
    stage_id: Option<String>,
    llvm_version: Option<String>,
    git_hash: bool,
    system_llvm: bool,
    profiler_support: bool,
}

impl ConfigBuilder {
    fn mode(&mut self, s: &str) -> &mut Self {
        self.mode = Some(s.to_owned());
        self
    }

    fn channel(&mut self, s: &str) -> &mut Self {
        self.channel = Some(s.to_owned());
        self
    }

    fn host(&mut self, s: &str) -> &mut Self {
        self.host = Some(s.to_owned());
        self
    }

    fn target(&mut self, s: &str) -> &mut Self {
        self.target = Some(s.to_owned());
        self
    }

    fn stage_id(&mut self, s: &str) -> &mut Self {
        self.stage_id = Some(s.to_owned());
        self
    }

    fn llvm_version(&mut self, s: &str) -> &mut Self {
        self.llvm_version = Some(s.to_owned());
        self
    }

    fn git_hash(&mut self, b: bool) -> &mut Self {
        self.git_hash = b;
        self
    }

    fn system_llvm(&mut self, s: bool) -> &mut Self {
        self.system_llvm = s;
        self
    }

    fn profiler_support(&mut self, s: bool) -> &mut Self {
        self.profiler_support = s;
        self
    }

    fn build(&mut self) -> Config {
        let args = &[
            "compiletest",
            "--mode",
            self.mode.as_deref().unwrap_or("ui"),
            "--suite=ui",
            "--compile-lib-path=",
            "--run-lib-path=",
            "--python=",
            "--jsondocck-path=",
            "--src-base=",
            "--build-base=",
            "--sysroot-base=",
            "--cc=c",
            "--cxx=c++",
            "--cflags=",
            "--cxxflags=",
            "--llvm-components=",
            "--android-cross-path=",
            "--stage-id",
            self.stage_id.as_deref().unwrap_or("stage2-x86_64-unknown-linux-gnu"),
            "--channel",
            self.channel.as_deref().unwrap_or("nightly"),
            "--host",
            self.host.as_deref().unwrap_or("x86_64-unknown-linux-gnu"),
            "--target",
            self.target.as_deref().unwrap_or("x86_64-unknown-linux-gnu"),
            "--git-repository=",
            "--nightly-branch=",
            "--git-merge-commit-email=",
        ];
        let mut args: Vec<String> = args.iter().map(ToString::to_string).collect();

        if let Some(ref llvm_version) = self.llvm_version {
            args.push("--llvm-version".to_owned());
            args.push(llvm_version.clone());
        }

        if self.git_hash {
            args.push("--git-hash".to_owned());
        }
        if self.system_llvm {
            args.push("--system-llvm".to_owned());
        }
        if self.profiler_support {
            args.push("--profiler-support".to_owned());
        }

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
}

fn cfg() -> ConfigBuilder {
    ConfigBuilder::default()
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
    let config: Config = cfg().build();
    let tn = test::DynTestName(String::new());
    let p = Path::new("a.rs");

    let d = make_test_description(&config, tn.clone(), p, std::io::Cursor::new(""), None);
    assert_eq!(d.should_panic, test::ShouldPanic::No);
    let d = make_test_description(&config, tn, p, std::io::Cursor::new("//@ should-fail"), None);
    assert_eq!(d.should_panic, test::ShouldPanic::Yes);
}

#[test]
fn revisions() {
    let config: Config = cfg().build();

    assert_eq!(parse_rs(&config, "//@ revisions: a b c").revisions, vec!["a", "b", "c"],);
    assert_eq!(parse_makefile(&config, "# revisions: hello there").revisions, vec![
        "hello", "there"
    ],);
}

#[test]
fn aux_build() {
    let config: Config = cfg().build();

    assert_eq!(
        parse_rs(
            &config,
            r"
        //@ aux-build: a.rs
        //@ aux-build: b.rs
        "
        )
        .aux,
        vec!["a.rs", "b.rs"],
    );
}

#[test]
fn llvm_version() {
    let config: Config = cfg().llvm_version("8.1.2").build();
    assert!(check_ignore(&config, "//@ min-llvm-version: 9.0"));

    let config: Config = cfg().llvm_version("9.0.1").build();
    assert!(check_ignore(&config, "//@ min-llvm-version: 9.2"));

    let config: Config = cfg().llvm_version("9.3.1").build();
    assert!(!check_ignore(&config, "//@ min-llvm-version: 9.2"));

    let config: Config = cfg().llvm_version("10.0.0").build();
    assert!(!check_ignore(&config, "//@ min-llvm-version: 9.0"));
}

#[test]
fn system_llvm_version() {
    let config: Config = cfg().system_llvm(true).llvm_version("17.0.0").build();
    assert!(check_ignore(&config, "//@ min-system-llvm-version: 18.0"));

    let config: Config = cfg().system_llvm(true).llvm_version("18.0.0").build();
    assert!(!check_ignore(&config, "//@ min-system-llvm-version: 18.0"));

    let config: Config = cfg().llvm_version("17.0.0").build();
    assert!(!check_ignore(&config, "//@ min-system-llvm-version: 18.0"));
}

#[test]
fn ignore_target() {
    let config: Config = cfg().target("x86_64-unknown-linux-gnu").build();

    assert!(check_ignore(&config, "//@ ignore-x86_64-unknown-linux-gnu"));
    assert!(check_ignore(&config, "//@ ignore-x86_64"));
    assert!(check_ignore(&config, "//@ ignore-linux"));
    assert!(check_ignore(&config, "//@ ignore-unix"));
    assert!(check_ignore(&config, "//@ ignore-gnu"));
    assert!(check_ignore(&config, "//@ ignore-64bit"));

    assert!(!check_ignore(&config, "//@ ignore-x86"));
    assert!(!check_ignore(&config, "//@ ignore-windows"));
    assert!(!check_ignore(&config, "//@ ignore-msvc"));
    assert!(!check_ignore(&config, "//@ ignore-32bit"));
}

#[test]
fn only_target() {
    let config: Config = cfg().target("x86_64-pc-windows-gnu").build();

    assert!(check_ignore(&config, "//@ only-x86"));
    assert!(check_ignore(&config, "//@ only-linux"));
    assert!(check_ignore(&config, "//@ only-unix"));
    assert!(check_ignore(&config, "//@ only-msvc"));
    assert!(check_ignore(&config, "//@ only-32bit"));

    assert!(!check_ignore(&config, "//@ only-x86_64-pc-windows-gnu"));
    assert!(!check_ignore(&config, "//@ only-x86_64"));
    assert!(!check_ignore(&config, "//@ only-windows"));
    assert!(!check_ignore(&config, "//@ only-gnu"));
    assert!(!check_ignore(&config, "//@ only-64bit"));
}

#[test]
fn stage() {
    let config: Config = cfg().stage_id("stage1-x86_64-unknown-linux-gnu").build();

    assert!(check_ignore(&config, "//@ ignore-stage1"));
    assert!(!check_ignore(&config, "//@ ignore-stage2"));
}

#[test]
fn cross_compile() {
    let config: Config = cfg().host("x86_64-apple-darwin").target("wasm32-unknown-unknown").build();
    assert!(check_ignore(&config, "//@ ignore-cross-compile"));

    let config: Config = cfg().host("x86_64-apple-darwin").target("x86_64-apple-darwin").build();
    assert!(!check_ignore(&config, "//@ ignore-cross-compile"));
}

#[test]
fn debugger() {
    let mut config = cfg().build();
    config.debugger = None;
    assert!(!check_ignore(&config, "//@ ignore-cdb"));

    config.debugger = Some(Debugger::Cdb);
    assert!(check_ignore(&config, "//@ ignore-cdb"));

    config.debugger = Some(Debugger::Gdb);
    assert!(check_ignore(&config, "//@ ignore-gdb"));

    config.debugger = Some(Debugger::Lldb);
    assert!(check_ignore(&config, "//@ ignore-lldb"));
}

#[test]
fn git_hash() {
    let config: Config = cfg().git_hash(false).build();
    assert!(check_ignore(&config, "//@ needs-git-hash"));

    let config: Config = cfg().git_hash(true).build();
    assert!(!check_ignore(&config, "//@ needs-git-hash"));
}

#[test]
fn sanitizers() {
    // Target that supports all sanitizers:
    let config: Config = cfg().target("x86_64-unknown-linux-gnu").build();
    assert!(!check_ignore(&config, "//@ needs-sanitizer-address"));
    assert!(!check_ignore(&config, "//@ needs-sanitizer-leak"));
    assert!(!check_ignore(&config, "//@ needs-sanitizer-memory"));
    assert!(!check_ignore(&config, "//@ needs-sanitizer-thread"));

    // Target that doesn't support sanitizers:
    let config: Config = cfg().target("wasm32-unknown-emscripten").build();
    assert!(check_ignore(&config, "//@ needs-sanitizer-address"));
    assert!(check_ignore(&config, "//@ needs-sanitizer-leak"));
    assert!(check_ignore(&config, "//@ needs-sanitizer-memory"));
    assert!(check_ignore(&config, "//@ needs-sanitizer-thread"));
}

#[test]
fn profiler_support() {
    let config: Config = cfg().profiler_support(false).build();
    assert!(check_ignore(&config, "//@ needs-profiler-support"));

    let config: Config = cfg().profiler_support(true).build();
    assert!(!check_ignore(&config, "//@ needs-profiler-support"));
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
        let config = cfg().target(target).build();
        assert_eq!(config.has_asm_support(), has_asm);
        assert_eq!(check_ignore(&config, "//@ needs-asm-support"), !has_asm)
    }
}

#[test]
fn channel() {
    let config: Config = cfg().channel("beta").build();

    assert!(check_ignore(&config, "//@ ignore-beta"));
    assert!(check_ignore(&config, "//@ only-nightly"));
    assert!(check_ignore(&config, "//@ only-stable"));

    assert!(!check_ignore(&config, "//@ only-beta"));
    assert!(!check_ignore(&config, "//@ ignore-nightly"));
    assert!(!check_ignore(&config, "//@ ignore-stable"));
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
    let config: Config = cfg().build();
    parse_rs(&config, "//@ revisions: rpass1 rpass1");
}

#[test]
fn ignore_arch() {
    let archs = [
        ("x86_64-unknown-linux-gnu", "x86_64"),
        ("i686-unknown-linux-gnu", "x86"),
        ("nvptx64-nvidia-cuda", "nvptx64"),
        ("thumbv7m-none-eabi", "thumb"),
    ];
    for (target, arch) in archs {
        let config: Config = cfg().target(target).build();
        assert!(config.matches_arch(arch), "{target} {arch}");
        assert!(check_ignore(&config, &format!("//@ ignore-{arch}")));
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
        let config = cfg().target(target).build();
        assert!(config.matches_os(os), "{target} {os}");
        assert!(check_ignore(&config, &format!("//@ ignore-{os}")));
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
        let config: Config = cfg().target(target).build();
        assert!(config.matches_env(env), "{target} {env}");
        assert!(check_ignore(&config, &format!("//@ ignore-{env}")));
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
        let config: Config = cfg().target(target).build();
        assert!(config.matches_abi(abi), "{target} {abi}");
        assert!(check_ignore(&config, &format!("//@ ignore-{abi}")));
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
        let config = cfg().target(target).build();
        assert_eq!(config.is_big_endian(), is_big, "{target} {is_big}");
        assert_eq!(check_ignore(&config, "//@ ignore-endian-big"), is_big);
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
        let config: Config = cfg().target(target).build();
        assert_eq!(config.get_pointer_width(), width, "{target} {width}");
        assert_eq!(check_ignore(&config, "//@ ignore-16bit"), width == 16);
        assert_eq!(check_ignore(&config, "//@ ignore-32bit"), width == 32);
        assert_eq!(check_ignore(&config, "//@ ignore-64bit"), width == 64);
    }
}

#[test]
fn wasm_special() {
    let ignores = [
        ("wasm32-unknown-unknown", "emscripten", true),
        ("wasm32-unknown-unknown", "wasm32", true),
        ("wasm32-unknown-unknown", "wasm32-bare", true),
        ("wasm32-unknown-unknown", "wasm64", false),
        ("wasm32-unknown-emscripten", "emscripten", true),
        ("wasm32-unknown-emscripten", "wasm32", true),
        ("wasm32-unknown-emscripten", "wasm32-bare", false),
        ("wasm32-wasi", "emscripten", false),
        ("wasm32-wasi", "wasm32", true),
        ("wasm32-wasi", "wasm32-bare", false),
        ("wasm32-wasi", "wasi", true),
        ("wasm32-wasip1", "emscripten", false),
        ("wasm32-wasip1", "wasm32", true),
        ("wasm32-wasip1", "wasm32-bare", false),
        ("wasm32-wasip1", "wasi", true),
        ("wasm64-unknown-unknown", "emscripten", false),
        ("wasm64-unknown-unknown", "wasm32", false),
        ("wasm64-unknown-unknown", "wasm32-bare", false),
        ("wasm64-unknown-unknown", "wasm64", true),
    ];
    for (target, pattern, ignore) in ignores {
        let config: Config = cfg().target(target).build();
        assert_eq!(
            check_ignore(&config, &format!("//@ ignore-{pattern}")),
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
        let config: Config = cfg().target(target).build();
        assert!(config.matches_family(family));
        let other = if family == "windows" { "unix" } else { "windows" };
        assert!(!config.matches_family(other));
        assert!(check_ignore(&config, &format!("//@ ignore-{family}")));
        assert!(!check_ignore(&config, &format!("//@ ignore-{other}")));
    }
}

#[test]
fn ignore_mode() {
    for mode in ["coverage-map", "coverage-run"] {
        // Indicate profiler support so that "coverage-run" tests aren't skipped.
        let config: Config = cfg().mode(mode).profiler_support(true).build();
        let other = if mode == "coverage-run" { "coverage-map" } else { "coverage-run" };

        assert_ne!(mode, other);

        assert!(check_ignore(&config, &format!("//@ ignore-mode-{mode}")));
        assert!(!check_ignore(&config, &format!("//@ ignore-mode-{other}")));
    }
}

#[test]
fn threads_support() {
    let threads = [
        ("x86_64-unknown-linux-gnu", true),
        ("aarch64-apple-darwin", true),
        ("wasm32-unknown-unknown", false),
        ("wasm64-unknown-unknown", false),
        ("wasm32-wasip1", false),
        ("wasm32-wasip1-threads", true),
    ];
    for (target, has_threads) in threads {
        let config = cfg().target(target).build();
        assert_eq!(config.has_threads(), has_threads);
        assert_eq!(check_ignore(&config, "//@ needs-threads"), !has_threads)
    }
}

fn run_path(poisoned: &mut bool, path: &Path, buf: &[u8]) {
    let rdr = std::io::Cursor::new(&buf);
    iter_header(Mode::Ui, "ui", poisoned, path, rdr, &mut |_| {});
}

#[test]
fn test_unknown_directive_check() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Path::new("a.rs"),
        include_bytes!("./test-auxillary/unknown_directive.rs"),
    );
    assert!(poisoned);
}

#[test]
fn test_known_legacy_directive_check() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Path::new("a.rs"),
        include_bytes!("./test-auxillary/known_legacy_directive.rs"),
    );
    assert!(poisoned);
}

#[test]
fn test_known_directive_check_no_error() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Path::new("a.rs"),
        include_bytes!("./test-auxillary/known_directive.rs"),
    );
    assert!(!poisoned);
}

#[test]
fn test_error_annotation_no_error() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Path::new("a.rs"),
        include_bytes!("./test-auxillary/error_annotation.rs"),
    );
    assert!(!poisoned);
}

#[test]
fn test_non_rs_unknown_directive_not_checked() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Path::new("a.Makefile"),
        include_bytes!("./test-auxillary/not_rs.Makefile"),
    );
    assert!(!poisoned);
}

#[test]
fn test_trailing_directive() {
    let mut poisoned = false;
    run_path(&mut poisoned, Path::new("a.rs"), b"//@ only-x86 only-arm");
    assert!(poisoned);
}

#[test]
fn test_trailing_directive_with_comment() {
    let mut poisoned = false;
    run_path(&mut poisoned, Path::new("a.rs"), b"//@ only-x86   only-arm with comment");
    assert!(poisoned);
}

#[test]
fn test_not_trailing_directive() {
    let mut poisoned = false;
    run_path(&mut poisoned, Path::new("a.rs"), b"//@ revisions: incremental");
    assert!(!poisoned);
}
