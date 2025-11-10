use camino::Utf8Path;
use semver::Version;

use crate::common::{Config, Debugger, TestMode};
use crate::directives::{
    self, AuxProps, DirectivesCache, EarlyProps, Edition, EditionRange, FileDirectives,
    extract_llvm_version, extract_version_range, line_directive, parse_edition,
    parse_normalize_rule,
};
use crate::executor::{CollectedTestDesc, ShouldFail};

fn make_test_description(
    config: &Config,
    name: String,
    path: &Utf8Path,
    filterable_path: &Utf8Path,
    file_contents: &str,
    revision: Option<&str>,
) -> CollectedTestDesc {
    let cache = DirectivesCache::load(config);
    let mut poisoned = false;
    let file_directives = FileDirectives::from_file_contents(path, file_contents);

    let mut aux_props = AuxProps::default();
    let test = crate::directives::make_test_description(
        config,
        &cache,
        name,
        path,
        filterable_path,
        &file_directives,
        revision,
        &mut poisoned,
        &mut aux_props,
    );
    if poisoned {
        panic!("poisoned!");
    }
    test
}

#[test]
fn test_parse_normalize_rule() {
    let good_data = &[
        (
            r#""something (32 bits)" -> "something ($WORD bits)""#,
            "something (32 bits)",
            "something ($WORD bits)",
        ),
        (r#"  " with whitespace"   ->   "    replacement""#, " with whitespace", "    replacement"),
    ];

    for &(input, expected_regex, expected_replacement) in good_data {
        let parsed = parse_normalize_rule(input);
        let parsed =
            parsed.as_ref().map(|(regex, replacement)| (regex.as_str(), replacement.as_str()));
        assert_eq!(parsed, Some((expected_regex, expected_replacement)));
    }

    let bad_data = &[
        r#"something (11 bits) -> something ($WORD bits)"#,
        r#"something (12 bits) -> something ($WORD bits)"#,
        r#""something (13 bits) -> something ($WORD bits)"#,
        r#""something (14 bits)" -> "something ($WORD bits)"#,
        r#""something (15 bits)" -> "something ($WORD bits)"."#,
    ];

    for &input in bad_data {
        println!("- {input:?}");
        let parsed = parse_normalize_rule(input);
        assert_eq!(parsed, None);
    }
}

#[derive(Default)]
struct ConfigBuilder {
    mode: Option<String>,
    channel: Option<String>,
    edition: Option<Edition>,
    host: Option<String>,
    target: Option<String>,
    stage: Option<u32>,
    stage_id: Option<String>,
    llvm_version: Option<String>,
    git_hash: bool,
    system_llvm: bool,
    profiler_runtime: bool,
    rustc_debug_assertions: bool,
    std_debug_assertions: bool,
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

    fn edition(&mut self, e: Edition) -> &mut Self {
        self.edition = Some(e);
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

    fn stage(&mut self, n: u32) -> &mut Self {
        self.stage = Some(n);
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

    fn profiler_runtime(&mut self, is_available: bool) -> &mut Self {
        self.profiler_runtime = is_available;
        self
    }

    fn rustc_debug_assertions(&mut self, is_enabled: bool) -> &mut Self {
        self.rustc_debug_assertions = is_enabled;
        self
    }

    fn std_debug_assertions(&mut self, is_enabled: bool) -> &mut Self {
        self.std_debug_assertions = is_enabled;
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
            "--src-root=",
            "--src-test-suite-root=",
            "--build-root=",
            "--build-test-suite-root=",
            "--sysroot-base=",
            "--cc=c",
            "--cxx=c++",
            "--cflags=",
            "--cxxflags=",
            "--llvm-components=",
            "--android-cross-path=",
            "--stage",
            &self.stage.unwrap_or(2).to_string(),
            "--stage-id",
            self.stage_id.as_deref().unwrap_or("stage2-x86_64-unknown-linux-gnu"),
            "--channel",
            self.channel.as_deref().unwrap_or("nightly"),
            "--host",
            self.host.as_deref().unwrap_or("x86_64-unknown-linux-gnu"),
            "--target",
            self.target.as_deref().unwrap_or("x86_64-unknown-linux-gnu"),
            "--nightly-branch=",
            "--git-merge-commit-email=",
            "--minicore-path=",
        ];
        let mut args: Vec<String> = args.iter().map(ToString::to_string).collect();

        if let Some(edition) = &self.edition {
            args.push(format!("--edition={edition}"));
        }

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
        if self.profiler_runtime {
            args.push("--profiler-runtime".to_owned());
        }
        if self.rustc_debug_assertions {
            args.push("--with-rustc-debug-assertions".to_owned());
        }
        if self.std_debug_assertions {
            args.push("--with-std-debug-assertions".to_owned());
        }

        args.push("--rustc-path".to_string());
        args.push(std::env::var("TEST_RUSTC").expect("must be configured by bootstrap"));

        crate::parse_config(args)
    }
}

fn cfg() -> ConfigBuilder {
    ConfigBuilder::default()
}

fn parse_early_props(config: &Config, contents: &str) -> EarlyProps {
    let file_directives = FileDirectives::from_file_contents(Utf8Path::new("a.rs"), contents);
    EarlyProps::from_file_directives(config, &file_directives)
}

fn check_ignore(config: &Config, contents: &str) -> bool {
    let tn = String::new();
    let p = Utf8Path::new("a.rs");
    let d = make_test_description(&config, tn, p, p, contents, None);
    d.ignore
}

#[test]
fn should_fail() {
    let config: Config = cfg().build();
    let tn = String::new();
    let p = Utf8Path::new("a.rs");

    let d = make_test_description(&config, tn.clone(), p, p, "", None);
    assert_eq!(d.should_fail, ShouldFail::No);
    let d = make_test_description(&config, tn, p, p, "//@ should-fail", None);
    assert_eq!(d.should_fail, ShouldFail::Yes);
}

#[test]
fn revisions() {
    let config: Config = cfg().build();

    assert_eq!(parse_early_props(&config, "//@ revisions: a b c").revisions, vec!["a", "b", "c"],);
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

    let config: Config = cfg().llvm_version("10.0.0").build();
    assert!(check_ignore(&config, "//@ exact-llvm-major-version: 9.0"));

    let config: Config = cfg().llvm_version("9.0.0").build();
    assert!(check_ignore(&config, "//@ exact-llvm-major-version: 10.0"));

    let config: Config = cfg().llvm_version("10.0.0").build();
    assert!(!check_ignore(&config, "//@ exact-llvm-major-version: 10.0"));

    let config: Config = cfg().llvm_version("10.0.0").build();
    assert!(!check_ignore(&config, "//@ exact-llvm-major-version: 10"));

    let config: Config = cfg().llvm_version("10.6.2").build();
    assert!(!check_ignore(&config, "//@ exact-llvm-major-version: 10"));

    let config: Config = cfg().llvm_version("19.0.0").build();
    assert!(!check_ignore(&config, "//@ max-llvm-major-version: 19"));

    let config: Config = cfg().llvm_version("19.1.2").build();
    assert!(!check_ignore(&config, "//@ max-llvm-major-version: 19"));

    let config: Config = cfg().llvm_version("20.0.0").build();
    assert!(check_ignore(&config, "//@ max-llvm-major-version: 19"));
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
fn rustc_debug_assertions() {
    let config: Config = cfg().rustc_debug_assertions(false).build();

    assert!(check_ignore(&config, "//@ needs-rustc-debug-assertions"));
    assert!(!check_ignore(&config, "//@ ignore-rustc-debug-assertions"));

    let config: Config = cfg().rustc_debug_assertions(true).build();

    assert!(!check_ignore(&config, "//@ needs-rustc-debug-assertions"));
    assert!(check_ignore(&config, "//@ ignore-rustc-debug-assertions"));
}

#[test]
fn std_debug_assertions() {
    let config: Config = cfg().std_debug_assertions(false).build();

    assert!(check_ignore(&config, "//@ needs-std-debug-assertions"));
    assert!(!check_ignore(&config, "//@ ignore-std-debug-assertions"));

    let config: Config = cfg().std_debug_assertions(true).build();

    assert!(!check_ignore(&config, "//@ needs-std-debug-assertions"));
    assert!(check_ignore(&config, "//@ ignore-std-debug-assertions"));
}

#[test]
fn stage() {
    let config: Config = cfg().stage(1).stage_id("stage1-x86_64-unknown-linux-gnu").build();

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
fn profiler_runtime() {
    let config: Config = cfg().profiler_runtime(false).build();
    assert!(check_ignore(&config, "//@ needs-profiler-runtime"));

    let config: Config = cfg().profiler_runtime(true).build();
    assert!(!check_ignore(&config, "//@ needs-profiler-runtime"));
}

#[test]
fn asm_support() {
    let asms = [
        ("avr-none", false),
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
fn test_extract_llvm_version() {
    // Note: officially, semver *requires* that versions at the minimum have all three
    // `major.minor.patch` numbers, though for test-writer's convenience we allow omitting the minor
    // and patch numbers (which will be stubbed out as 0).
    assert_eq!(extract_llvm_version("0"), Version::new(0, 0, 0));
    assert_eq!(extract_llvm_version("0.0"), Version::new(0, 0, 0));
    assert_eq!(extract_llvm_version("0.0.0"), Version::new(0, 0, 0));
    assert_eq!(extract_llvm_version("1"), Version::new(1, 0, 0));
    assert_eq!(extract_llvm_version("1.2"), Version::new(1, 2, 0));
    assert_eq!(extract_llvm_version("1.2.3"), Version::new(1, 2, 3));
    assert_eq!(extract_llvm_version("4.5.6git"), Version::new(4, 5, 6));
    assert_eq!(extract_llvm_version("4.5.6-rc1"), Version::new(4, 5, 6));
    assert_eq!(extract_llvm_version("123.456.789-rc1"), Version::new(123, 456, 789));
    assert_eq!(extract_llvm_version("8.1.2-rust"), Version::new(8, 1, 2));
    assert_eq!(extract_llvm_version("9.0.1-rust-1.43.0-dev"), Version::new(9, 0, 1));
    assert_eq!(extract_llvm_version("9.3.1-rust-1.43.0-dev"), Version::new(9, 3, 1));
    assert_eq!(extract_llvm_version("10.0.0-rust"), Version::new(10, 0, 0));
    assert_eq!(extract_llvm_version("11.1.0"), Version::new(11, 1, 0));
    assert_eq!(extract_llvm_version("12.0.0libcxx"), Version::new(12, 0, 0));
    assert_eq!(extract_llvm_version("12.0.0-rc3"), Version::new(12, 0, 0));
    assert_eq!(extract_llvm_version("13.0.0git"), Version::new(13, 0, 0));
}

#[test]
#[should_panic]
fn test_llvm_version_invalid_components() {
    extract_llvm_version("4.x.6");
}

#[test]
#[should_panic]
fn test_llvm_version_invalid_prefix() {
    extract_llvm_version("meow4.5.6");
}

#[test]
#[should_panic]
fn test_llvm_version_too_many_components() {
    extract_llvm_version("4.5.6.7");
}

#[test]
fn test_extract_version_range() {
    let wrapped_extract = |s: &str| Some(extract_llvm_version(s));

    assert_eq!(
        extract_version_range("1.2.3 - 4.5.6", wrapped_extract),
        Some((Version::new(1, 2, 3), Version::new(4, 5, 6)))
    );
    assert_eq!(
        extract_version_range("0   - 4.5.6", wrapped_extract),
        Some((Version::new(0, 0, 0), Version::new(4, 5, 6)))
    );
    assert_eq!(extract_version_range("1.2.3 -", wrapped_extract), None);
    assert_eq!(extract_version_range("1.2.3 - ", wrapped_extract), None);
    assert_eq!(extract_version_range("- 4.5.6", wrapped_extract), None);
    assert_eq!(extract_version_range("-", wrapped_extract), None);
    assert_eq!(extract_version_range(" - 4.5.6", wrapped_extract), None);
    assert_eq!(extract_version_range("   - 4.5.6", wrapped_extract), None);
    assert_eq!(extract_version_range("0  -", wrapped_extract), None);
}

#[test]
#[should_panic(expected = "duplicate revision: `rpass1` in line ` rpass1 rpass1`")]
fn test_duplicate_revisions() {
    let config: Config = cfg().build();
    parse_early_props(&config, "//@ revisions: rpass1 rpass1");
}

#[test]
#[should_panic(
    expected = "revision name `CHECK` is not permitted in a test suite that uses `FileCheck` annotations"
)]
fn test_assembly_mode_forbidden_revisions() {
    let config = cfg().mode("assembly").build();
    parse_early_props(&config, "//@ revisions: CHECK");
}

#[test]
#[should_panic(expected = "revision name `true` is not permitted")]
fn test_forbidden_revisions() {
    let config = cfg().mode("ui").build();
    parse_early_props(&config, "//@ revisions: true");
}

#[test]
#[should_panic(
    expected = "revision name `CHECK` is not permitted in a test suite that uses `FileCheck` annotations"
)]
fn test_codegen_mode_forbidden_revisions() {
    let config = cfg().mode("codegen").build();
    parse_early_props(&config, "//@ revisions: CHECK");
}

#[test]
#[should_panic(
    expected = "revision name `CHECK` is not permitted in a test suite that uses `FileCheck` annotations"
)]
fn test_miropt_mode_forbidden_revisions() {
    let config = cfg().mode("mir-opt").build();
    parse_early_props(&config, "//@ revisions: CHECK");
}

#[test]
fn test_forbidden_revisions_allowed_in_non_filecheck_dir() {
    let revisions = ["CHECK", "COM", "NEXT", "SAME", "EMPTY", "NOT", "COUNT", "DAG", "LABEL"];
    let modes = [
        "pretty",
        "debuginfo",
        "rustdoc",
        "rustdoc-json",
        "codegen-units",
        "incremental",
        "ui",
        "rustdoc-js",
        "coverage-map",
        "coverage-run",
        "crashes",
    ];

    for rev in revisions {
        let content = format!("//@ revisions: {rev}");
        for mode in modes {
            let config = cfg().mode(mode).build();
            parse_early_props(&config, &content);
        }
    }
}

#[test]
fn ignore_arch() {
    let archs = [
        ("x86_64-unknown-linux-gnu", "x86_64"),
        ("i686-unknown-linux-gnu", "x86"),
        ("nvptx64-nvidia-cuda", "nvptx64"),
        ("thumbv7m-none-eabi", "thumb"),
        ("i586-unknown-linux-gnu", "x86"),
        ("i586-unknown-linux-gnu", "i586"),
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
        ("aarch64-apple-ios-macabi", "macabi"),
    ];
    for (target, env) in envs {
        let config: Config = cfg().target(target).build();
        assert!(config.matches_env(env), "{target} {env}");
        assert!(check_ignore(&config, &format!("//@ ignore-{env}")));
    }
}

#[test]
fn matches_abi() {
    let abis = [("x86_64-unknown-linux-gnux32", "x32"), ("arm-unknown-linux-gnueabi", "eabi")];
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
fn ignore_coverage() {
    // Indicate profiler runtime availability so that "coverage-run" tests aren't skipped.
    let config = cfg().mode("coverage-map").profiler_runtime(true).build();
    assert!(check_ignore(&config, "//@ ignore-coverage-map"));
    assert!(!check_ignore(&config, "//@ ignore-coverage-run"));

    let config = cfg().mode("coverage-run").profiler_runtime(true).build();
    assert!(!check_ignore(&config, "//@ ignore-coverage-map"));
    assert!(check_ignore(&config, "//@ ignore-coverage-run"));
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

fn run_path(poisoned: &mut bool, path: &Utf8Path, file_contents: &str) {
    let file_directives = FileDirectives::from_file_contents(path, file_contents);
    let result = directives::do_early_directives_check(TestMode::Ui, &file_directives);
    if result.is_err() {
        *poisoned = true;
    }
}

#[test]
fn test_unknown_directive_check() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Utf8Path::new("a.rs"),
        include_str!("./test-auxillary/unknown_directive.rs"),
    );
    assert!(poisoned);
}

#[test]
fn test_known_directive_check_no_error() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Utf8Path::new("a.rs"),
        include_str!("./test-auxillary/known_directive.rs"),
    );
    assert!(!poisoned);
}

#[test]
fn test_error_annotation_no_error() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Utf8Path::new("a.rs"),
        include_str!("./test-auxillary/error_annotation.rs"),
    );
    assert!(!poisoned);
}

#[test]
fn test_non_rs_unknown_directive_not_checked() {
    let mut poisoned = false;
    run_path(
        &mut poisoned,
        Utf8Path::new("a.Makefile"),
        include_str!("./test-auxillary/not_rs.Makefile"),
    );
    assert!(!poisoned);
}

#[test]
fn test_trailing_directive() {
    let mut poisoned = false;
    run_path(&mut poisoned, Utf8Path::new("a.rs"), "//@ only-x86 only-arm");
    assert!(poisoned);
}

#[test]
fn test_trailing_directive_with_comment() {
    let mut poisoned = false;
    run_path(&mut poisoned, Utf8Path::new("a.rs"), "//@ only-x86   only-arm with comment");
    assert!(poisoned);
}

#[test]
fn test_not_trailing_directive() {
    let mut poisoned = false;
    run_path(&mut poisoned, Utf8Path::new("a.rs"), "//@ revisions: incremental");
    assert!(!poisoned);
}

#[test]
fn test_needs_target_has_atomic() {
    use std::collections::BTreeSet;

    // `x86_64-unknown-linux-gnu` supports `["8", "16", "32", "64", "ptr"]` but not `128`.

    let config = cfg().target("x86_64-unknown-linux-gnu").build();
    // Expectation sanity check.
    assert_eq!(
        config.target_cfg().target_has_atomic,
        BTreeSet::from([
            "8".to_string(),
            "16".to_string(),
            "32".to_string(),
            "64".to_string(),
            "ptr".to_string()
        ]),
        "expected `x86_64-unknown-linux-gnu` to not have 128-bit atomic support"
    );

    assert!(!check_ignore(&config, "//@ needs-target-has-atomic: 8"));
    assert!(!check_ignore(&config, "//@ needs-target-has-atomic: 16"));
    assert!(!check_ignore(&config, "//@ needs-target-has-atomic: 32"));
    assert!(!check_ignore(&config, "//@ needs-target-has-atomic: 64"));
    assert!(!check_ignore(&config, "//@ needs-target-has-atomic: ptr"));

    assert!(check_ignore(&config, "//@ needs-target-has-atomic: 128"));

    assert!(!check_ignore(&config, "//@ needs-target-has-atomic: 8,16,32,64,ptr"));

    assert!(check_ignore(&config, "//@ needs-target-has-atomic: 8,16,32,64,ptr,128"));

    // Check whitespace between widths is permitted.
    assert!(!check_ignore(&config, "//@ needs-target-has-atomic: 8, ptr"));
    assert!(check_ignore(&config, "//@ needs-target-has-atomic: 8, ptr, 128"));
}

#[test]
fn test_rustc_abi() {
    let config = cfg().target("i686-unknown-linux-gnu").build();
    assert_eq!(config.target_cfg().rustc_abi, Some("x86-sse2".to_string()));
    assert!(check_ignore(&config, "//@ ignore-rustc_abi-x86-sse2"));
    assert!(!check_ignore(&config, "//@ only-rustc_abi-x86-sse2"));
    let config = cfg().target("x86_64-unknown-linux-gnu").build();
    assert_eq!(config.target_cfg().rustc_abi, None);
    assert!(!check_ignore(&config, "//@ ignore-rustc_abi-x86-sse2"));
    assert!(check_ignore(&config, "//@ only-rustc_abi-x86-sse2"));
}

#[test]
fn test_supported_crate_types() {
    // Basic assumptions check on under-test compiler's `--print=supported-crate-types` output based
    // on knowledge about the cherry-picked `x86_64-unknown-linux-gnu` and `wasm32-unknown-unknown`
    // targets. Also smoke tests the `needs-crate-type` directive itself.

    use std::collections::HashSet;

    let config = cfg().target("x86_64-unknown-linux-gnu").build();
    assert_eq!(
        config.supported_crate_types().iter().map(String::as_str).collect::<HashSet<_>>(),
        HashSet::from(["bin", "cdylib", "dylib", "lib", "proc-macro", "rlib", "staticlib"]),
    );
    assert!(!check_ignore(&config, "//@ needs-crate-type: rlib"));
    assert!(!check_ignore(&config, "//@ needs-crate-type: dylib"));
    assert!(!check_ignore(
        &config,
        "//@ needs-crate-type: bin, cdylib, dylib, lib, proc-macro, rlib, staticlib"
    ));

    let config = cfg().target("wasm32-unknown-unknown").build();
    assert_eq!(
        config.supported_crate_types().iter().map(String::as_str).collect::<HashSet<_>>(),
        HashSet::from(["bin", "cdylib", "lib", "rlib", "staticlib"]),
    );

    // rlib is supported
    assert!(!check_ignore(&config, "//@ needs-crate-type: rlib"));
    // dylib is not
    assert!(check_ignore(&config, "//@ needs-crate-type: dylib"));
    // If multiple crate types are specified, then all specified crate types need to be supported.
    assert!(check_ignore(&config, "//@ needs-crate-type: cdylib, dylib"));
    assert!(check_ignore(
        &config,
        "//@ needs-crate-type: bin, cdylib, dylib, lib, proc-macro, rlib, staticlib"
    ));
}

#[test]
fn test_ignore_auxiliary() {
    let config = cfg().build();
    assert!(check_ignore(&config, "//@ ignore-auxiliary"));
}

#[test]
fn test_needs_target_std() {
    // Cherry-picks two targets:
    // 1. `x86_64-unknown-none`: Tier 2, intentionally never supports std.
    // 2. `x86_64-unknown-linux-gnu`: Tier 1, always supports std.
    let config = cfg().target("x86_64-unknown-none").build();
    assert!(check_ignore(&config, "//@ needs-target-std"));
    let config = cfg().target("x86_64-unknown-linux-gnu").build();
    assert!(!check_ignore(&config, "//@ needs-target-std"));
}

fn parse_edition_range(line: &str) -> Option<EditionRange> {
    let config = cfg().build();

    let line_with_comment = format!("//@ {line}");
    let line = line_directive(Utf8Path::new("tmp.rs"), 0, &line_with_comment).unwrap();

    super::parse_edition_range(&config, &line)
}

#[test]
fn edition_order() {
    let editions = &[
        Edition::Year(2015),
        Edition::Year(2018),
        Edition::Year(2021),
        Edition::Year(2024),
        Edition::Future,
    ];
    assert!(editions.is_sorted(), "{editions:#?}");
}

#[test]
fn test_parse_edition_range() {
    assert_eq!(None, parse_edition_range("hello-world"));
    assert_eq!(None, parse_edition_range("edition"));

    assert_eq!(Some(EditionRange::Exact(2018.into())), parse_edition_range("edition: 2018"));
    assert_eq!(Some(EditionRange::Exact(2021.into())), parse_edition_range("edition:2021"));
    assert_eq!(Some(EditionRange::Exact(2024.into())), parse_edition_range("edition: 2024 "));
    assert_eq!(Some(EditionRange::Exact(Edition::Future)), parse_edition_range("edition: future"));

    assert_eq!(Some(EditionRange::RangeFrom(2018.into())), parse_edition_range("edition: 2018.."));
    assert_eq!(Some(EditionRange::RangeFrom(2021.into())), parse_edition_range("edition:2021 .."));
    assert_eq!(
        Some(EditionRange::RangeFrom(2024.into())),
        parse_edition_range("edition: 2024 .. ")
    );
    assert_eq!(
        Some(EditionRange::RangeFrom(Edition::Future)),
        parse_edition_range("edition: future.. ")
    );

    assert_eq!(
        Some(EditionRange::Range { lower_bound: 2018.into(), upper_bound: 2024.into() }),
        parse_edition_range("edition: 2018..2024")
    );
    assert_eq!(
        Some(EditionRange::Range { lower_bound: 2015.into(), upper_bound: 2021.into() }),
        parse_edition_range("edition:2015 .. 2021 ")
    );
    assert_eq!(
        Some(EditionRange::Range { lower_bound: 2021.into(), upper_bound: 2027.into() }),
        parse_edition_range("edition: 2021 .. 2027 ")
    );
    assert_eq!(
        Some(EditionRange::Range { lower_bound: 2021.into(), upper_bound: Edition::Future }),
        parse_edition_range("edition: 2021..future")
    );
}

#[test]
#[should_panic]
fn test_parse_edition_range_empty() {
    parse_edition_range("edition:");
}

#[test]
#[should_panic]
fn test_parse_edition_range_invalid_edition() {
    parse_edition_range("edition: hello");
}

#[test]
#[should_panic]
fn test_parse_edition_range_double_dots() {
    parse_edition_range("edition: ..");
}

#[test]
#[should_panic]
fn test_parse_edition_range_inverted_range() {
    parse_edition_range("edition: 2021..2015");
}

#[test]
#[should_panic]
fn test_parse_edition_range_inverted_range_future() {
    parse_edition_range("edition: future..2015");
}

#[test]
#[should_panic]
fn test_parse_edition_range_empty_range() {
    parse_edition_range("edition: 2021..2021");
}

#[track_caller]
fn assert_edition_to_test(
    expected: impl Into<Edition>,
    range: EditionRange,
    default: Option<Edition>,
) {
    let mut cfg = cfg();
    if let Some(default) = default {
        cfg.edition(default);
    }
    assert_eq!(expected.into(), range.edition_to_test(cfg.build().edition));
}

#[test]
fn test_edition_range_edition_to_test() {
    let e2015 = parse_edition("2015");
    let e2018 = parse_edition("2018");
    let e2021 = parse_edition("2021");
    let e2024 = parse_edition("2024");
    let efuture = parse_edition("future");

    let exact = EditionRange::Exact(2021.into());
    assert_edition_to_test(2021, exact, None);
    assert_edition_to_test(2021, exact, Some(e2018));
    assert_edition_to_test(2021, exact, Some(efuture));

    assert_edition_to_test(Edition::Future, EditionRange::Exact(Edition::Future), None);

    let greater_equal_than = EditionRange::RangeFrom(2021.into());
    assert_edition_to_test(2021, greater_equal_than, None);
    assert_edition_to_test(2021, greater_equal_than, Some(e2015));
    assert_edition_to_test(2021, greater_equal_than, Some(e2018));
    assert_edition_to_test(2021, greater_equal_than, Some(e2021));
    assert_edition_to_test(2024, greater_equal_than, Some(e2024));
    assert_edition_to_test(Edition::Future, greater_equal_than, Some(efuture));

    let range = EditionRange::Range { lower_bound: 2018.into(), upper_bound: 2024.into() };
    assert_edition_to_test(2018, range, None);
    assert_edition_to_test(2018, range, Some(e2015));
    assert_edition_to_test(2018, range, Some(e2018));
    assert_edition_to_test(2021, range, Some(e2021));
    assert_edition_to_test(2018, range, Some(e2024));
    assert_edition_to_test(2018, range, Some(efuture));
}
