use std::assert_matches;
use std::collections::BTreeSet;

use camino::Utf8Path;
use semver::Version;

use crate::common::{Config, Debugger, TestMode};
use crate::directives::line::{DirectiveLine, DirectivePayload, Error, ErrorKind, line_directive};
use crate::directives::{
    self, AuxProps, DIRECTIVE_HANDLERS_MAP, DirectivesCache, EarlyProps, EarlyValidationError,
    EarlyValidationErrorKind, Edition, EditionRange, FileDirectives, KNOWN_DIRECTIVE_NAMES_SET,
    LineNumber, Location, RevisionsParseError, RevisionsParseErrorKind, extract_llvm_version,
    extract_version_range, parse_edition, parse_normalize_rule,
};
use crate::executor::{CollectedTestDesc, ShouldFail, TestVariant};

/// All directive handlers should have a name that is also in `KNOWN_DIRECTIVE_NAMES_SET`.
#[test]
fn handler_names() {
    let unknown_names = DIRECTIVE_HANDLERS_MAP
        .keys()
        .copied()
        .filter(|name| !KNOWN_DIRECTIVE_NAMES_SET.contains(name))
        .collect::<BTreeSet<_>>();

    assert!(
        unknown_names.is_empty(),
        "Directive handler names not in `directive_names.rs`: {unknown_names:#?}"
    );
}

#[test]
fn external_ignores() {
    let unknown_names = directives::cfg::EXTERNAL_IGNORES_SET
        .difference(&KNOWN_DIRECTIVE_NAMES_SET)
        .into_iter()
        .collect::<BTreeSet<_>>();

    assert!(
        unknown_names.is_empty(),
        "Directive names not in `directive_names.rs`: {unknown_names:#?}"
    );
}

fn make_test_description(
    config: &Config,
    name: String,
    path: &Utf8Path,
    filterable_path: &Utf8Path,
    file_contents: &str,
    revision: Option<&str>,
    debugger: Option<Debugger>,
) -> CollectedTestDesc {
    let cache = DirectivesCache::load(config);
    let mut poisoned = false;
    let file_directives = FileDirectives::from_file_contents(path, file_contents);

    let variant = TestVariant { revision: revision.map(str::to_owned), debugger };

    let mut aux_props = AuxProps::default();
    let test = crate::directives::make_test_description(
        config,
        &cache,
        name,
        filterable_path,
        &file_directives,
        &variant,
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
        (r#"" with whitespace"   ->   "    replacement""#, " with whitespace", "    replacement"),
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
    suite: Option<String>,
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
    std_remap_debuginfo: bool,
    disable_minification: bool,
}

impl ConfigBuilder {
    fn mode(&mut self, s: &str) -> &mut Self {
        self.mode = Some(s.to_owned());
        self
    }

    fn suite(&mut self, s: &str) -> &mut Self {
        self.suite = Some(s.to_owned());
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

    fn std_remap_debuginfo(&mut self, is_enabled: bool) -> &mut Self {
        self.std_remap_debuginfo = is_enabled;
        self
    }

    fn disable_minification(&mut self, is_enabled: bool) -> &mut Self {
        self.disable_minification = is_enabled;
        self
    }

    fn build(&mut self) -> Config {
        let args = &[
            "compiletest",
            "--mode",
            self.mode.as_deref().unwrap_or("ui"),
            "--suite",
            self.suite.as_deref().unwrap_or("ui"),
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
            "--jobs=0",
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
        if self.std_remap_debuginfo {
            args.push("--with-std-remap-debuginfo".to_owned());
        }
        if self.disable_minification {
            args.push("--disable-minification".to_owned());
        }

        args.push("--rustc-path".to_string());
        args.push(std::env::var("TEST_RUSTC").expect("must be configured by bootstrap"));

        crate::cli::parse_config(args)
    }
}

fn cfg() -> ConfigBuilder {
    ConfigBuilder::default()
}

fn parse_early_props<'a>(
    config: &Config,
    contents: &'a str,
) -> Result<EarlyProps, RevisionsParseError<'a>> {
    let file_directives = FileDirectives::from_file_contents(Utf8Path::new("a.rs"), contents);
    EarlyProps::from_file_directives(config, &file_directives)
}

fn check_ignore(config: &Config, contents: &str) -> bool {
    let tn = String::new();
    let p = Utf8Path::new("a.rs");
    let d = make_test_description(&config, tn, p, p, contents, None, None);
    d.is_ignored()
}

fn check_ignore_debugger(config: &Config, contents: &str, debugger: Option<Debugger>) -> bool {
    let tn = String::new();
    let p = Utf8Path::new("a.rs");
    let d = make_test_description(&config, tn, p, p, contents, None, debugger);
    d.is_ignored()
}

#[test]
fn should_fail() {
    let config: Config = cfg().build();
    let tn = String::new();
    let p = Utf8Path::new("a.rs");

    let d = make_test_description(&config, tn.clone(), p, p, "", None, None);
    assert_eq!(d.should_fail, ShouldFail::No);
    let d = make_test_description(&config, tn, p, p, "//@ should-fail", None, None);
    assert_eq!(d.should_fail, ShouldFail::Yes);
}

#[test]
fn disable_minification_flag() {
    let config: Config = cfg().build();
    assert!(!config.disable_minification);

    let config: Config = cfg().disable_minification(true).build();
    assert!(config.disable_minification);
}

#[test]
fn revisions() {
    let config: Config = cfg().build();

    assert_eq!(
        parse_early_props(&config, "//@ revisions: a b c").map(|props| props.revisions),
        Ok(vec!["a".into(), "b".into(), "c".into()])
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
fn std_remap_debuginfo() {
    let config: Config = cfg().std_remap_debuginfo(false).build();

    assert!(check_ignore(&config, "//@ needs-std-remap-debuginfo"));
    assert!(!check_ignore(&config, "//@ ignore-std-remap-debuginfo"));

    let config: Config = cfg().std_remap_debuginfo(true).build();

    assert!(!check_ignore(&config, "//@ needs-std-remap-debuginfo"));
    assert!(check_ignore(&config, "//@ ignore-std-remap-debuginfo"));
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
    let config = cfg().build();
    assert!(!check_ignore_debugger(&config, "//@ ignore-cdb", None));
    assert!(check_ignore_debugger(&config, "//@ ignore-cdb", Some(Debugger::Cdb)));
    assert!(check_ignore_debugger(&config, "//@ ignore-gdb", Some(Debugger::Gdb)));
    assert!(check_ignore_debugger(&config, "//@ ignore-lldb", Some(Debugger::Lldb)));
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
fn test_invalid_revisions_in_declaration() {
    let config: Config = cfg().build();
    assert_eq!(
        parse_early_props(&config, "//@ revisions: ok #&$?+)^").map_err(|e| e.kind),
        Err(RevisionsParseErrorKind::InvalidRevisionName("#&$?+)^".into()))
    );
}

#[test]
fn test_duplicate_revisions_in_declaration() {
    let config: Config = cfg().build();
    assert_eq!(
        parse_early_props(&config, "//@ revisions: rpass1 rpass1").map_err(|e| e.kind),
        Err(RevisionsParseErrorKind::DuplicateRevisions("rpass1".into()))
    );
}

#[test]
fn test_assembly_mode_forbidden_revisions_in_declaration() {
    let config = cfg().mode("assembly").build();
    assert_eq!(
        parse_early_props(&config, "//@ revisions: CHECK").map_err(|e| e.kind),
        Err(RevisionsParseErrorKind::ForbiddenFileCheckRevisionName("CHECK".into()))
    );
}

#[test]
fn test_forbidden_revisions_in_declaration() {
    let config = cfg().mode("ui").build();
    assert_eq!(
        parse_early_props(&config, "//@ revisions: true").map_err(|e| e.kind),
        Err(RevisionsParseErrorKind::InvalidRevisionName("true".into()))
    );
}

#[test]
fn test_codegen_mode_forbidden_revisions_in_declaration() {
    let config = cfg().mode("codegen").build();
    assert_eq!(
        parse_early_props(&config, "//@ revisions: CHECK").map_err(|e| e.kind),
        Err(RevisionsParseErrorKind::ForbiddenFileCheckRevisionName("CHECK".into()))
    );
}

#[test]
fn test_miropt_mode_forbidden_revisions_in_declaration() {
    let config = cfg().mode("mir-opt").build();
    assert_eq!(
        parse_early_props(&config, "//@ revisions: CHECK").map_err(|e| e.kind),
        Err(RevisionsParseErrorKind::ForbiddenFileCheckRevisionName("CHECK".into()))
    );
}

#[test]
fn test_comma_separated_revisions_in_declaration() {
    let config = cfg().build();
    assert_eq!(
        parse_early_props(&config, "//@ revisions: a, b, c").map_err(|e| e.kind),
        Err(RevisionsParseErrorKind::CommaSeparated("a, b, c".into()))
    );
}

#[test]
fn test_revision_conditional_revisions_directive() {
    let config = cfg().build();
    assert_eq!(
        parse_early_props(&config, "//@[recursive] revisions: recursive end").map_err(|e| e.kind),
        Err(RevisionsParseErrorKind::Conditional("recursive".into()))
    );
}

#[test]
fn test_empty_directive() {
    let line = "//@";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error { kind: ErrorKind::MissingDirectiveName, .. }))
    );
}

#[test]
fn test_missing_directive_name() {
    let line = "//@: value";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error { kind: ErrorKind::MissingDirectiveName, .. }))
    );
}

#[test]
fn test_key_value_directive() {
    let line = "//@ aux-build: file.rs";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine {
            revision: None,
            name: "aux-build",
            payload: Some(DirectivePayload::Value("file.rs")),
            ..
        }))
    );
}

#[test]
fn test_valueless_directive() {
    let line = "//@ only-unix";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine { revision: None, name: "only-unix", payload: None, .. }))
    );
}

#[test]
fn test_non_alphanumeric_directive_name() {
    let line = "//@ #&$?+)^";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine { revision: None, name: "#&$?+)^", payload: None, .. }))
    );
}

#[test]
fn test_non_alphanumeric_directive_name_with_payload() {
    let line = "//@ #&$?+)^ payload";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine {
            revision: None,
            name: "#&$?+)^",
            payload: Some(DirectivePayload::Remark("payload")),
            ..
        }))
    );
}

#[test]
fn test_ok_conditional_directive() {
    let line = "//@[revision] directive";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine {
            revision: Some("revision"),
            name: "directive",
            payload: None,
            ..
        }))
    );
}

#[test]
fn test_indented_directive() {
    let line = "  //@ key: value";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine {
            revision: None,
            name: "key",
            payload: Some(DirectivePayload::Value("value")),
            ..
        }))
    );
}

#[test]
fn test_directive_padded_colon() {
    // Historical context: For the longest time we used to parse this as a valueless directive with
    // remark ` : value` since we were naively splitting by [':', ' '].

    let line = "//@ key : value";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine {
            revision: None,
            name: "key",
            payload: Some(DirectivePayload::Value("value")),
            ..
        }))
    );
}

#[test]
fn test_directive_two_names() {
    // We parse the second half as a remark so we can later reject it as a trailing directive if
    // `key1` turns out to be the name of a real directive.

    let line = "//@ key0 key1: value";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine {
            revision: None,
            name: "key0",
            payload: Some(DirectivePayload::Remark("key1: value")),
            ..
        }))
    );
}

#[test]
fn test_key_value_directive_trailing_whitespace() {
    let line = "//@ key: value  \t ";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine {
            revision: None,
            name: "key",
            payload: Some(DirectivePayload::Value("value")),
            ..
        }))
    );
}

#[test]
fn test_valueless_directive_with_trailing_whitespace() {
    let line = "//@ key   \t ";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine { revision: None, name: "key", payload: None, .. }))
    );
}

#[test]
fn test_directive_empty_value_0() {
    let line = "//@ key:";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error { kind: ErrorKind::MissingValue, .. }))
    );
}

#[test]
fn test_directive_empty_value_1() {
    let line = "//@ key:   \t ";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error { kind: ErrorKind::MissingValue, .. }))
    );
}

#[test]
fn test_doc_comment_at_sign_not_a_directive() {
    let line = "///@ aux-build: file.rs";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(line_directive(location, line), None);
}

#[test]
fn test_directive_padded_revision_condition() {
    let line = "//@ [ rev ] key";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Ok(DirectiveLine { revision: Some("rev"), name: "key", payload: None, .. }))
    );
}

#[test]
fn test_directive_empty_revision_condition() {
    let line = "//@ [] key";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error { kind: ErrorKind::InvalidRevisionName { revision: "" }, .. }))
    );
}

#[test]
fn test_directive_invalid_revision_name_in_condition() {
    let line = "//@ [ #&$?+)^ ] key";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error { kind: ErrorKind::InvalidRevisionName { revision: "#&$?+)^" }, .. }))
    );
}

#[test]
fn test_directive_multiple_revisions_in_condition() {
    // The problem: If we don't reject this early, this will be seen as a single revision.
    let line = "//@ [foo,bar] compile-flags: -Z hello";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error {
            kind: ErrorKind::CommaInRevisionCondition {
                revision: "foo,bar",
                directive: "compile-flags: -Z hello"
            },
            ..
        }))
    );
}

#[test]
fn test_whitespace_before_at_sign_in_directive() {
    let line = "// @ check-pass";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error { kind: ErrorKind::WhitespaceBeforeAtSign, .. }))
    );
}

#[test]
fn test_leading_revision_condition_in_directive_0() {
    let line = "//[leading]@ compile-flags: -Aunused";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error {
            kind: ErrorKind::LeadingRevisionCondition {
                revision: "leading",
                context: "//[leading]@ "
            },
            ..
        }))
    );
}

#[test]
fn test_leading_revision_condition_in_directive_1() {
    let line = "//  [leading]  @ compile-flags: -Aunused";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error {
            kind: ErrorKind::LeadingRevisionCondition {
                revision: "leading",
                context: "//  [leading]  @ "
            },
            ..
        }))
    );
}

#[test]
fn test_not_a_leading_revision_condition_in_directive_0() {
    let line = "//  [leading  @ compile-flags: -Aunused";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(line_directive(location, line), None);
}

#[test]
fn test_not_a_leading_revision_condition_in_directive_1() {
    let line = "// [closure@file.rs]";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(line_directive(location, line), None);
}

#[test]
fn test_unclosed_revision_condition_in_directive() {
    let line = "//@[revision edition: 2021";

    let location = Location { file_path: Utf8Path::new("file.rs"), line_number: LineNumber::ZERO };
    assert_matches!(
        line_directive(location, line),
        Some(Err(Error { kind: ErrorKind::UnclosedRevisionCondition, .. }))
    );
}

#[test]
fn test_forbidden_revisions_allowed_in_non_filecheck_dir() {
    let revisions = ["CHECK", "COM", "NEXT", "SAME", "EMPTY", "NOT", "COUNT", "DAG", "LABEL"];
    let modes = [
        "pretty",
        "debuginfo",
        "rustdoc-html",
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
            parse_early_props(&config, &content).unwrap();
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
        ("wasm32-unknown-unknown", "emscripten", false),
        ("wasm32-unknown-unknown", "wasm32", true),
        ("wasm32-unknown-unknown", "wasm64", false),
        ("wasm32-unknown-emscripten", "emscripten", true),
        ("wasm32-unknown-emscripten", "wasm32", true),
        ("wasm32-wasip1", "emscripten", false),
        ("wasm32-wasip1", "wasm32", true),
        ("wasm32-wasip1", "wasi", true),
        ("wasm64-unknown-unknown", "emscripten", false),
        ("wasm64-unknown-unknown", "wasm32", false),
        ("wasm64-unknown-unknown", "wasm64", true),
    ];
    for (target, pattern, ignore) in ignores {
        let config: Config = cfg().target(target).build();
        assert_eq!(
            check_ignore(&config, &format!("//@ ignore-{pattern}")),
            ignore,
            "target `{target}` vs `//@ ignore-{pattern}`"
        );
        assert_eq!(
            check_ignore(&config, &format!("//@ only-{pattern}")),
            !ignore,
            "target `{target}` vs `//@ only-{pattern}`"
        )
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
fn only_ignore_elf() {
    let cases = &[
        ("aarch64-apple-darwin", false),
        ("aarch64-unknown-linux-gnu", true),
        ("powerpc64-ibm-aix", false),
        ("wasm32-unknown-unknown", false),
        ("wasm32-wasip1", false),
        ("x86_64-apple-darwin", false),
        ("x86_64-pc-windows-msvc", false),
        ("x86_64-unknown-freebsd", true),
        ("x86_64-unknown-illumos", true),
        ("x86_64-unknown-linux-gnu", true),
        ("x86_64-unknown-none", true),
        ("x86_64-unknown-uefi", false),
    ];
    for &(target, is_elf) in cases {
        let config = &cfg().target(target).build();
        assert_eq!(is_elf, check_ignore(config, "//@ ignore-elf"), "`//@ ignore-elf` for {target}");
        assert_eq!(is_elf, !check_ignore(config, "//@ only-elf"), "`//@ only-elf` for {target}");
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

fn parse_and_early_validate_directives<'a>(
    path: &'a Utf8Path,
    file_contents: &'a str,
) -> Result<Vec<DirectiveLine<'a>>, EarlyValidationError<'a>> {
    let file_directives = FileDirectives::from_file_contents(path, file_contents);
    directives::do_early_directives_check(TestMode::Ui, &file_directives)?;
    Ok(file_directives.lines)
}

#[test]
fn test_unknown_directive_check() {
    let result = parse_and_early_validate_directives(
        Utf8Path::new("a.rs"),
        include_str!("./test-auxillary/unknown_directive.rs"),
    );
    assert_matches!(
        result,
        Err(EarlyValidationError {
            kind: EarlyValidationErrorKind::UnknownDirective { name: "needs-headpat" },
            ..
        })
    );
}

#[test]
fn test_known_directive_check_no_error() {
    let result = parse_and_early_validate_directives(
        Utf8Path::new("a.rs"),
        include_str!("./test-auxillary/known_directive.rs"),
    );
    assert_matches!(
        result.as_deref(),
        Ok([
            DirectiveLine { revision: None, name: "ignore-wasm", payload: None, .. },
            DirectiveLine { revision: None, name: "check-pass", payload: None, .. },
        ])
    );
}

#[test]
fn test_error_annotation_no_error() {
    let result = parse_and_early_validate_directives(
        Utf8Path::new("a.rs"),
        include_str!("./test-auxillary/error_annotation.rs"),
    );
    assert_matches!(
        result.as_deref(),
        Ok([DirectiveLine { revision: None, name: "check-pass", payload: None, .. },])
    );
}

#[test]
fn test_non_rs_unknown_directive_not_checked() {
    let result = parse_and_early_validate_directives(
        Utf8Path::new("a.Makefile"),
        include_str!("./test-auxillary/not_rs.Makefile"),
    );
    assert_matches!(result.as_deref(), Ok([]));
}

#[test]
fn test_trailing_directive() {
    let result =
        parse_and_early_validate_directives(Utf8Path::new("a.rs"), "//@ only-x86 only-arm");
    assert_matches!(
        result,
        Err(EarlyValidationError {
            kind: EarlyValidationErrorKind::TrailingDirective {
                leading_directive: "only-x86",
                trailing_directive: "only-arm",
            },
            ..
        })
    );
}

#[test]
fn test_trailing_directive_with_comment() {
    let result = parse_and_early_validate_directives(
        Utf8Path::new("a.rs"),
        "//@ only-x86   only-arm with comment",
    );
    assert_matches!(
        result,
        Err(EarlyValidationError {
            kind: EarlyValidationErrorKind::TrailingDirective {
                leading_directive: "only-x86",
                trailing_directive: "only-arm with comment",
            },
            ..
        })
    );
}

#[test]
fn test_trailing_directive_with_colon() {
    let result =
        parse_and_early_validate_directives(Utf8Path::new("a.rs"), "//@ check-pass edition: 2015");
    assert_matches!(
        result,
        Err(EarlyValidationError {
            kind: EarlyValidationErrorKind::TrailingDirective {
                leading_directive: "check-pass",
                trailing_directive: "edition: 2015",
            },
            ..
        })
    );
}

#[test]
fn test_not_trailing_directive() {
    let result =
        parse_and_early_validate_directives(Utf8Path::new("a.rs"), "//@ revisions: incremental");
    assert_matches!(
        result.as_deref(),
        Ok([DirectiveLine {
            revision: None,
            name: "revisions",
            payload: Some(DirectivePayload::Value("incremental")),
            ..
        }])
    );
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

#[test]
fn implied_needs_target_std() {
    let config = cfg().mode("codegen").suite("codegen-llvm").target("x86_64-unknown-none").build();
    // Implied `needs-target-std` due to no `#![no_std]`/`#![no_core]`.
    assert!(check_ignore(&config, ""));
    assert!(check_ignore(&config, "//@ needs-target-std"));
    assert!(!check_ignore(&config, "#![no_std]"));
    assert!(!check_ignore(&config, "#![no_core]"));
    // Make sure that `//@ needs-target-std` takes precedence.
    assert!(check_ignore(
        &config,
        r#"
        //@ needs-target-std
        #![no_std]
        "#
    ));
    assert!(check_ignore(
        &config,
        r#"
        //@ needs-target-std
        #![no_core]
        "#
    ));

    let config =
        cfg().mode("codegen").suite("codegen-llvm").target("x86_64-unknown-linux-gnu").build();
    assert!(!check_ignore(&config, ""));
    assert!(!check_ignore(&config, "//@ needs-target-std"));
    assert!(!check_ignore(&config, "#![no_std]"));
    assert!(!check_ignore(&config, "#![no_core]"));
    assert!(!check_ignore(
        &config,
        r#"
        //@ needs-target-std
        #![no_std]
        "#
    ));
    assert!(!check_ignore(
        &config,
        r#"
        //@ needs-target-std
        #![no_core]
        "#
    ));

    let config = cfg().mode("ui").suite("ui").target("x86_64-unknown-none").build();
    // The implied `//@ needs-target-std` is only applicable for mode=codegen tests.
    assert!(!check_ignore(&config, ""));
    assert!(check_ignore(&config, "//@ needs-target-std"));
    assert!(!check_ignore(&config, "#![no_std]"));
    assert!(!check_ignore(&config, "#![no_core]"));
    assert!(check_ignore(
        &config,
        r#"
        //@ needs-target-std
        #![no_std]
        "#
    ));
    assert!(check_ignore(
        &config,
        r#"
        //@ needs-target-std
        #![no_core]
        "#
    ));

    let config = cfg().mode("ui").suite("ui").target("x86_64-unknown-linux-gnu").build();
    assert!(!check_ignore(&config, ""));
    assert!(!check_ignore(&config, "//@ needs-target-std"));
    assert!(!check_ignore(&config, "#![no_std]"));
    assert!(!check_ignore(&config, "#![no_core]"));
    assert!(!check_ignore(
        &config,
        r#"
        //@ needs-target-std
        #![no_std]
        "#
    ));
    assert!(!check_ignore(
        &config,
        r#"
        //@ needs-target-std
        #![no_core]
        "#
    ));
}

fn parse_edition_range(line: &str) -> Option<EditionRange> {
    let config = cfg().build();

    let location = Location { file_path: Utf8Path::new("tmp.rs"), line_number: LineNumber::ZERO };
    let line_with_comment = format!("//@ {line}");
    let line = line_directive(location, &line_with_comment).unwrap().unwrap();

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

#[test]
fn needs_asm_ret() {
    let config_x86_64 = cfg().target("x86_64-unknown-linux-gnu").build();
    let config_aarch64 = cfg().target("aarch64-unknown-linux-gnu").build();
    // 32-bit ARM does not have a "ret" mnemonic.
    let config_arm32 = cfg().target("armv7a-none-eabi").build();
    let config_wasm = cfg().target("wasm32v1-none").build();

    assert!(!check_ignore(&config_x86_64, "//@ needs-asm-ret"));
    assert!(!check_ignore(&config_aarch64, "//@ needs-asm-ret"));
    assert!(check_ignore(&config_arm32, "//@ needs-asm-ret"));
    assert!(check_ignore(&config_wasm, "//@ needs-asm-ret"));
}
