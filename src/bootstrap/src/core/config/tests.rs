use std::env;
use std::fs::{remove_file, File};
use std::io::Write;
use std::path::Path;

use clap::CommandFactory;
use serde::Deserialize;

use super::flags::Flags;
use super::{ChangeIdWrapper, Config};
use crate::core::build_steps::clippy::get_clippy_rules_in_order;
use crate::core::config::{LldMode, Target, TargetSelection, TomlConfig};

fn parse(config: &str) -> Config {
    Config::parse_inner(
        Flags::parse(&["check".to_string(), "--config=/does/not/exist".to_string()]),
        |&_| toml::from_str(&config),
    )
}

#[test]
fn download_ci_llvm() {
    if crate::core::build_steps::llvm::is_ci_llvm_modified(&parse("")) {
        eprintln!("Detected LLVM as non-available: running in CI and modified LLVM in this change");
        return;
    }

    let parse_llvm = |s| parse(s).llvm_from_ci;
    let if_unchanged = parse_llvm("llvm.download-ci-llvm = \"if-unchanged\"");

    assert!(parse_llvm("llvm.download-ci-llvm = true"));
    assert!(!parse_llvm("llvm.download-ci-llvm = false"));
    assert_eq!(parse_llvm(""), if_unchanged);
    assert_eq!(parse_llvm("rust.channel = \"dev\""), if_unchanged);
    assert!(!parse_llvm("rust.channel = \"stable\""));
    assert_eq!(parse_llvm("build.build = \"x86_64-unknown-linux-gnu\""), if_unchanged);
    assert_eq!(
        parse_llvm(
            "llvm.assertions = true \r\n build.build = \"x86_64-unknown-linux-gnu\" \r\n llvm.download-ci-llvm = \"if-unchanged\""
        ),
        if_unchanged
    );
    assert!(!parse_llvm(
        "llvm.assertions = true \r\n build.build = \"aarch64-apple-darwin\" \r\n llvm.download-ci-llvm = \"if-unchanged\""
    ));
}

// FIXME(onur-ozkan): extend scope of the test
// refs:
//   - https://github.com/rust-lang/rust/issues/109120
//   - https://github.com/rust-lang/rust/pull/109162#issuecomment-1496782487
#[test]
fn detect_src_and_out() {
    fn test(cfg: Config, build_dir: Option<&str>) {
        // This will bring absolute form of `src/bootstrap` path
        let current_dir = std::env::current_dir().unwrap();

        // get `src` by moving into project root path
        let expected_src = current_dir.ancestors().nth(2).unwrap();
        assert_eq!(&cfg.src, expected_src);

        // Sanity check for `src`
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let expected_src = manifest_dir.ancestors().nth(2).unwrap();
        assert_eq!(&cfg.src, expected_src);

        // test if build-dir was manually given in config.toml
        if let Some(custom_build_dir) = build_dir {
            assert_eq!(&cfg.out, Path::new(custom_build_dir));
        }
        // test the native bootstrap way
        else {
            // This should bring output path of bootstrap in absolute form
            let cargo_target_dir = env::var_os("CARGO_TARGET_DIR").expect(
                "CARGO_TARGET_DIR must been provided for the test environment from bootstrap",
            );

            // Move to `build` from `build/bootstrap`
            let expected_out = Path::new(&cargo_target_dir).parent().unwrap();
            assert_eq!(&cfg.out, expected_out);

            let args: Vec<String> = env::args().collect();

            // Another test for `out` as a sanity check
            //
            // This will bring something similar to:
            //     `{build-dir}/bootstrap/debug/deps/bootstrap-c7ee91d5661e2804`
            // `{build-dir}` can be anywhere, not just in the rust project directory.
            let dep = Path::new(args.first().unwrap());
            let expected_out = dep.ancestors().nth(4).unwrap();

            assert_eq!(&cfg.out, expected_out);
        }
    }

    test(parse(""), None);

    {
        let build_dir = if cfg!(windows) { "C:\\tmp" } else { "/tmp" };
        test(parse(&format!("build.build-dir = '{build_dir}'")), Some(build_dir));
    }
}

#[test]
fn clap_verify() {
    Flags::command().debug_assert();
}

#[test]
fn override_toml() {
    let config = Config::parse_inner(
        Flags::parse(&[
            "check".to_owned(),
            "--config=/does/not/exist".to_owned(),
            "--set=change-id=1".to_owned(),
            "--set=rust.lto=fat".to_owned(),
            "--set=rust.deny-warnings=false".to_owned(),
            "--set=build.gdb=\"bar\"".to_owned(),
            "--set=build.tools=[\"cargo\"]".to_owned(),
            "--set=llvm.build-config={\"foo\" = \"bar\"}".to_owned(),
            "--set=target.x86_64-unknown-linux-gnu.runner=bar".to_owned(),
            "--set=target.x86_64-unknown-linux-gnu.rpath=false".to_owned(),
            "--set=target.aarch64-unknown-linux-gnu.sanitizers=false".to_owned(),
            "--set=target.aarch64-apple-darwin.runner=apple".to_owned(),
        ]),
        |&_| {
            toml::from_str(
                r#"
change-id = 0
[rust]
lto = "off"
deny-warnings = true

[build]
gdb = "foo"
tools = []

[llvm]
download-ci-llvm = false
build-config = {}

[target.aarch64-unknown-linux-gnu]
sanitizers = true
rpath = true
runner = "aarch64-runner"

[target.x86_64-unknown-linux-gnu]
sanitizers = true
rpath = true
runner = "x86_64-runner"

                "#,
            )
        },
    );
    assert_eq!(config.change_id, Some(1), "setting top-level value");
    assert_eq!(
        config.rust_lto,
        crate::core::config::RustcLto::Fat,
        "setting string value without quotes"
    );
    assert_eq!(config.gdb, Some("bar".into()), "setting string value with quotes");
    assert!(!config.deny_warnings, "setting boolean value");
    assert_eq!(
        config.tools,
        Some(["cargo".to_string()].into_iter().collect()),
        "setting list value"
    );
    assert_eq!(
        config.llvm_build_config,
        [("foo".to_string(), "bar".to_string())].into_iter().collect(),
        "setting dictionary value"
    );

    let x86_64 = TargetSelection::from_user("x86_64-unknown-linux-gnu");
    let x86_64_values = Target {
        sanitizers: Some(true),
        rpath: Some(false),
        runner: Some("bar".into()),
        ..Default::default()
    };
    let aarch64 = TargetSelection::from_user("aarch64-unknown-linux-gnu");
    let aarch64_values = Target {
        sanitizers: Some(false),
        rpath: Some(true),
        runner: Some("aarch64-runner".into()),
        ..Default::default()
    };
    let darwin = TargetSelection::from_user("aarch64-apple-darwin");
    let darwin_values = Target { runner: Some("apple".into()), ..Default::default() };
    assert_eq!(
        config.target_config,
        [(x86_64, x86_64_values), (aarch64, aarch64_values), (darwin, darwin_values)]
            .into_iter()
            .collect(),
        "setting dictionary value"
    );
}

#[test]
#[should_panic]
fn override_toml_duplicate() {
    Config::parse_inner(
        Flags::parse(&[
            "check".to_owned(),
            "--config=/does/not/exist".to_string(),
            "--set=change-id=1".to_owned(),
            "--set=change-id=2".to_owned(),
        ]),
        |&_| toml::from_str("change-id = 0"),
    );
}

#[test]
fn profile_user_dist() {
    fn get_toml(file: &Path) -> Result<TomlConfig, toml::de::Error> {
        let contents =
            if file.ends_with("config.toml") || env::var_os("RUST_BOOTSTRAP_CONFIG").is_some() {
                "profile = \"user\"".to_owned()
            } else {
                assert!(file.ends_with("config.dist.toml"));
                std::fs::read_to_string(file).unwrap()
            };

        toml::from_str(&contents).and_then(|table: toml::Value| TomlConfig::deserialize(table))
    }
    Config::parse_inner(Flags::parse(&["check".to_owned()]), get_toml);
}

#[test]
fn rust_optimize() {
    assert!(parse("").rust_optimize.is_release());
    assert!(!parse("rust.optimize = false").rust_optimize.is_release());
    assert!(parse("rust.optimize = true").rust_optimize.is_release());
    assert!(!parse("rust.optimize = 0").rust_optimize.is_release());
    assert!(parse("rust.optimize = 1").rust_optimize.is_release());
    assert!(parse("rust.optimize = \"s\"").rust_optimize.is_release());
    assert_eq!(parse("rust.optimize = 1").rust_optimize.get_opt_level(), Some("1".to_string()));
    assert_eq!(parse("rust.optimize = \"s\"").rust_optimize.get_opt_level(), Some("s".to_string()));
}

#[test]
#[should_panic]
fn invalid_rust_optimize() {
    parse("rust.optimize = \"a\"");
}

#[test]
fn verify_file_integrity() {
    let config = parse("");

    let tempfile = config.tempdir().join(".tmp-test-file");
    File::create(&tempfile).unwrap().write_all(b"dummy value").unwrap();
    assert!(tempfile.exists());

    assert!(
        config
            .verify(&tempfile, "7e255dd9542648a8779268a0f268b891a198e9828e860ed23f826440e786eae5")
    );

    remove_file(tempfile).unwrap();
}

#[test]
fn rust_lld() {
    assert!(matches!(parse("").lld_mode, LldMode::Unused));
    assert!(matches!(parse("rust.use-lld = \"self-contained\"").lld_mode, LldMode::SelfContained));
    assert!(matches!(parse("rust.use-lld = \"external\"").lld_mode, LldMode::External));
    assert!(matches!(parse("rust.use-lld = true").lld_mode, LldMode::External));
    assert!(matches!(parse("rust.use-lld = false").lld_mode, LldMode::Unused));
}

#[test]
#[should_panic]
fn parse_config_with_unknown_field() {
    parse("unknown-key = 1");
}

#[test]
fn parse_change_id_with_unknown_field() {
    let config = r#"
        change-id = 3461
        unknown-key = 1
    "#;

    let change_id_wrapper: ChangeIdWrapper = toml::from_str(config).unwrap();
    assert_eq!(change_id_wrapper.inner, Some(3461));
}

#[test]
fn order_of_clippy_rules() {
    let args = vec![
        "clippy".to_string(),
        "--fix".to_string(),
        "--allow-dirty".to_string(),
        "--allow-staged".to_string(),
        "-Aclippy:all".to_string(),
        "-Wclippy::style".to_string(),
        "-Aclippy::foo1".to_string(),
        "-Aclippy::foo2".to_string(),
    ];
    let config = Config::parse(Flags::parse(&args));

    let actual = match &config.cmd {
        crate::Subcommand::Clippy { allow, deny, warn, forbid, .. } => {
            get_clippy_rules_in_order(&args, &allow, &deny, &warn, &forbid)
        }
        _ => panic!("invalid subcommand"),
    };

    let expected = vec![
        "-Aclippy:all".to_string(),
        "-Wclippy::style".to_string(),
        "-Aclippy::foo1".to_string(),
        "-Aclippy::foo2".to_string(),
    ];

    assert_eq!(expected, actual);
}

#[test]
fn verbose_tests_default_value() {
    let config = Config::parse(Flags::parse(&["build".into(), "compiler".into()]));
    assert_eq!(config.verbose_tests, false);

    let config = Config::parse(Flags::parse(&["build".into(), "compiler".into(), "-v".into()]));
    assert_eq!(config.verbose_tests, true);
}
