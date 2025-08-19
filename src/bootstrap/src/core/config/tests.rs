use std::collections::BTreeSet;
use std::fs::{File, remove_file};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{env, fs};

use build_helper::ci::CiEnv;
use build_helper::git::PathFreshness;
use clap::CommandFactory;
use serde::Deserialize;

use super::flags::Flags;
use super::toml::change_id::ChangeIdWrapper;
use super::{Config, RUSTC_IF_UNCHANGED_ALLOWED_PATHS};
use crate::ChangeId;
use crate::core::build_steps::clippy::{LintConfig, get_clippy_rules_in_order};
use crate::core::build_steps::llvm;
use crate::core::build_steps::llvm::LLVM_INVALIDATION_PATHS;
use crate::core::config::toml::TomlConfig;
use crate::core::config::{LldMode, Target, TargetSelection};
use crate::utils::tests::git::git_test;

pub(crate) fn parse(config: &str) -> Config {
    Config::parse_inner(
        Flags::parse(&["check".to_string(), "--config=/does/not/exist".to_string()]),
        |&_| toml::from_str(&config),
    )
}

fn get_toml(file: &Path) -> Result<TomlConfig, toml::de::Error> {
    let contents = std::fs::read_to_string(file).unwrap();
    toml::from_str(&contents).and_then(|table: toml::Value| TomlConfig::deserialize(table))
}

/// Helps with debugging by using consistent test-specific directories instead of
/// random temporary directories.
fn prepare_test_specific_dir() -> PathBuf {
    let current = std::thread::current();
    // Replace "::" with "_" to make it safe for directory names on Windows systems
    let test_path = current.name().unwrap().replace("::", "_");

    let testdir = parse("").tempdir().join(test_path);

    // clean up any old test files
    let _ = fs::remove_dir_all(&testdir);
    let _ = fs::create_dir_all(&testdir);

    testdir
}

#[test]
fn download_ci_llvm() {
    let config = parse("llvm.download-ci-llvm = false");
    assert!(!config.llvm_from_ci);

    let if_unchanged_config = parse("llvm.download-ci-llvm = \"if-unchanged\"");
    if if_unchanged_config.llvm_from_ci && if_unchanged_config.is_running_on_ci {
        let has_changes = if_unchanged_config.has_changes_from_upstream(LLVM_INVALIDATION_PATHS);

        assert!(
            !has_changes,
            "CI LLVM can't be enabled with 'if-unchanged' while there are changes in LLVM submodule."
        );
    }
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

        // test if build-dir was manually given in bootstrap.toml
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
            let expected_out = dep.ancestors().nth(5).unwrap();

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
            "--set=build.optimized-compiler-builtins=true".to_owned(),
            "--set=build.gdb=\"bar\"".to_owned(),
            "--set=build.tools=[\"cargo\"]".to_owned(),
            "--set=llvm.build-config={\"foo\" = \"bar\"}".to_owned(),
            "--set=target.x86_64-unknown-linux-gnu.runner=bar".to_owned(),
            "--set=target.x86_64-unknown-linux-gnu.rpath=false".to_owned(),
            "--set=target.aarch64-unknown-linux-gnu.sanitizers=false".to_owned(),
            "--set=target.aarch64-apple-darwin.runner=apple".to_owned(),
            "--set=target.aarch64-apple-darwin.optimized-compiler-builtins=false".to_owned(),
        ]),
        |&_| {
            toml::from_str(
                r#"
change-id = 0
[rust]
lto = "off"
deny-warnings = true
download-rustc=false

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
    assert_eq!(config.change_id, Some(ChangeId::Id(1)), "setting top-level value");
    assert_eq!(
        config.rust_lto,
        crate::core::config::RustcLto::Fat,
        "setting string value without quotes"
    );
    assert_eq!(config.gdb, Some("bar".into()), "setting string value with quotes");
    assert!(!config.deny_warnings, "setting boolean value");
    assert!(config.optimized_compiler_builtins, "setting boolean value");
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
    let darwin_values = Target {
        runner: Some("apple".into()),
        optimized_compiler_builtins: Some(false),
        ..Default::default()
    };
    assert_eq!(
        config.target_config,
        [(x86_64, x86_64_values), (aarch64, aarch64_values), (darwin, darwin_values)]
            .into_iter()
            .collect(),
        "setting dictionary value"
    );
    assert!(!config.llvm_from_ci);
    assert!(!config.download_rustc());
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
        let contents = if file.ends_with("bootstrap.toml")
            || file.ends_with("config.toml")
            || env::var_os("RUST_BOOTSTRAP_CONFIG").is_some()
        {
            "profile = \"user\"".to_owned()
        } else {
            assert!(file.ends_with("config.dist.toml") || file.ends_with("bootstrap.dist.toml"));
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
    assert_eq!(change_id_wrapper.inner, Some(ChangeId::Id(3461)));
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

    let actual = match config.cmd.clone() {
        crate::Subcommand::Clippy { allow, deny, warn, forbid, .. } => {
            let cfg = LintConfig { allow, deny, warn, forbid };
            get_clippy_rules_in_order(&args, &cfg)
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
fn clippy_rule_separate_prefix() {
    let args =
        vec!["clippy".to_string(), "-A clippy:all".to_string(), "-W clippy::style".to_string()];
    let config = Config::parse(Flags::parse(&args));

    let actual = match config.cmd.clone() {
        crate::Subcommand::Clippy { allow, deny, warn, forbid, .. } => {
            let cfg = LintConfig { allow, deny, warn, forbid };
            get_clippy_rules_in_order(&args, &cfg)
        }
        _ => panic!("invalid subcommand"),
    };

    let expected = vec!["-A clippy:all".to_string(), "-W clippy::style".to_string()];
    assert_eq!(expected, actual);
}

#[test]
fn verbose_tests_default_value() {
    let config = Config::parse(Flags::parse(&["build".into(), "compiler".into()]));
    assert_eq!(config.verbose_tests, false);

    let config = Config::parse(Flags::parse(&["build".into(), "compiler".into(), "-v".into()]));
    assert_eq!(config.verbose_tests, true);
}

#[test]
fn parse_rust_std_features() {
    let config = parse("rust.std-features = [\"panic-unwind\", \"backtrace\"]");
    let expected_features: BTreeSet<String> =
        ["panic-unwind", "backtrace"].into_iter().map(|s| s.to_string()).collect();
    assert_eq!(config.rust_std_features, expected_features);
}

#[test]
fn parse_rust_std_features_empty() {
    let config = parse("rust.std-features = []");
    let expected_features: BTreeSet<String> = BTreeSet::new();
    assert_eq!(config.rust_std_features, expected_features);
}

#[test]
#[should_panic]
fn parse_rust_std_features_invalid() {
    parse("rust.std-features = \"backtrace\"");
}

#[test]
fn parse_jobs() {
    assert_eq!(parse("build.jobs = 1").jobs, Some(1));
}

#[test]
fn jobs_precedence() {
    // `--jobs` should take precedence over using `--set build.jobs`.

    let config = Config::parse_inner(
        Flags::parse(&[
            "check".to_owned(),
            "--config=/does/not/exist".to_owned(),
            "--jobs=67890".to_owned(),
            "--set=build.jobs=12345".to_owned(),
        ]),
        |&_| toml::from_str(""),
    );
    assert_eq!(config.jobs, Some(67890));

    // `--set build.jobs` should take precedence over `bootstrap.toml`.
    let config = Config::parse_inner(
        Flags::parse(&[
            "check".to_owned(),
            "--config=/does/not/exist".to_owned(),
            "--set=build.jobs=12345".to_owned(),
        ]),
        |&_| {
            toml::from_str(
                r#"
            [build]
            jobs = 67890
        "#,
            )
        },
    );
    assert_eq!(config.jobs, Some(12345));

    // `--jobs` > `--set build.jobs` > `bootstrap.toml`
    let config = Config::parse_inner(
        Flags::parse(&[
            "check".to_owned(),
            "--jobs=123".to_owned(),
            "--config=/does/not/exist".to_owned(),
            "--set=build.jobs=456".to_owned(),
        ]),
        |&_| {
            toml::from_str(
                r#"
            [build]
            jobs = 789
        "#,
            )
        },
    );
    assert_eq!(config.jobs, Some(123));
}

#[test]
fn check_rustc_if_unchanged_paths() {
    let config = parse("");
    let normalised_allowed_paths: Vec<_> = RUSTC_IF_UNCHANGED_ALLOWED_PATHS
        .iter()
        .map(|t| {
            t.strip_prefix(":!").expect(&format!("{t} doesn't have ':!' prefix, but it should."))
        })
        .collect();

    for p in normalised_allowed_paths {
        assert!(config.src.join(p).exists(), "{p} doesn't exist.");
    }
}

#[test]
fn test_explicit_stage() {
    let config = Config::parse_inner(
        Flags::parse(&["check".to_owned(), "--config=/does/not/exist".to_owned()]),
        |&_| {
            toml::from_str(
                r#"
            [build]
            test-stage = 1
        "#,
            )
        },
    );

    assert!(!config.explicit_stage_from_cli);
    assert!(config.explicit_stage_from_config);
    assert!(config.is_explicit_stage());

    let config = Config::parse_inner(
        Flags::parse(&[
            "check".to_owned(),
            "--stage=2".to_owned(),
            "--config=/does/not/exist".to_owned(),
        ]),
        |&_| toml::from_str(""),
    );

    assert!(config.explicit_stage_from_cli);
    assert!(!config.explicit_stage_from_config);
    assert!(config.is_explicit_stage());

    let config = Config::parse_inner(
        Flags::parse(&[
            "check".to_owned(),
            "--stage=2".to_owned(),
            "--config=/does/not/exist".to_owned(),
        ]),
        |&_| {
            toml::from_str(
                r#"
            [build]
            test-stage = 1
        "#,
            )
        },
    );

    assert!(config.explicit_stage_from_cli);
    assert!(config.explicit_stage_from_config);
    assert!(config.is_explicit_stage());

    let config = Config::parse_inner(
        Flags::parse(&["check".to_owned(), "--config=/does/not/exist".to_owned()]),
        |&_| toml::from_str(""),
    );

    assert!(!config.explicit_stage_from_cli);
    assert!(!config.explicit_stage_from_config);
    assert!(!config.is_explicit_stage());
}

#[test]
fn test_exclude() {
    let exclude_path = "compiler";
    let config = parse(&format!("build.exclude=[\"{}\"]", exclude_path));

    let first_excluded = config
        .skip
        .first()
        .expect("Expected at least one excluded path")
        .to_str()
        .expect("Failed to convert excluded path to string");

    assert_eq!(first_excluded, exclude_path);
}

#[test]
fn test_ci_flag() {
    let config = Config::parse_inner(Flags::parse(&["check".into(), "--ci=false".into()]), |&_| {
        toml::from_str("")
    });
    assert!(!config.is_running_on_ci);

    let config = Config::parse_inner(Flags::parse(&["check".into(), "--ci=true".into()]), |&_| {
        toml::from_str("")
    });
    assert!(config.is_running_on_ci);

    let config = Config::parse_inner(Flags::parse(&["check".into()]), |&_| toml::from_str(""));
    assert_eq!(config.is_running_on_ci, CiEnv::is_ci());
}

#[test]
fn test_precedence_of_includes() {
    let testdir = prepare_test_specific_dir();

    let root_config = testdir.join("config.toml");
    let root_config_content = br#"
        include = ["./extension.toml"]

        [llvm]
        link-jobs = 2
    "#;
    File::create(&root_config).unwrap().write_all(root_config_content).unwrap();

    let extension = testdir.join("extension.toml");
    let extension_content = br#"
        change-id=543
        include = ["./extension2.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let extension = testdir.join("extension2.toml");
    let extension_content = br#"
        change-id=742

        [llvm]
        link-jobs = 10

        [build]
        description = "Some creative description"
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let config = Config::parse_inner(
        Flags::parse(&["check".to_owned(), format!("--config={}", root_config.to_str().unwrap())]),
        get_toml,
    );

    assert_eq!(config.change_id.unwrap(), ChangeId::Id(543));
    assert_eq!(config.llvm_link_jobs.unwrap(), 2);
    assert_eq!(config.description.unwrap(), "Some creative description");
}

#[test]
#[should_panic(expected = "Cyclic inclusion detected")]
fn test_cyclic_include_direct() {
    let testdir = prepare_test_specific_dir();

    let root_config = testdir.join("config.toml");
    let root_config_content = br#"
        include = ["./extension.toml"]
    "#;
    File::create(&root_config).unwrap().write_all(root_config_content).unwrap();

    let extension = testdir.join("extension.toml");
    let extension_content = br#"
        include = ["./config.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let config = Config::parse_inner(
        Flags::parse(&["check".to_owned(), format!("--config={}", root_config.to_str().unwrap())]),
        get_toml,
    );
}

#[test]
#[should_panic(expected = "Cyclic inclusion detected")]
fn test_cyclic_include_indirect() {
    let testdir = prepare_test_specific_dir();

    let root_config = testdir.join("config.toml");
    let root_config_content = br#"
        include = ["./extension.toml"]
    "#;
    File::create(&root_config).unwrap().write_all(root_config_content).unwrap();

    let extension = testdir.join("extension.toml");
    let extension_content = br#"
        include = ["./extension2.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let extension = testdir.join("extension2.toml");
    let extension_content = br#"
        include = ["./extension3.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let extension = testdir.join("extension3.toml");
    let extension_content = br#"
        include = ["./extension.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let config = Config::parse_inner(
        Flags::parse(&["check".to_owned(), format!("--config={}", root_config.to_str().unwrap())]),
        get_toml,
    );
}

#[test]
fn test_include_absolute_paths() {
    let testdir = prepare_test_specific_dir();

    let extension = testdir.join("extension.toml");
    File::create(&extension).unwrap().write_all(&[]).unwrap();

    let root_config = testdir.join("config.toml");
    let extension_absolute_path =
        extension.canonicalize().unwrap().to_str().unwrap().replace('\\', r"\\");
    let root_config_content = format!(r#"include = ["{}"]"#, extension_absolute_path);
    File::create(&root_config).unwrap().write_all(root_config_content.as_bytes()).unwrap();

    let config = Config::parse_inner(
        Flags::parse(&["check".to_owned(), format!("--config={}", root_config.to_str().unwrap())]),
        get_toml,
    );
}

#[test]
fn test_include_relative_paths() {
    let testdir = prepare_test_specific_dir();

    let _ = fs::create_dir_all(&testdir.join("subdir/another_subdir"));

    let root_config = testdir.join("config.toml");
    let root_config_content = br#"
        include = ["./subdir/extension.toml"]
    "#;
    File::create(&root_config).unwrap().write_all(root_config_content).unwrap();

    let extension = testdir.join("subdir/extension.toml");
    let extension_content = br#"
        include = ["../extension2.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let extension = testdir.join("extension2.toml");
    let extension_content = br#"
        include = ["./subdir/another_subdir/extension3.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let extension = testdir.join("subdir/another_subdir/extension3.toml");
    let extension_content = br#"
        include = ["../../extension4.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let extension = testdir.join("extension4.toml");
    File::create(extension).unwrap().write_all(&[]).unwrap();

    let config = Config::parse_inner(
        Flags::parse(&["check".to_owned(), format!("--config={}", root_config.to_str().unwrap())]),
        get_toml,
    );
}

#[test]
fn test_include_precedence_over_profile() {
    let testdir = prepare_test_specific_dir();

    let root_config = testdir.join("config.toml");
    let root_config_content = br#"
        profile = "dist"
        include = ["./extension.toml"]
    "#;
    File::create(&root_config).unwrap().write_all(root_config_content).unwrap();

    let extension = testdir.join("extension.toml");
    let extension_content = br#"
        [rust]
        channel = "dev"
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let config = Config::parse_inner(
        Flags::parse(&["check".to_owned(), format!("--config={}", root_config.to_str().unwrap())]),
        get_toml,
    );

    // "dist" profile would normally set the channel to "auto-detect", but includes should
    // override profile settings, so we expect this to be "dev" here.
    assert_eq!(config.channel, "dev");
}

#[test]
fn test_pr_ci_unchanged_anywhere() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_nonupstream_merge(&["b"]);
        let src = ctx.check_modifications(&["c"], CiEnv::GitHubActions);
        assert_eq!(src, PathFreshness::LastModifiedUpstream { upstream: sha });
    });
}

#[test]
fn test_pr_ci_changed_in_pr() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_nonupstream_merge(&["b"]);
        let src = ctx.check_modifications(&["b"], CiEnv::GitHubActions);
        assert_eq!(src, PathFreshness::HasLocalModifications { upstream: sha });
    });
}

#[test]
fn test_auto_ci_unchanged_anywhere_select_parent() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_upstream_merge(&["b"]);
        let src = ctx.check_modifications(&["c"], CiEnv::GitHubActions);
        assert_eq!(src, PathFreshness::LastModifiedUpstream { upstream: sha });
    });
}

#[test]
fn test_auto_ci_changed_in_pr() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_upstream_merge(&["b", "c"]);
        let src = ctx.check_modifications(&["c", "d"], CiEnv::GitHubActions);
        assert_eq!(src, PathFreshness::HasLocalModifications { upstream: sha });
    });
}

#[test]
fn test_local_uncommitted_modifications() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_branch("feature");
        ctx.modify("a");

        assert_eq!(
            ctx.check_modifications(&["a", "d"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream: sha }
        );
    });
}

#[test]
fn test_local_committed_modifications() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_upstream_merge(&["b", "c"]);
        ctx.create_branch("feature");
        ctx.modify("x");
        ctx.commit();
        ctx.modify("a");
        ctx.commit();

        assert_eq!(
            ctx.check_modifications(&["a", "d"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream: sha }
        );
    });
}

#[test]
fn test_local_committed_modifications_subdirectory() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a/b/c"]);
        ctx.create_upstream_merge(&["b", "c"]);
        ctx.create_branch("feature");
        ctx.modify("a/b/d");
        ctx.commit();

        assert_eq!(
            ctx.check_modifications(&["a/b"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream: sha }
        );
    });
}

#[test]
fn test_local_changes_in_head_upstream() {
    git_test(|ctx| {
        // We want to resolve to the upstream commit that made modifications to a,
        // even if it is currently HEAD
        let sha = ctx.create_upstream_merge(&["a"]);
        assert_eq!(
            ctx.check_modifications(&["a", "d"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: sha }
        );
    });
}

#[test]
fn test_local_changes_in_previous_upstream() {
    git_test(|ctx| {
        // We want to resolve to this commit, which modified a
        let sha = ctx.create_upstream_merge(&["a", "e"]);
        // Not to this commit, which is the latest upstream commit
        ctx.create_upstream_merge(&["b", "c"]);
        ctx.create_branch("feature");
        ctx.modify("d");
        ctx.commit();

        assert_eq!(
            ctx.check_modifications(&["a"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: sha }
        );
    });
}

#[test]
fn test_local_no_upstream_commit_with_changes() {
    git_test(|ctx| {
        ctx.create_upstream_merge(&["a", "e"]);
        ctx.create_upstream_merge(&["a", "e"]);
        // We want to fall back to this commit, because there are no commits
        // that modified `x`.
        let sha = ctx.create_upstream_merge(&["a", "e"]);
        ctx.create_branch("feature");
        ctx.modify("d");
        ctx.commit();
        assert_eq!(
            ctx.check_modifications(&["x"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: sha }
        );
    });
}

#[test]
fn test_local_no_upstream_commit() {
    git_test(|ctx| {
        let src = ctx.check_modifications(&["c", "d"], CiEnv::None);
        assert_eq!(src, PathFreshness::MissingUpstream);
    });
}

#[test]
fn test_local_changes_negative_path() {
    git_test(|ctx| {
        let upstream = ctx.create_upstream_merge(&["a"]);
        ctx.create_branch("feature");
        ctx.modify("b");
        ctx.modify("d");
        ctx.commit();

        assert_eq!(
            ctx.check_modifications(&[":!b", ":!d"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: upstream.clone() }
        );
        assert_eq!(
            ctx.check_modifications(&[":!c"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream: upstream.clone() }
        );
        assert_eq!(
            ctx.check_modifications(&[":!d", ":!x"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream }
        );
    });
}

#[test]
fn test_local_changes_subtree_that_used_bors() {
    // Here we simulate a very specific situation related to subtrees.
    // When you have merge commits locally, we should ignore them w.r.t. the artifact download
    // logic.
    // The upstream search code currently uses a simple heuristic:
    // - Find commits by bors (or in general an author with the merge commit e-mail)
    // - Find the newest such commit
    // This should make it work even for subtrees that:
    // - Used bors in the past (so they have bors merge commits in their history).
    // - Use Josh to merge rustc into the subtree, in a way that the rustc history is the second
    // parent, not the first one.
    //
    // In addition, when searching for modified files, we cannot simply start from HEAD, because
    // in this situation git wouldn't find the right commit.
    //
    // This test checks that this specific scenario will resolve to the right rustc commit, both
    // when finding a modified file and when finding a non-existent file (which essentially means
    // that we just lookup the most recent upstream commit).
    //
    // See https://github.com/rust-lang/rust/issues/101907#issuecomment-2697671282 for more details.
    git_test(|ctx| {
        ctx.create_upstream_merge(&["a"]);

        // Start unrelated subtree history
        ctx.run_git(&["switch", "--orphan", "subtree"]);
        ctx.modify("bar");
        ctx.commit();
        // Now we need to emulate old bors commits in the subtree.
        // Git only has a resolution of one second, which is a problem, since our git logic orders
        // merge commits by their date.
        // To avoid sleeping in the test, we modify the commit date to be forcefully in the past.
        ctx.create_upstream_merge(&["subtree/a"]);
        ctx.run_git(&["commit", "--amend", "--date", "Wed Feb 16 14:00 2011 +0100", "--no-edit"]);

        // Merge the subtree history into rustc
        ctx.switch_to_branch("main");
        ctx.run_git(&["merge", "subtree", "--allow-unrelated"]);

        // Create a rustc commit that modifies a path that we're interested in (`x`)
        let upstream_1 = ctx.create_upstream_merge(&["x"]);
        // Create another bors commit
        let upstream_2 = ctx.create_upstream_merge(&["a"]);

        ctx.switch_to_branch("subtree");

        // Create a subtree branch
        ctx.create_branch("subtree-pr");
        ctx.modify("baz");
        ctx.commit();
        // We merge rustc into this branch (simulating a "subtree pull")
        ctx.merge("main", "committer <committer@foo.bar>");

        // And then merge that branch into the subtree (simulating a situation right before a
        // "subtree push")
        ctx.switch_to_branch("subtree");
        ctx.merge("subtree-pr", "committer <committer@foo.bar>");

        // And we want to check that we resolve to the right commits.
        assert_eq!(
            ctx.check_modifications(&["x"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: upstream_1 }
        );
        assert_eq!(
            ctx.check_modifications(&["nonexistent"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: upstream_2 }
        );
    });
}
