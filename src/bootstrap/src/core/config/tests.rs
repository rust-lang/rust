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
use crate::core::build_steps::llvm::LLVM_INVALIDATION_PATHS;
use crate::core::build_steps::{llvm, test};
use crate::core::config::toml::TomlConfig;
use crate::core::config::{
    BootstrapOverrideLld, CompilerBuiltins, StringOrBool, Target, TargetSelection,
};
use crate::utils::tests::TestCtx;
use crate::utils::tests::git::git_test;

pub(crate) fn parse(config: &str) -> Config {
    TestCtx::new().config("check").with_default_toml_config(config).create_config()
}

fn get_toml(file: &Path) -> Result<TomlConfig, toml::de::Error> {
    let contents = std::fs::read_to_string(file).unwrap();
    toml::from_str(&contents).and_then(|table: toml::Value| TomlConfig::deserialize(table))
}

#[test]
fn download_ci_llvm() {
    let config = TestCtx::new().config("check").create_config();
    assert!(!config.llvm_from_ci);

    // this doesn't make sense, as we are overriding it later.
    let if_unchanged_config = TestCtx::new()
        .config("check")
        .with_default_toml_config("llvm.download-ci-llvm = \"if-unchanged\"")
        .create_config();
    if if_unchanged_config.llvm_from_ci && if_unchanged_config.is_running_on_ci {
        let has_changes = if_unchanged_config.has_changes_from_upstream(LLVM_INVALIDATION_PATHS);

        assert!(
            !has_changes,
            "CI LLVM can't be enabled with 'if-unchanged' while there are changes in LLVM submodule."
        );
    }
}

#[test]
fn clap_verify() {
    Flags::command().debug_assert();
}

#[test]
fn override_toml() {
    let config_toml: &str = r#"
    change-id = 0

    [rust]
    lto = "off"
    deny-warnings = true
    download-rustc = false

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
    "#;

    let args = [
        "--set=change-id=1",
        "--set=rust.lto=fat",
        "--set=rust.deny-warnings=false",
        "--set=build.optimized-compiler-builtins=true",
        "--set=build.gdb=\"bar\"",
        "--set=build.tools=[\"cargo\"]",
        "--set=llvm.build-config={\"foo\" = \"bar\"}",
        "--set=target.x86_64-unknown-linux-gnu.runner=bar",
        "--set=target.x86_64-unknown-linux-gnu.rpath=false",
        "--set=target.aarch64-unknown-linux-gnu.sanitizers=false",
        "--set=target.aarch64-apple-darwin.runner=apple",
        "--set=target.aarch64-apple-darwin.optimized-compiler-builtins=false",
    ];

    let config = TestCtx::new()
        .config("check")
        .with_default_toml_config(config_toml)
        .args(&args)
        .create_config();

    assert_eq!(config.change_id, Some(ChangeId::Id(1)), "setting top-level value");
    assert_eq!(
        config.rust_lto,
        crate::core::config::RustcLto::Fat,
        "setting string value without quotes"
    );
    assert_eq!(config.gdb, Some("bar".into()), "setting string value with quotes");
    assert!(!config.deny_warnings, "setting boolean value");
    assert_eq!(
        config.optimized_compiler_builtins,
        CompilerBuiltins::BuildLLVMFuncs,
        "setting boolean value"
    );
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
        optimized_compiler_builtins: Some(CompilerBuiltins::BuildRustOnly),
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
    TestCtx::new()
        .config("check")
        .with_default_toml_config("change-id = 0")
        .arg("--set")
        .arg("change-id=1")
        .arg("--set")
        .arg("change-id=2")
        .create_config();
}

#[test]
fn profile_user_dist() {
    TestCtx::new()
        .config("check")
        .with_default_toml_config(
            r#"
        profile = "user"
    "#,
        )
        .create_config();
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
    TestCtx::new()
        .config("check")
        .with_default_toml_config("rust.optimize = \"a\"")
        .create_config();
}

#[test]
fn verify_file_integrity() {
    let config = TestCtx::new().config("check").no_dry_run().create_config();

    let tempfile = config.tempdir().join(".tmp-test-file");
    File::create(&tempfile).unwrap().write_all(b"dummy value").unwrap();
    assert!(tempfile.exists());

    assert!(
        config
            .verify(&tempfile, "7e255dd9542648a8779268a0f268b891a198e9828e860ed23f826440e786eae5")
    );
}

#[test]
fn rust_lld() {
    assert!(matches!(parse("").bootstrap_override_lld, BootstrapOverrideLld::None));
    assert!(matches!(
        parse("rust.bootstrap-override-lld = \"self-contained\"").bootstrap_override_lld,
        BootstrapOverrideLld::SelfContained
    ));
    assert!(matches!(
        parse("rust.bootstrap-override-lld = \"external\"").bootstrap_override_lld,
        BootstrapOverrideLld::External
    ));
    assert!(matches!(
        parse("rust.bootstrap-override-lld = true").bootstrap_override_lld,
        BootstrapOverrideLld::External
    ));
    assert!(matches!(
        parse("rust.bootstrap-override-lld = false").bootstrap_override_lld,
        BootstrapOverrideLld::None
    ));

    // Also check the legacy options
    assert!(matches!(
        parse("rust.use-lld = true").bootstrap_override_lld,
        BootstrapOverrideLld::External
    ));
    assert!(matches!(
        parse("rust.use-lld = false").bootstrap_override_lld,
        BootstrapOverrideLld::None
    ));
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
    let args = [
        "clippy",
        "--fix",
        "--allow-dirty",
        "--allow-staged",
        "-Aclippy:all",
        "-Wclippy::style",
        "-Aclippy::foo1",
        "-Aclippy::foo2",
    ];
    let config = TestCtx::new().config(&args[0]).args(&args[1..]).create_config();

    let actual = match config.cmd.clone() {
        crate::Subcommand::Clippy { allow, deny, warn, forbid, .. } => {
            let cfg = LintConfig { allow, deny, warn, forbid };
            let args_vec: Vec<String> = args.iter().map(|s| s.to_string()).collect();
            get_clippy_rules_in_order(&args_vec, &cfg)
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
    let args = ["clippy", "-A clippy:all", "-W clippy::style"];
    let config = TestCtx::new().config(&args[0]).args(&args[1..]).create_config();

    let actual = match config.cmd.clone() {
        crate::Subcommand::Clippy { allow, deny, warn, forbid, .. } => {
            let cfg = LintConfig { allow, deny, warn, forbid };
            let args_vec: Vec<String> = args.iter().map(|s| s.to_string()).collect();
            get_clippy_rules_in_order(&args_vec, &cfg)
        }
        _ => panic!("invalid subcommand"),
    };

    let expected = vec!["-A clippy:all".to_string(), "-W clippy::style".to_string()];
    assert_eq!(expected, actual);
}

#[test]
fn verbose_tests_default_value() {
    let config = TestCtx::new().config("build").args(&["compiler".into()]).create_config();
    assert_eq!(config.verbose_tests, false);

    let config =
        TestCtx::new().config("build").args(&["compiler".into(), "-v".into()]).create_config();
    assert_eq!(config.verbose_tests, true);
}

#[test]
fn parse_rust_std_features() {
    let config = TestCtx::new()
        .config("check")
        .with_default_toml_config("rust.std-features = [\"panic-unwind\", \"backtrace\"]")
        .create_config();
    let expected_features: BTreeSet<String> =
        ["panic-unwind", "backtrace"].into_iter().map(|s| s.to_string()).collect();
    assert_eq!(config.rust_std_features, expected_features);
}

#[test]
fn parse_rust_std_features_empty() {
    let config = TestCtx::new()
        .config("check")
        .with_default_toml_config("rust.std-features = []")
        .create_config();
    let expected_features: BTreeSet<String> = BTreeSet::new();
    assert_eq!(config.rust_std_features, expected_features);
}

#[test]
#[should_panic]
fn parse_rust_std_features_invalid() {
    TestCtx::new()
        .config("check")
        .with_default_toml_config("rust.std-features = \"backtrace\"")
        .create_config();
}

#[test]
fn parse_jobs() {
    assert_eq!(
        TestCtx::new()
            .config("check")
            .with_default_toml_config("build.jobs = 1")
            .create_config()
            .jobs,
        Some(1)
    );
}

#[test]
fn jobs_precedence() {
    // `--jobs` should take precedence over using `--set build.jobs`.

    let config = TestCtx::new()
        .config("check")
        .args(&["--jobs=67890", "--set=build.jobs=12345"])
        .create_config();
    assert_eq!(config.jobs, Some(67890));

    // `--set build.jobs` should take precedence over `bootstrap.toml`.
    let config = TestCtx::new()
        .config("check")
        .args(&["--set=build.jobs=12345"])
        .with_default_toml_config(
            r#"
        [build]
        jobs = 67890
    "#,
        )
        .create_config();

    assert_eq!(config.jobs, Some(12345));

    // `--jobs` > `--set build.jobs` > `bootstrap.toml`
    let config = TestCtx::new()
        .config("check")
        .args(&["--jobs=123", "--set=build.jobs=456"])
        .with_default_toml_config(
            r#"
        [build]
        jobs = 789
    "#,
        )
        .create_config();
    assert_eq!(config.jobs, Some(123));
}

#[test]
fn check_rustc_if_unchanged_paths() {
    let config = TestCtx::new().config("check").create_config();
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
    let config = TestCtx::new()
        .config("check")
        .with_default_toml_config(
            r#"
            [build]
            test-stage = 1
        "#,
        )
        .create_config();

    assert!(!config.explicit_stage_from_cli);
    assert!(config.explicit_stage_from_config);
    assert!(config.is_explicit_stage());

    let config = TestCtx::new().config("check").stage(2).create_config();

    assert!(config.explicit_stage_from_cli);
    assert!(!config.explicit_stage_from_config);
    assert!(config.is_explicit_stage());

    let config = TestCtx::new()
        .config("check")
        .stage(2)
        .with_default_toml_config(
            r#"
            [build]
            test-stage = 1
        "#,
        )
        .create_config();

    assert!(config.explicit_stage_from_cli);
    assert!(config.explicit_stage_from_config);
    assert!(config.is_explicit_stage());

    let config = TestCtx::new().config("check").create_config();

    assert!(!config.explicit_stage_from_cli);
    assert!(!config.explicit_stage_from_config);
    assert!(!config.is_explicit_stage());
}

#[test]
fn test_exclude() {
    let exclude_path = "compiler";
    let config = TestCtx::new()
        .config("check")
        .with_default_toml_config(&format!("build.exclude=[\"{}\"]", exclude_path))
        .create_config();

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
    let config = TestCtx::new().config("check").arg("--ci").arg("false").create_config();
    assert!(!config.is_running_on_ci);

    let config = TestCtx::new().config("check").arg("--ci").arg("true").create_config();
    assert!(config.is_running_on_ci);

    let config = TestCtx::new().config("check").create_config();
    assert_eq!(config.is_running_on_ci, CiEnv::is_ci());
}

#[test]
fn test_precedence_of_includes() {
    let test_ctx = TestCtx::new();
    let testdir = test_ctx.dir();

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

    let config = test_ctx
        .config("check")
        .with_default_toml_config(
            r#"
        include = ["./extension.toml"]

        [llvm]
        link-jobs = 2
    "#,
        )
        .create_config();

    assert_eq!(config.change_id.unwrap(), ChangeId::Id(543));
    assert_eq!(config.llvm_link_jobs.unwrap(), 2);
    assert_eq!(config.description.unwrap(), "Some creative description");
}

#[test]
#[should_panic(expected = "Cyclic inclusion detected")]
fn test_cyclic_include_direct() {
    let test_ctx = TestCtx::new();
    let testdir = test_ctx.dir();
    let extension = testdir.join("extension.toml");
    let extension_content = br#"
        include = ["./bootstrap.toml"]
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    test_ctx
        .config("check")
        .with_default_toml_config(
            r#"
        include = ["./extension.toml"]
    "#,
        )
        .create_config();
}

#[test]
#[should_panic(expected = "Cyclic inclusion detected")]
fn test_cyclic_include_indirect() {
    let test_ctx = TestCtx::new();
    let testdir = test_ctx.dir();

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

    test_ctx
        .config("check")
        .with_default_toml_config(
            r#"
        include = ["./extension.toml"]
    "#,
        )
        .create_config();
}

#[test]
fn test_include_absolute_paths() {
    let test_ctx = TestCtx::new();
    let testdir = test_ctx.dir();

    let extension = testdir.join("extension.toml");
    File::create(&extension).unwrap().write_all(&[]).unwrap();

    let extension_absolute_path =
        extension.canonicalize().unwrap().to_str().unwrap().replace('\\', r"\\");
    let root_config_content = format!(r#"include = ["{}"]"#, extension_absolute_path);
    test_ctx.config("check").with_default_toml_config(&root_config_content).create_config();
}

#[test]
fn test_include_relative_paths() {
    let test_ctx = TestCtx::new();
    let testdir = test_ctx.dir();

    let _ = fs::create_dir_all(&testdir.join("subdir/another_subdir"));

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

    test_ctx
        .config("check")
        .with_default_toml_config(
            r#"
        include = ["./subdir/extension.toml"]
    "#,
        )
        .create_config();
}

#[test]
fn test_include_precedence_over_profile() {
    let test_ctx = TestCtx::new();
    let testdir = test_ctx.dir();

    let extension = testdir.join("extension.toml");
    let extension_content = br#"
        [rust]
        channel = "dev"
    "#;
    File::create(extension).unwrap().write_all(extension_content).unwrap();

    let config = test_ctx
        .config("check")
        .with_default_toml_config(
            r#"
        profile = "dist"
        include = ["./extension.toml"]
    "#,
        )
        .create_config();

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
