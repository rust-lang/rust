use std::env::VarError;
use std::{panic, thread};

use build_helper::stage0_parser::parse_stage0_file;
use llvm::prebuilt_llvm_config;

use super::*;
use crate::Flags;
use crate::core::build_steps::doc::DocumentationFormat;
use crate::core::config::Config;
use crate::utils::cache::ExecutedStep;
use crate::utils::helpers::get_host_target;
use crate::utils::tests::git::{GitCtx, git_test};
use crate::utils::tests::{ConfigBuilder, TestCtx};

static TEST_TRIPLE_1: &str = "i686-unknown-haiku";
static TEST_TRIPLE_2: &str = "i686-unknown-hurd-gnu";
static TEST_TRIPLE_3: &str = "i686-unknown-netbsd";

fn configure(cmd: &str, host: &[&str], target: &[&str]) -> Config {
    configure_with_args(&[cmd], host, target)
}

fn configure_with_args(cmd: &[&str], host: &[&str], target: &[&str]) -> Config {
    TestCtx::new()
        .config(cmd[0])
        .args(&cmd[1..])
        .hosts(host)
        .targets(target)
        .args(&["--build", TEST_TRIPLE_1])
        .create_config()
}

fn first<A, B>(v: Vec<(A, B)>) -> Vec<A> {
    v.into_iter().map(|(a, _)| a).collect::<Vec<_>>()
}

fn run_build(paths: &[PathBuf], config: Config) -> Cache {
    let kind = config.cmd.kind();
    let build = Build::new(config);
    let builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(kind), paths);
    builder.cache
}

fn check_cli<const N: usize>(paths: [&str; N]) {
    run_build(
        &paths.map(PathBuf::from),
        configure_with_args(&paths, &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]),
    );
}

macro_rules! std {
    ($host:ident => $target:ident, stage = $stage:literal) => {
        compile::Std::new(
            Compiler::new($stage, TargetSelection::from_user($host)),
            TargetSelection::from_user($target),
        )
    };
}

macro_rules! doc_std {
    ($host:ident => $target:ident, stage = $stage:literal) => {{ doc::Std::new($stage, TargetSelection::from_user($target), DocumentationFormat::Html) }};
}

macro_rules! rustc {
    ($host:ident => $target:ident, stage = $stage:literal) => {
        compile::Rustc::new(
            Compiler::new($stage, TargetSelection::from_user($host)),
            TargetSelection::from_user($target),
        )
    };
}

#[test]
fn test_valid() {
    // make sure multi suite paths are accepted
    check_cli(["test", "tests/ui/bootstrap/self-test/a.rs", "tests/ui/bootstrap/self-test/b.rs"]);
}

#[test]
#[should_panic]
fn test_invalid() {
    // make sure that invalid paths are caught, even when combined with valid paths
    check_cli(["test", "library/std", "x"]);
}

#[test]
fn test_intersection() {
    let set = |paths: &[&str]| {
        PathSet::Set(paths.into_iter().map(|p| TaskPath { path: p.into(), kind: None }).collect())
    };
    let library_set = set(&["library/core", "library/alloc", "library/std"]);
    let mut command_paths = vec![
        CLIStepPath::from(PathBuf::from("library/core")),
        CLIStepPath::from(PathBuf::from("library/alloc")),
        CLIStepPath::from(PathBuf::from("library/stdarch")),
    ];
    let subset = library_set.intersection_removing_matches(&mut command_paths, Kind::Build);
    assert_eq!(subset, set(&["library/core", "library/alloc"]),);
    assert_eq!(
        command_paths,
        vec![
            CLIStepPath::from(PathBuf::from("library/core")).will_be_executed(true),
            CLIStepPath::from(PathBuf::from("library/alloc")).will_be_executed(true),
            CLIStepPath::from(PathBuf::from("library/stdarch")).will_be_executed(false),
        ]
    );
}

#[test]
fn test_resolve_parent_and_subpaths() {
    let set = |paths: &[&str]| {
        PathSet::Set(paths.into_iter().map(|p| TaskPath { path: p.into(), kind: None }).collect())
    };

    let mut command_paths = vec![
        CLIStepPath::from(PathBuf::from("src/tools/miri")),
        CLIStepPath::from(PathBuf::from("src/tools/miri/cargo-miri")),
    ];

    let library_set = set(&["src/tools/miri", "src/tools/miri/cargo-miri"]);
    library_set.intersection_removing_matches(&mut command_paths, Kind::Build);

    assert_eq!(
        command_paths,
        vec![
            CLIStepPath::from(PathBuf::from("src/tools/miri")).will_be_executed(true),
            CLIStepPath::from(PathBuf::from("src/tools/miri/cargo-miri")).will_be_executed(true),
        ]
    );
}

#[test]
fn validate_path_remap() {
    let build = Build::new(configure("test", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]));

    PATH_REMAP
        .iter()
        .flat_map(|(_, paths)| paths.iter())
        .map(|path| build.src.join(path))
        .for_each(|path| {
            assert!(path.exists(), "{} should exist.", path.display());
        });
}

#[test]
fn check_missing_paths_for_x_test_tests() {
    let build = Build::new(configure("test", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]));

    let (_, tests_remap_paths) =
        PATH_REMAP.iter().find(|(target_path, _)| *target_path == "tests").unwrap();

    let tests_dir = fs::read_dir(build.src.join("tests")).unwrap();
    for dir in tests_dir {
        let path = dir.unwrap().path();

        // Skip if not a test directory.
        if path.ends_with("tests/auxiliary") || !path.is_dir() {
            continue;
        }

        assert!(
            tests_remap_paths.iter().any(|item| path.ends_with(*item)),
            "{} is missing in PATH_REMAP tests list.",
            path.display()
        );
    }
}

#[test]
fn ci_rustc_if_unchanged_invalidate_on_compiler_changes() {
    git_test(|ctx| {
        prepare_rustc_checkout(ctx);
        ctx.create_upstream_merge(&["compiler/bar"]);
        // This change should invalidate download-ci-rustc
        ctx.create_nonupstream_merge(&["compiler/foo"]);

        let config = parse_config_download_rustc_at(ctx.get_path(), "if-unchanged", true);
        assert_eq!(config.download_rustc_commit, None);
    });
}

#[test]
fn ci_rustc_if_unchanged_do_not_invalidate_on_library_changes_outside_ci() {
    git_test(|ctx| {
        prepare_rustc_checkout(ctx);
        let sha = ctx.create_upstream_merge(&["compiler/bar"]);
        // This change should not invalidate download-ci-rustc
        ctx.create_nonupstream_merge(&["library/foo"]);

        let config = parse_config_download_rustc_at(ctx.get_path(), "if-unchanged", false);
        assert_eq!(config.download_rustc_commit, Some(sha));
    });
}

#[test]
fn ci_rustc_if_unchanged_do_not_invalidate_on_tool_changes() {
    git_test(|ctx| {
        prepare_rustc_checkout(ctx);
        let sha = ctx.create_upstream_merge(&["compiler/bar"]);
        // This change should not invalidate download-ci-rustc
        ctx.create_nonupstream_merge(&["src/tools/foo"]);

        let config = parse_config_download_rustc_at(ctx.get_path(), "if-unchanged", true);
        assert_eq!(config.download_rustc_commit, Some(sha));
    });
}

/// Prepares the given directory so that it looks like a rustc checkout.
/// Also configures `GitCtx` to use the correct merge bot e-mail for upstream merge commits.
fn prepare_rustc_checkout(ctx: &mut GitCtx) {
    ctx.merge_bot_email =
        format!("Merge bot <{}>", parse_stage0_file().config.git_merge_commit_email);
    ctx.write("src/ci/channel", "nightly");
    ctx.commit();
}

/// Parses a Config directory from `path`, with the given value of `download_rustc`.
fn parse_config_download_rustc_at(path: &Path, download_rustc: &str, ci: bool) -> Config {
    Config::parse_inner(
        Flags::parse(&[
            "build".to_owned(),
            "--dry-run".to_owned(),
            "--ci".to_owned(),
            if ci { "true" } else { "false" }.to_owned(),
            format!("--set=rust.download-rustc='{download_rustc}'"),
            "--src".to_owned(),
            path.to_str().unwrap().to_owned(),
        ]),
        |&_| Ok(Default::default()),
    )
}

mod dist {
    use pretty_assertions::assert_eq;

    use super::{Config, TEST_TRIPLE_1, TEST_TRIPLE_2, TEST_TRIPLE_3, first, run_build};
    use crate::Flags;
    use crate::core::builder::*;

    fn configure(host: &[&str], target: &[&str]) -> Config {
        Config { stage: 2, ..super::configure("dist", host, target) }
    }

    #[test]
    fn llvm_out_behaviour() {
        let mut config = configure(&[TEST_TRIPLE_1], &[TEST_TRIPLE_2]);
        config.llvm_from_ci = true;
        let build = Build::new(config.clone());

        let target = TargetSelection::from_user(TEST_TRIPLE_1);
        assert!(build.llvm_out(target).ends_with("ci-llvm"));
        let target = TargetSelection::from_user(TEST_TRIPLE_2);
        assert!(build.llvm_out(target).ends_with("llvm"));

        config.llvm_from_ci = false;
        let build = Build::new(config.clone());
        let target = TargetSelection::from_user(TEST_TRIPLE_1);
        assert!(build.llvm_out(target).ends_with("llvm"));
    }
}

mod sysroot_target_dirs {
    use super::{
        Build, Builder, Compiler, TEST_TRIPLE_1, TEST_TRIPLE_2, TargetSelection, configure,
    };

    #[test]
    fn test_sysroot_target_libdir() {
        let build = Build::new(configure("build", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]));
        let builder = Builder::new(&build);
        let target_triple_1 = TargetSelection::from_user(TEST_TRIPLE_1);
        let compiler = Compiler::new(1, target_triple_1);
        let target_triple_2 = TargetSelection::from_user(TEST_TRIPLE_2);
        let actual = builder.sysroot_target_libdir(compiler, target_triple_2);

        assert_eq!(
            builder
                .sysroot(compiler)
                .join(builder.sysroot_libdir_relative(compiler))
                .join("rustlib")
                .join(TEST_TRIPLE_2)
                .join("lib"),
            actual
        );
    }

    #[test]
    fn test_sysroot_target_bindir() {
        let build = Build::new(configure("build", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]));
        let builder = Builder::new(&build);
        let target_triple_1 = TargetSelection::from_user(TEST_TRIPLE_1);
        let compiler = Compiler::new(1, target_triple_1);
        let target_triple_2 = TargetSelection::from_user(TEST_TRIPLE_2);
        let actual = builder.sysroot_target_bindir(compiler, target_triple_2);

        assert_eq!(
            builder
                .sysroot(compiler)
                .join(builder.sysroot_libdir_relative(compiler))
                .join("rustlib")
                .join(TEST_TRIPLE_2)
                .join("bin"),
            actual
        );
    }
}

/// Regression test for <https://github.com/rust-lang/rust/issues/134916>.
///
/// The command `./x test compiler` should invoke the step that runs unit tests
/// for (most) compiler crates; it should not be hijacked by the cg_clif or
/// cg_gcc tests instead.
#[test]
fn test_test_compiler() {
    let config = configure_with_args(&["test", "compiler"], &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);
    let cache = run_build(&config.paths.clone(), config);

    let compiler = cache.contains::<test::CrateLibrustc>();
    let cranelift = cache.contains::<test::CodegenCranelift>();
    let gcc = cache.contains::<test::CodegenGCC>();

    assert_eq!((compiler, cranelift, gcc), (true, false, false));
}

#[test]
fn test_test_coverage() {
    struct Case {
        cmd: &'static [&'static str],
        expected: &'static [&'static str],
    }
    let cases = &[
        Case { cmd: &["test"], expected: &["coverage-map", "coverage-run"] },
        Case { cmd: &["test", "coverage"], expected: &["coverage-map", "coverage-run"] },
        Case { cmd: &["test", "coverage-map"], expected: &["coverage-map"] },
        Case { cmd: &["test", "coverage-run"], expected: &["coverage-run"] },
        Case { cmd: &["test", "coverage", "--skip=coverage"], expected: &[] },
        Case { cmd: &["test", "coverage", "--skip=tests/coverage"], expected: &[] },
        Case { cmd: &["test", "coverage", "--skip=coverage-map"], expected: &["coverage-run"] },
        Case { cmd: &["test", "coverage", "--skip=coverage-run"], expected: &["coverage-map"] },
        Case { cmd: &["test", "--skip=coverage-map", "--skip=coverage-run"], expected: &[] },
        Case { cmd: &["test", "coverage", "--skip=tests"], expected: &[] },
    ];

    for &Case { cmd, expected } in cases {
        // Print each test case so that if one fails, the most recently printed
        // case is the one that failed.
        println!("Testing case: {cmd:?}");
        let config = configure_with_args(cmd, &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);
        let mut cache = run_build(&config.paths.clone(), config);

        let modes =
            cache.all::<test::Coverage>().iter().map(|(step, ())| step.mode).collect::<Vec<_>>();
        assert_eq!(modes, expected);
    }
}

#[test]
fn test_prebuilt_llvm_config_path_resolution() {
    fn configure(config: &str) -> Config {
        Config::parse_inner(
            Flags::parse(&[
                "build".to_string(),
                "--dry-run".to_string(),
                "--config=/does/not/exist".to_string(),
            ]),
            |&_| toml::from_str(&config),
        )
    }

    // Removes Windows disk prefix if present
    fn drop_win_disk_prefix_if_present(path: PathBuf) -> PathBuf {
        let path_str = path.to_str().unwrap();
        if let Some((_, without_prefix)) = path_str.split_once(":/") {
            return PathBuf::from(format!("/{}", without_prefix));
        }

        path
    }

    let config = configure(
        r#"
            [llvm]
            download-ci-llvm = false

            [build]
            build = "x86_64-unknown-linux-gnu"
            host = ["arm-unknown-linux-gnueabihf"]
            target = ["arm-unknown-linux-gnueabihf"]

            [target.x86_64-unknown-linux-gnu]
            llvm-config = "/some/path/to/llvm-config"

            [target.arm-unknown-linux-gnueabihf]
            cc = "arm-linux-gnueabihf-gcc"
            cxx = "arm-linux-gnueabihf-g++"
        "#,
    );

    let build = Build::new(config);
    let builder = Builder::new(&build);

    let expected = PathBuf::from("/some/path/to/llvm-config");

    let actual = prebuilt_llvm_config(
        &builder,
        TargetSelection::from_user("arm-unknown-linux-gnueabihf"),
        false,
    )
    .llvm_result()
    .host_llvm_config
    .clone();
    let actual = drop_win_disk_prefix_if_present(actual);
    assert_eq!(expected, actual);

    let actual = prebuilt_llvm_config(&builder, builder.config.host_target, false)
        .llvm_result()
        .host_llvm_config
        .clone();
    let actual = drop_win_disk_prefix_if_present(actual);
    assert_eq!(expected, actual);
    assert_eq!(expected, actual);

    let config = configure(
        r#"
            [llvm]
            download-ci-llvm = false
        "#,
    );

    let build = Build::new(config.clone());
    let builder = Builder::new(&build);

    let actual = prebuilt_llvm_config(&builder, builder.config.host_target, false)
        .llvm_result()
        .host_llvm_config
        .clone();
    let expected = builder
        .out
        .join(builder.config.host_target)
        .join("llvm/bin")
        .join(exe("llvm-config", builder.config.host_target));
    assert_eq!(expected, actual);

    let config = configure(
        r#"
            [llvm]
            download-ci-llvm = "if-unchanged"
        "#,
    );

    // CI-LLVM isn't always available; check if it's enabled before testing.
    if config.llvm_from_ci {
        let build = Build::new(config.clone());
        let builder = Builder::new(&build);

        let actual = prebuilt_llvm_config(&builder, builder.config.host_target, false)
            .llvm_result()
            .host_llvm_config
            .clone();
        let expected = builder
            .out
            .join(builder.config.host_target)
            .join("ci-llvm/bin")
            .join(exe("llvm-config", builder.config.host_target));
        assert_eq!(expected, actual);
    }
}

#[test]
fn test_is_builder_target() {
    let target1 = TargetSelection::from_user(TEST_TRIPLE_1);
    let target2 = TargetSelection::from_user(TEST_TRIPLE_2);

    for (target1, target2) in [(target1, target2), (target2, target1)] {
        let mut config = configure("build", &[], &[]);
        config.host_target = target1;
        let build = Build::new(config);
        let builder = Builder::new(&build);

        assert!(builder.config.is_host_target(target1));
        assert!(!builder.config.is_host_target(target2));
    }
}

/// When bootstrap detects a step dependency cycle (which is a bug), its panic
/// message should show the actual steps on the stack, not just several copies
/// of `Any { .. }`.
#[test]
fn step_cycle_debug() {
    let config = configure_with_args(&["run", "cyclic-step"], &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);

    let err = panic::catch_unwind(|| run_build(&config.paths.clone(), config)).unwrap_err();
    let err = err.downcast_ref::<String>().unwrap().as_str();

    assert!(!err.contains("Any"));
    assert!(err.contains("CyclicStep { n: 1 }"));
}

/// The `AnyDebug` trait should delegate to the underlying type's `Debug`, and
/// should also allow downcasting as expected.
#[test]
fn any_debug() {
    #[derive(Debug, PartialEq, Eq)]
    struct MyStruct {
        x: u32,
    }

    let x: &dyn AnyDebug = &MyStruct { x: 7 };

    // Debug-formatting should delegate to the underlying type.
    assert_eq!(format!("{x:?}"), format!("{:?}", MyStruct { x: 7 }));
    // Downcasting to the underlying type should succeed.
    assert_eq!(x.downcast_ref::<MyStruct>(), Some(&MyStruct { x: 7 }));
}

/// These tests use insta for snapshot testing.
/// See bootstrap's README on how to bless the snapshots.
mod snapshot {
    use std::path::PathBuf;

    use crate::core::build_steps::{compile, dist, doc, test, tool};
    use crate::core::builder::tests::{
        RenderConfig, TEST_TRIPLE_1, TEST_TRIPLE_2, TEST_TRIPLE_3, configure, first, host_target,
        render_steps, run_build,
    };
    use crate::core::builder::{Builder, Kind, StepDescription, StepMetadata};
    use crate::core::config::TargetSelection;
    use crate::core::config::toml::rust::with_lld_opt_in_targets;
    use crate::utils::cache::Cache;
    use crate::utils::helpers::get_host_target;
    use crate::utils::tests::{ConfigBuilder, TestCtx};
    use crate::{Build, Compiler, Config, Flags, Subcommand};

    #[test]
    fn build_default() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustdoc 1 <host>
        ");
    }

    #[test]
    fn build_cross_compile() {
        let ctx = TestCtx::new();

        insta::assert_snapshot!(
            ctx.config("build")
                // Cross-compilation fails on stage 1, as we don't have a stage0 std available
                // for non-host targets.
                .stage(2)
                .hosts(&[&host_target(), TEST_TRIPLE_1])
                .targets(&[&host_target(), TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 2 <host> -> std 2 <target1>
        [build] rustdoc 2 <host>
        [build] llvm <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustdoc 2 <target1>
        ");
    }

    #[test]
    fn build_with_empty_host() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("build")
                .hosts(&[])
                .targets(&[TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        "
        );
    }

    #[test]
    fn build_compiler_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        ");
    }

    #[test]
    fn build_rustc_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("rustc")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        ");
    }

    #[test]
    #[should_panic]
    fn build_compiler_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("build").path("compiler").stage(0).run();
    }

    #[test]
    fn build_compiler_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .stage(1)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        ");
    }

    #[test]
    fn build_compiler_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        ");
    }

    #[test]
    fn build_compiler_stage_3() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .stage(3)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 2 <host> -> rustc 3 <host>
        ");
    }

    #[test]
    fn build_compiler_stage_3_cross() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .hosts(&[TEST_TRIPLE_1])
                .stage(3)
                .render_steps(), @r"
        [build] llvm <host>
        [build] llvm <target1>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 2 <host> -> std 2 <target1>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 2 <host> -> rustc 3 <target1>
        ");
    }

    #[test]
    fn build_compiler_stage_3_full_bootstrap() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .stage(3)
                .args(&["--set", "build.full-bootstrap=true"])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 2 <host> -> rustc 3 <host>
        ");
    }

    #[test]
    fn build_compiler_stage_3_cross_full_bootstrap() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .stage(3)
                .hosts(&[TEST_TRIPLE_1])
                .args(&["--set", "build.full-bootstrap=true"])
                .render_steps(), @r"
        [build] llvm <host>
        [build] llvm <target1>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <target1>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 2 <host> -> rustc 3 <target1>
        ");
    }

    #[test]
    fn build_compiler_codegen_backend() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("build")
                .args(&["--set", "rust.codegen-backends=['llvm', 'cranelift']"])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> rustc_codegen_cranelift 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustdoc 1 <host>
        "
        );
    }

    #[test]
    fn build_compiler_tools() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("build")
                .stage(2)
                .args(&["--set", "rust.lld=true", "--set", "rust.llvm-bitcode-linker=true"])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> LldWrapper 1 <host>
        [build] rustc 0 <host> -> LlvmBitcodeLinker 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> LldWrapper 2 <host>
        [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustdoc 2 <host>
        "
        );
    }

    #[test]
    fn build_compiler_tools_cross() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("build")
                .stage(2)
                .args(&["--set", "rust.lld=true", "--set", "rust.llvm-bitcode-linker=true"])
                .hosts(&[TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> LldWrapper 1 <host>
        [build] rustc 0 <host> -> LlvmBitcodeLinker 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> LldWrapper 2 <host>
        [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 2 <host> -> std 2 <target1>
        [build] llvm <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustc 1 <host> -> LldWrapper 2 <target1>
        [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <target1>
        [build] rustdoc 2 <target1>
        "
        );
    }

    #[test]
    fn build_compiler_lld_opt_in() {
        with_lld_opt_in_targets(vec![host_target()], || {
            let ctx = TestCtx::new();
            insta::assert_snapshot!(
                ctx.config("build")
                    .path("compiler")
                    .render_steps(), @r"
            [build] llvm <host>
            [build] rustc 0 <host> -> rustc 1 <host>
            [build] rustc 0 <host> -> LldWrapper 1 <host>
            ");
        });
    }

    #[test]
    fn build_library_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
            .path("library")
            .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        ");
    }

    #[test]
    #[should_panic]
    fn build_library_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("build").path("library").stage(0).run();
    }

    #[test]
    fn build_library_stage_0_local_rebuild() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("library")
                .stage(0)
                .targets(&[TEST_TRIPLE_1])
                .args(&["--set", "build.local-rebuild=true"])
                .render_steps(), @"[build] rustc 0 <host> -> std 0 <target1>");
    }

    #[test]
    fn build_library_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("library")
                .stage(1)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        ");
    }

    #[test]
    fn build_library_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("library")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        ");
    }

    #[test]
    fn build_miri_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("miri")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> miri 1 <host>
        ");
    }

    #[test]
    #[should_panic]
    fn build_miri_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("build").path("miri").stage(0).run();
    }

    #[test]
    fn build_miri_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("miri")
                .stage(1)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> miri 1 <host>
        ");
    }

    #[test]
    fn build_miri_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("miri")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> miri 2 <host>
        ");
    }

    #[test]
    fn build_error_index() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("error_index_generator")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> error-index 1 <host>
        ");
    }

    #[test]
    fn build_bootstrap_tool_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("opt-dist")
                .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist 1 <host>");
    }

    #[test]
    #[should_panic]
    fn build_bootstrap_tool_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("build").path("opt-dist").stage(0).run();
    }

    #[test]
    fn build_bootstrap_tool_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("opt-dist")
                .stage(1)
                .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist 1 <host>");
    }

    #[test]
    fn build_bootstrap_tool_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("opt-dist")
                .stage(2)
                .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist 1 <host>");
    }

    #[test]
    fn build_default_stage() {
        let ctx = TestCtx::new();
        assert_eq!(ctx.config("build").path("compiler").create_config().stage, 1);
    }

    /// Ensure that if someone passes both a single crate and `library`, all
    /// library crates get built.
    #[test]
    fn alias_and_path_for_library() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(ctx.config("build")
            .paths(&["library", "core"])
            .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        ");

        insta::assert_snapshot!(ctx.config("build")
            .paths(&["std"])
            .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        ");

        insta::assert_snapshot!(ctx.config("build")
            .paths(&["core"])
            .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        ");

        insta::assert_snapshot!(ctx.config("build")
            .paths(&["alloc"])
            .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        ");

        insta::assert_snapshot!(ctx.config("doc")
            .paths(&["library", "core"])
            .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        ");
    }

    #[test]
    fn build_all() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .stage(2)
                .paths(&["compiler/rustc", "library"])
                .hosts(&[&host_target(), TEST_TRIPLE_1])
                .targets(&[&host_target(), TEST_TRIPLE_1, TEST_TRIPLE_2])
            .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] llvm <target1>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 2 <host> -> std 2 <target1>
        [build] rustc 1 <host> -> std 1 <target2>
        [build] rustc 2 <host> -> std 2 <target2>
        ");
    }

    #[test]
    fn build_cargo() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .paths(&["cargo"])
            .render_steps(), @"[build] rustc 0 <host> -> cargo 1 <host>");
    }

    #[test]
    fn build_cargo_cross() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .paths(&["cargo"])
                .hosts(&[TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 1 <host> -> cargo 2 <target1>
        ");
    }

    #[test]
    fn dist_default_stage() {
        let ctx = TestCtx::new();
        assert_eq!(ctx.config("dist").path("compiler").create_config().stage, 2);
    }

    #[test]
    fn dist_baseline() {
        let ctx = TestCtx::new();
        // Note that stdlib is uplifted, that is why `[dist] rustc 1 <host> -> std <host>` is in
        // the output.
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 1 <host> -> releases 2 <host>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <host>
        [dist] mingw <host>
        [build] rustdoc 2 <host>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [dist] rustc <host>
        [dist] rustc 1 <host> -> std 1 <host>
        [dist] rustc 1 <host> -> rustc-dev 2 <host>
        [dist] src <>
        [dist] reproducible-artifacts <host>
        "
        );
    }

    #[test]
    fn dist_compiler_docs() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("dist")
                .path("rustc-docs")
                .args(&["--set", "build.compiler-docs=true"])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [doc] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [doc] rustc 1 <host> -> Rustdoc 2 <host>
        [doc] rustc 1 <host> -> Rustfmt 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] rustc 1 <host> -> Cargo 2 <host>
        [doc] cargo (book) <host>
        [doc] rustc 1 <host> -> Clippy 2 <host>
        [doc] clippy (book) <host>
        [doc] rustc 1 <host> -> Miri 2 <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [build] rustdoc 0 <host>
        [doc] rustc 0 <host> -> Tidy 1 <host>
        [doc] rustc 0 <host> -> Bootstrap 1 <host>
        [doc] rustc 1 <host> -> releases 2 <host>
        [doc] rustc 0 <host> -> RunMakeSupport 1 <host>
        [doc] rustc 0 <host> -> BuildHelper 1 <host>
        [doc] rustc 0 <host> -> Compiletest 1 <host>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        "
        );
    }

    #[test]
    fn dist_extended() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("dist")
            .args(&[
                "--set",
                "build.extended=true",
                "--set",
                "rust.llvm-bitcode-linker=true",
                "--set",
                "rust.lld=true",
            ])
            .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> LldWrapper 1 <host>
        [build] rustc 0 <host> -> WasmComponentLd 1 <host>
        [build] rustc 0 <host> -> LlvmBitcodeLinker 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> LldWrapper 2 <host>
        [build] rustc 1 <host> -> WasmComponentLd 2 <host>
        [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 1 <host> -> releases 2 <host>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <host>
        [dist] mingw <host>
        [build] rustdoc 2 <host>
        [build] rustc 1 <host> -> rust-analyzer-proc-macro-srv 2 <host>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [dist] rustc <host>
        [dist] rustc 1 <host> -> std 1 <host>
        [dist] rustc 1 <host> -> rustc-dev 2 <host>
        [dist] rustc 1 <host> -> analysis 2 <host>
        [dist] src <>
        [build] rustc 1 <host> -> cargo 2 <host>
        [dist] rustc 1 <host> -> cargo 2 <host>
        [build] rustc 1 <host> -> rust-analyzer 2 <host>
        [dist] rustc 1 <host> -> rust-analyzer 2 <host>
        [build] rustc 1 <host> -> rustfmt 2 <host>
        [build] rustc 1 <host> -> cargo-fmt 2 <host>
        [dist] rustc 1 <host> -> rustfmt 2 <host>
        [build] rustc 1 <host> -> clippy-driver 2 <host>
        [build] rustc 1 <host> -> cargo-clippy 2 <host>
        [dist] rustc 1 <host> -> clippy 2 <host>
        [build] rustc 1 <host> -> miri 2 <host>
        [build] rustc 1 <host> -> cargo-miri 2 <host>
        [dist] rustc 1 <host> -> miri 2 <host>
        [doc] rustc 2 <host> -> std 2 <host> crates=[]
        [dist] rustc 2 <host> -> json-docs 3 <host>
        [dist] rustc 1 <host> -> extended 2 <host>
        [dist] reproducible-artifacts <host>
        ");
    }

    #[test]
    fn dist_with_targets() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .hosts(&[&host_target()])
                .targets(&[&host_target(), TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [doc] unstable-book (book) <target1>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [doc] book (book) <target1>
        [doc] book/first-edition (book) <target1>
        [doc] book/second-edition (book) <target1>
        [doc] book/2018-edition (book) <target1>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> standalone 2 <target1>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [doc] nomicon (book) <host>
        [doc] nomicon (book) <target1>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustc 1 <host> -> reference (book) 2 <target1>
        [doc] rustdoc (book) <host>
        [doc] rustdoc (book) <target1>
        [doc] rust-by-example (book) <host>
        [doc] rust-by-example (book) <target1>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] cargo (book) <target1>
        [doc] clippy (book) <host>
        [doc] clippy (book) <target1>
        [doc] embedded-book (book) <host>
        [doc] embedded-book (book) <target1>
        [doc] edition-guide (book) <host>
        [doc] edition-guide (book) <target1>
        [doc] style-guide (book) <host>
        [doc] style-guide (book) <target1>
        [doc] rustc 1 <host> -> releases 2 <host>
        [doc] rustc 1 <host> -> releases 2 <target1>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <host>
        [dist] docs <target1>
        [doc] rustc 1 <host> -> std 1 <host> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <host>
        [doc] rustc 1 <host> -> std 1 <target1> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <target1>
        [dist] mingw <host>
        [dist] mingw <target1>
        [build] rustdoc 2 <host>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [dist] rustc <host>
        [dist] rustc 1 <host> -> std 1 <host>
        [dist] rustc 1 <host> -> std 1 <target1>
        [dist] rustc 1 <host> -> rustc-dev 2 <host>
        [dist] src <>
        [dist] reproducible-artifacts <host>
        "
        );
    }

    #[test]
    fn dist_with_hosts() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .hosts(&[&host_target(), TEST_TRIPLE_1])
                .targets(&[&host_target()])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [build] llvm <target1>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustc 1 <host> -> error-index 2 <target1>
        [doc] rustc 1 <host> -> error-index 2 <target1>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] rustc (book) <target1>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 1 <host> -> releases 2 <host>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <host>
        [dist] mingw <host>
        [build] rustdoc 2 <host>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [dist] rustc <host>
        [build] rustdoc 2 <target1>
        [dist] rustc <target1>
        [dist] rustc 1 <host> -> std 1 <host>
        [dist] rustc 1 <host> -> rustc-dev 2 <host>
        [dist] rustc 1 <host> -> rustc-dev 2 <target1>
        [dist] src <>
        [dist] reproducible-artifacts <host>
        [dist] reproducible-artifacts <target1>
        "
        );
    }

    #[test]
    fn dist_with_targets_and_hosts() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .hosts(&[&host_target(), TEST_TRIPLE_1])
                .targets(&[&host_target(), TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [doc] unstable-book (book) <target1>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [doc] book (book) <target1>
        [doc] book/first-edition (book) <target1>
        [doc] book/second-edition (book) <target1>
        [doc] book/2018-edition (book) <target1>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> standalone 2 <target1>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [build] llvm <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustc 1 <host> -> error-index 2 <target1>
        [doc] rustc 1 <host> -> error-index 2 <target1>
        [doc] nomicon (book) <host>
        [doc] nomicon (book) <target1>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustc 1 <host> -> reference (book) 2 <target1>
        [doc] rustdoc (book) <host>
        [doc] rustdoc (book) <target1>
        [doc] rust-by-example (book) <host>
        [doc] rust-by-example (book) <target1>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] rustc (book) <target1>
        [doc] cargo (book) <host>
        [doc] cargo (book) <target1>
        [doc] clippy (book) <host>
        [doc] clippy (book) <target1>
        [doc] embedded-book (book) <host>
        [doc] embedded-book (book) <target1>
        [doc] edition-guide (book) <host>
        [doc] edition-guide (book) <target1>
        [doc] style-guide (book) <host>
        [doc] style-guide (book) <target1>
        [doc] rustc 1 <host> -> releases 2 <host>
        [doc] rustc 1 <host> -> releases 2 <target1>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <host>
        [dist] docs <target1>
        [doc] rustc 1 <host> -> std 1 <host> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <host>
        [doc] rustc 1 <host> -> std 1 <target1> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <target1>
        [dist] mingw <host>
        [dist] mingw <target1>
        [build] rustdoc 2 <host>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [dist] rustc <host>
        [build] rustdoc 2 <target1>
        [dist] rustc <target1>
        [dist] rustc 1 <host> -> std 1 <host>
        [dist] rustc 1 <host> -> std 1 <target1>
        [dist] rustc 1 <host> -> rustc-dev 2 <host>
        [dist] rustc 1 <host> -> rustc-dev 2 <target1>
        [dist] src <>
        [dist] reproducible-artifacts <host>
        [dist] reproducible-artifacts <target1>
        "
        );
    }

    #[test]
    fn dist_with_empty_host() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .hosts(&[])
                .targets(&[TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <target1>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [doc] book (book) <target1>
        [doc] book/first-edition (book) <target1>
        [doc] book/second-edition (book) <target1>
        [doc] book/2018-edition (book) <target1>
        [build] rustdoc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <target1>
        [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [doc] nomicon (book) <target1>
        [doc] rustc 1 <host> -> reference (book) 2 <target1>
        [doc] rustdoc (book) <target1>
        [doc] rust-by-example (book) <target1>
        [doc] cargo (book) <target1>
        [doc] clippy (book) <target1>
        [doc] embedded-book (book) <target1>
        [doc] edition-guide (book) <target1>
        [doc] style-guide (book) <target1>
        [doc] rustc 1 <host> -> releases 2 <target1>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <target1>
        [doc] rustc 1 <host> -> std 1 <target1> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <target1>
        [dist] mingw <target1>
        [dist] rustc 1 <host> -> std 1 <target1>
        ");
    }

    #[test]
    fn dist_all_cross_extended() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .hosts(&[TEST_TRIPLE_1])
                .targets(&[TEST_TRIPLE_1])
                .args(&["--set", "rust.channel=nightly", "--set", "build.extended=true"])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <target1>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> WasmComponentLd 1 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [doc] book (book) <target1>
        [doc] book/first-edition (book) <target1>
        [doc] book/second-edition (book) <target1>
        [doc] book/2018-edition (book) <target1>
        [build] rustdoc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <target1>
        [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] llvm <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustc 1 <host> -> WasmComponentLd 2 <target1>
        [build] rustc 1 <host> -> error-index 2 <target1>
        [doc] rustc 1 <host> -> error-index 2 <target1>
        [doc] nomicon (book) <target1>
        [doc] rustc 1 <host> -> reference (book) 2 <target1>
        [doc] rustdoc (book) <target1>
        [doc] rust-by-example (book) <target1>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <target1>
        [doc] cargo (book) <target1>
        [doc] clippy (book) <target1>
        [doc] embedded-book (book) <target1>
        [doc] edition-guide (book) <target1>
        [doc] style-guide (book) <target1>
        [doc] rustc 1 <host> -> releases 2 <target1>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <target1>
        [doc] rustc 1 <host> -> std 1 <target1> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <target1>
        [dist] mingw <target1>
        [build] rustdoc 2 <target1>
        [build] rustc 1 <host> -> rust-analyzer-proc-macro-srv 2 <target1>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [dist] rustc <target1>
        [dist] rustc 1 <host> -> std 1 <target1>
        [dist] rustc 1 <host> -> rustc-dev 2 <target1>
        [dist] rustc 1 <host> -> analysis 2 <target1>
        [dist] src <>
        [build] rustc 1 <host> -> cargo 2 <target1>
        [dist] rustc 1 <host> -> cargo 2 <target1>
        [build] rustc 1 <host> -> rust-analyzer 2 <target1>
        [dist] rustc 1 <host> -> rust-analyzer 2 <target1>
        [build] rustc 1 <host> -> rustfmt 2 <target1>
        [build] rustc 1 <host> -> cargo-fmt 2 <target1>
        [dist] rustc 1 <host> -> rustfmt 2 <target1>
        [build] rustc 1 <host> -> clippy-driver 2 <target1>
        [build] rustc 1 <host> -> cargo-clippy 2 <target1>
        [dist] rustc 1 <host> -> clippy 2 <target1>
        [build] rustc 1 <host> -> miri 2 <target1>
        [build] rustc 1 <host> -> cargo-miri 2 <target1>
        [dist] rustc 1 <host> -> miri 2 <target1>
        [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <target1>
        [doc] rustc 2 <target1> -> std 2 <target1> crates=[]
        [dist] rustc 2 <target1> -> json-docs 3 <target1>
        [dist] rustc 1 <host> -> extended 2 <target1>
        [dist] reproducible-artifacts <target1>
        ");
    }

    /// Simulates e.g. the powerpc64 builder, which is fully cross-compiled from x64, but it does
    /// not build docs. Crucially, it shouldn't build host stage 2 rustc.
    ///
    /// This is a regression test for <https://github.com/rust-lang/rust/issues/138123>
    /// and <https://github.com/rust-lang/rust/issues/138004>.
    #[test]
    fn dist_all_cross_extended_no_docs() {
        let ctx = TestCtx::new();
        let steps = ctx
            .config("dist")
            .hosts(&[TEST_TRIPLE_1])
            .targets(&[TEST_TRIPLE_1])
            .args(&[
                "--set",
                "rust.channel=nightly",
                "--set",
                "build.extended=true",
                "--set",
                "build.docs=false",
            ])
            .get_steps();

        // Make sure that we don't build stage2 host rustc
        steps.assert_no_match(|m| {
            m.name == "rustc"
                && m.built_by.map(|b| b.stage) == Some(1)
                && *m.target.triple == host_target()
        });

        insta::assert_snapshot!(
                steps.render(), @r"
        [dist] mingw <target1>
        [build] llvm <host>
        [build] llvm <target1>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> WasmComponentLd 1 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustc 1 <host> -> WasmComponentLd 2 <target1>
        [build] rustdoc 2 <target1>
        [build] rustc 1 <host> -> rust-analyzer-proc-macro-srv 2 <target1>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] rustc <target1>
        [dist] rustc 1 <host> -> std 1 <target1>
        [dist] rustc 1 <host> -> rustc-dev 2 <target1>
        [dist] rustc 1 <host> -> analysis 2 <target1>
        [dist] src <>
        [build] rustc 1 <host> -> cargo 2 <target1>
        [dist] rustc 1 <host> -> cargo 2 <target1>
        [build] rustc 1 <host> -> rust-analyzer 2 <target1>
        [dist] rustc 1 <host> -> rust-analyzer 2 <target1>
        [build] rustc 1 <host> -> rustfmt 2 <target1>
        [build] rustc 1 <host> -> cargo-fmt 2 <target1>
        [dist] rustc 1 <host> -> rustfmt 2 <target1>
        [build] rustc 1 <host> -> clippy-driver 2 <target1>
        [build] rustc 1 <host> -> cargo-clippy 2 <target1>
        [dist] rustc 1 <host> -> clippy 2 <target1>
        [build] rustc 1 <host> -> miri 2 <target1>
        [build] rustc 1 <host> -> cargo-miri 2 <target1>
        [dist] rustc 1 <host> -> miri 2 <target1>
        [build] rustc 1 <host> -> LlvmBitcodeLinker 2 <target1>
        [dist] rustc 1 <host> -> extended 2 <target1>
        [dist] reproducible-artifacts <target1>
        ");
    }

    /// Enable dist cranelift tarball by default with `x dist` if cranelift is enabled in
    /// `rust.codegen-backends`.
    #[test]
    fn dist_cranelift_by_default() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .args(&["--set", "rust.codegen-backends=['llvm', 'cranelift']"])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> rustc_codegen_cranelift 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> rustc_codegen_cranelift 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 1 <host> -> releases 2 <host>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[]
        [dist] rustc 1 <host> -> json-docs 2 <host>
        [dist] mingw <host>
        [build] rustdoc 2 <host>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [dist] rustc <host>
        [dist] rustc 1 <host> -> rustc_codegen_cranelift 2 <host>
        [dist] rustc 1 <host> -> std 1 <host>
        [dist] rustc 1 <host> -> rustc-dev 2 <host>
        [dist] src <>
        [dist] reproducible-artifacts <host>
        ");
    }

    #[test]
    fn dist_bootstrap() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .path("bootstrap")
                .render_steps(), @r"
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] bootstrap <host>
        ");
    }

    #[test]
    fn dist_library_stage_0_local_rebuild() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("dist")
                .path("rust-std")
                .stage(0)
                .targets(&[TEST_TRIPLE_1])
                .args(&["--set", "build.local-rebuild=true"])
                .render_steps(), @r"
        [build] rustc 0 <host> -> std 0 <target1>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] rustc 0 <host> -> std 0 <target1>
        ");
    }

    #[test]
    fn dist_rustc_docs() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .path("rustc-docs")
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 1 <host> -> releases 2 <host>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        ");
    }

    #[test]
    fn check_compiler_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("compiler")
                .render_steps(), @"[check] rustc 0 <host> -> rustc 1 <host> (75 crates)");
    }

    #[test]
    fn check_rustc_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("rustc")
                .render_steps(), @"[check] rustc 0 <host> -> rustc 1 <host> (1 crates)");
    }

    #[test]
    #[should_panic]
    fn check_compiler_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("check").path("compiler").stage(0).run();
    }

    #[test]
    fn check_compiler_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("compiler")
                .stage(1)
                .render_steps(), @"[check] rustc 0 <host> -> rustc 1 <host> (75 crates)");
    }

    #[test]
    fn check_compiler_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("compiler")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [check] rustc 1 <host> -> rustc 2 <host> (75 crates)
        ");
    }

    #[test]
    fn check_cross_compile() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .targets(&[TEST_TRIPLE_1])
                .hosts(&[TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [check] rustc 1 <host> -> std 1 <target1>
        [check] rustc 1 <host> -> rustc 2 <target1> (75 crates)
        [check] rustc 1 <host> -> rustc 2 <target1>
        [check] rustc 1 <host> -> Rustdoc 2 <target1>
        [check] rustc 1 <host> -> rustc_codegen_cranelift 2 <target1>
        [check] rustc 1 <host> -> rustc_codegen_gcc 2 <target1>
        [check] rustc 1 <host> -> Clippy 2 <target1>
        [check] rustc 1 <host> -> Miri 2 <target1>
        [check] rustc 1 <host> -> CargoMiri 2 <target1>
        [check] rustc 1 <host> -> Rustfmt 2 <target1>
        [check] rustc 1 <host> -> RustAnalyzer 2 <target1>
        [check] rustc 1 <host> -> TestFloatParse 2 <target1>
        [check] rustc 1 <host> -> std 1 <target1>
        ");
    }

    #[test]
    fn check_library_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("library")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [check] rustc 1 <host> -> std 1 <host>
        ");
    }

    #[test]
    #[should_panic]
    fn check_library_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("check").path("library").stage(0).run();
    }

    #[test]
    fn check_library_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("library")
                .stage(1)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [check] rustc 1 <host> -> std 1 <host>
        ");
    }

    #[test]
    fn check_library_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("library")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [check] rustc 2 <host> -> std 2 <host>
        ");
    }

    #[test]
    fn check_library_cross_compile() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .paths(&["core", "alloc", "std"])
                .targets(&[TEST_TRIPLE_1, TEST_TRIPLE_2])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [check] rustc 1 <host> -> std 1 <target1>
        [check] rustc 1 <host> -> std 1 <target2>
        ");
    }

    /// Make sure that we don't check library when download-rustc is disabled
    /// when `--skip-std-check-if-no-download-rustc` was passed.
    #[test]
    fn check_library_skip_without_download_rustc() {
        let ctx = TestCtx::new();
        let args = ["--set", "rust.download-rustc=false", "--skip-std-check-if-no-download-rustc"];
        insta::assert_snapshot!(
            ctx.config("check")
                .paths(&["library"])
                .args(&args)
                .render_steps(), @"");

        insta::assert_snapshot!(
            ctx.config("check")
                .paths(&["library", "compiler"])
                .args(&args)
                .render_steps(), @"[check] rustc 0 <host> -> rustc 1 <host> (75 crates)");
    }

    #[test]
    fn check_miri_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("miri")
                .render_steps(), @r"
        [check] rustc 0 <host> -> rustc 1 <host>
        [check] rustc 0 <host> -> Miri 1 <host>
        ");
    }

    #[test]
    #[should_panic]
    fn check_miri_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("check").path("miri").stage(0).run();
    }

    #[test]
    fn check_miri_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("miri")
                .stage(1)
                .render_steps(), @r"
        [check] rustc 0 <host> -> rustc 1 <host>
        [check] rustc 0 <host> -> Miri 1 <host>
        ");
    }

    #[test]
    fn check_miri_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("miri")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [check] rustc 1 <host> -> rustc 2 <host>
        [check] rustc 1 <host> -> Miri 2 <host>
        ");
    }

    #[test]
    fn check_compiletest() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("compiletest")
                .render_steps(), @"[check] rustc 0 <host> -> Compiletest 1 <host>");
    }

    #[test]
    fn check_compiletest_stage1_libtest() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("compiletest")
                .args(&["--set", "build.compiletest-use-stage0-libtest=false"])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [check] rustc 1 <host> -> Compiletest 2 <host>
        ");
    }

    #[test]
    fn check_codegen() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("rustc_codegen_cranelift")
                .render_steps(), @r"
        [check] rustc 0 <host> -> rustc 1 <host>
        [check] rustc 0 <host> -> rustc_codegen_cranelift 1 <host>
        ");
    }

    #[test]
    fn check_rust_analyzer() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("rust-analyzer")
                .render_steps(), @r"
        [check] rustc 0 <host> -> rustc 1 <host>
        [check] rustc 0 <host> -> RustAnalyzer 1 <host>
        ");
    }

    #[test]
    fn check_bootstrap_tool() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("check")
                .path("run-make-support")
                .render_steps(), @"[check] rustc 0 <host> -> RunMakeSupport 1 <host>");
    }

    fn prepare_test_config(ctx: &TestCtx) -> ConfigBuilder {
        ctx.config("test")
            // Bootstrap only runs by default on CI, so we have to emulate that also locally.
            .args(&["--ci", "true"])
            // These rustdoc tests requires nodejs to be present.
            // We can't easily opt out of it, so if it is present on the local PC, the test
            // would have different result on CI, where nodejs might be missing.
            .args(&["--skip", "rustdoc-js-std"])
            .args(&["--skip", "rustdoc-js"])
            .args(&["--skip", "rustdoc-gui"])
    }

    #[test]
    fn test_all_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            prepare_test_config(&ctx)
                .render_steps(), @r"
        [build] rustc 0 <host> -> Tidy 1 <host>
        [test] tidy <>
        [build] rustdoc 0 <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> Compiletest 1 <host>
        [test] compiletest-ui 1 <host>
        [test] compiletest-crashes 1 <host>
        [build] rustc 0 <host> -> CoverageDump 1 <host>
        [test] compiletest-coverage 1 <host>
        [test] compiletest-coverage 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [test] compiletest-mir-opt 1 <host>
        [test] compiletest-codegen-llvm 1 <host>
        [test] compiletest-codegen-units 1 <host>
        [test] compiletest-assembly-llvm 1 <host>
        [test] compiletest-incremental 1 <host>
        [test] compiletest-debuginfo 1 <host>
        [test] compiletest-ui-fulldeps 1 <host>
        [build] rustdoc 1 <host>
        [test] compiletest-rustdoc 1 <host>
        [test] compiletest-coverage-run-rustdoc 1 <host>
        [test] compiletest-pretty 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> std 0 <host>
        [test] rustc 0 <host> -> CrateLibrustc 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [test] crate-bootstrap <host> src/tools/coverage-dump
        [test] crate-bootstrap <host> src/tools/jsondoclint
        [test] crate-bootstrap <host> src/tools/replace-version-placeholder
        [test] crate-bootstrap <host> tidyselftest
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [doc] rustc 0 <host> -> standalone 1 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 0 <host> -> error-index 1 <host>
        [doc] rustc 0 <host> -> error-index 1 <host>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 0 <host> -> releases 1 <host>
        [build] rustc 0 <host> -> Linkchecker 1 <host>
        [test] link-check <host>
        [test] tier-check <host>
        [test] rustc 0 <host> -> rust-analyzer 1 <host>
        [build] rustc 0 <host> -> RustdocTheme 1 <host>
        [test] rustdoc-theme 1 <host>
        [test] compiletest-rustdoc-ui 1 <host>
        [build] rustc 0 <host> -> JsonDocCk 1 <host>
        [build] rustc 0 <host> -> JsonDocLint 1 <host>
        [test] compiletest-rustdoc-json 1 <host>
        [doc] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> HtmlChecker 1 <host>
        [test] html-check <host>
        [build] rustc 0 <host> -> RunMakeSupport 1 <host>
        [test] compiletest-run-make 1 <host>
        [build] rustc 0 <host> -> cargo 1 <host>
        [test] compiletest-run-make-cargo 1 <host>
        ");
    }

    #[test]
    fn test_compiletest_suites_stage1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .args(&["ui", "ui-fulldeps", "run-make", "rustdoc", "rustdoc-gui", "incremental"])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> Compiletest 1 <host>
        [test] compiletest-ui 1 <host>
        [test] compiletest-ui-fulldeps 1 <host>
        [build] rustc 0 <host> -> RunMakeSupport 1 <host>
        [build] rustdoc 1 <host>
        [test] compiletest-run-make 1 <host>
        [test] compiletest-rustdoc 1 <host>
        [build] rustc 0 <host> -> RustdocGUITest 1 <host>
        [test] rustdoc-gui 1 <host>
        [test] compiletest-incremental 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        ");
    }

    #[test]
    fn test_compiletest_suites_stage2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .args(&["ui", "ui-fulldeps", "run-make", "rustdoc", "rustdoc-gui", "incremental"])
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> Compiletest 1 <host>
        [test] compiletest-ui 2 <host>
        [build] rustc 2 <host> -> rustc 3 <host>
        [test] compiletest-ui-fulldeps 2 <host>
        [build] rustc 0 <host> -> RunMakeSupport 1 <host>
        [build] rustdoc 2 <host>
        [test] compiletest-run-make 2 <host>
        [test] compiletest-rustdoc 2 <host>
        [build] rustc 0 <host> -> RustdocGUITest 1 <host>
        [test] rustdoc-gui 2 <host>
        [test] compiletest-incremental 2 <host>
        [build] rustdoc 1 <host>
        ");
    }

    #[test]
    fn test_compiletest_suites_stage2_cross() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .hosts(&[TEST_TRIPLE_1])
                .targets(&[TEST_TRIPLE_1])
                .args(&["ui", "ui-fulldeps", "run-make", "rustdoc", "rustdoc-gui", "incremental"])
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> Compiletest 1 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 2 <host> -> std 2 <target1>
        [test] compiletest-ui 2 <target1>
        [build] llvm <target1>
        [build] rustc 2 <host> -> rustc 3 <target1>
        [test] compiletest-ui-fulldeps 2 <target1>
        [build] rustc 0 <host> -> RunMakeSupport 1 <host>
        [build] rustdoc 2 <host>
        [test] compiletest-run-make 2 <target1>
        [test] compiletest-rustdoc 2 <target1>
        [build] rustc 0 <host> -> RustdocGUITest 1 <host>
        [test] rustdoc-gui 2 <target1>
        [test] compiletest-incremental 2 <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustdoc 1 <host>
        [build] rustc 2 <target1> -> std 2 <target1>
        [build] rustdoc 2 <target1>
        ");
    }

    #[test]
    fn test_all_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            prepare_test_config(&ctx)
                .stage(2)
                .render_steps(), @r"
        [build] rustc 0 <host> -> Tidy 1 <host>
        [test] tidy <>
        [build] rustdoc 0 <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> Compiletest 1 <host>
        [test] compiletest-ui 2 <host>
        [test] compiletest-crashes 2 <host>
        [build] rustc 0 <host> -> CoverageDump 1 <host>
        [test] compiletest-coverage 2 <host>
        [test] compiletest-coverage 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [test] compiletest-mir-opt 2 <host>
        [test] compiletest-codegen-llvm 2 <host>
        [test] compiletest-codegen-units 2 <host>
        [test] compiletest-assembly-llvm 2 <host>
        [test] compiletest-incremental 2 <host>
        [test] compiletest-debuginfo 2 <host>
        [build] rustc 2 <host> -> rustc 3 <host>
        [test] compiletest-ui-fulldeps 2 <host>
        [build] rustdoc 2 <host>
        [test] compiletest-rustdoc 2 <host>
        [test] compiletest-coverage-run-rustdoc 2 <host>
        [test] compiletest-pretty 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustdoc 1 <host>
        [test] rustc 1 <host> -> CrateLibrustc 2 <host>
        [test] crate-bootstrap <host> src/tools/coverage-dump
        [test] crate-bootstrap <host> src/tools/jsondoclint
        [test] crate-bootstrap <host> src/tools/replace-version-placeholder
        [test] crate-bootstrap <host> tidyselftest
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 1 <host> -> releases 2 <host>
        [build] rustc 0 <host> -> Linkchecker 1 <host>
        [test] link-check <host>
        [test] tier-check <host>
        [test] rustc 1 <host> -> rust-analyzer 2 <host>
        [doc] rustc (book) <host>
        [test] rustc 1 <host> -> lint-docs 2 <host>
        [build] rustc 0 <host> -> RustdocTheme 1 <host>
        [test] rustdoc-theme 2 <host>
        [test] compiletest-rustdoc-ui 2 <host>
        [build] rustc 0 <host> -> JsonDocCk 1 <host>
        [build] rustc 0 <host> -> JsonDocLint 1 <host>
        [test] compiletest-rustdoc-json 2 <host>
        [doc] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 0 <host> -> HtmlChecker 1 <host>
        [test] html-check <host>
        [build] rustc 0 <host> -> RunMakeSupport 1 <host>
        [test] compiletest-run-make 2 <host>
        [build] rustc 1 <host> -> cargo 2 <host>
        [test] compiletest-run-make-cargo 2 <host>
        ");
    }

    #[test]
    fn test_compiler_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .path("compiler")
                .stage(1)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> std 0 <host>
        [build] rustdoc 0 <host>
        [test] rustc 0 <host> -> CrateLibrustc 1 <host>
        ");
    }

    #[test]
    fn test_compiler_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .path("compiler")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustdoc 1 <host>
        [test] rustc 1 <host> -> CrateLibrustc 2 <host>
        ");
    }

    #[test]
    fn test_exclude() {
        let ctx = TestCtx::new();
        let steps = ctx.config("test").args(&["--skip", "src/tools/tidy"]).get_steps();

        let host = TargetSelection::from_user(&host_target());
        steps.assert_contains(StepMetadata::test("compiletest-rustdoc-ui", host).stage(1));
        steps.assert_not_contains(test::Tidy);
    }

    #[test]
    fn test_exclude_kind() {
        let ctx = TestCtx::new();
        let host = TargetSelection::from_user(&host_target());

        let get_steps = |args: &[&str]| ctx.config("test").args(args).get_steps();

        let rustc_metadata =
            || StepMetadata::test("CrateLibrustc", host).built_by(Compiler::new(0, host));
        // Ensure our test is valid, and `test::Rustc` would be run without the exclude.
        get_steps(&[]).assert_contains(rustc_metadata());

        let steps = get_steps(&["--skip", "compiler/rustc_data_structures"]);

        // Ensure tests for rustc are not skipped.
        steps.assert_contains(rustc_metadata());
        steps.assert_contains_fuzzy(StepMetadata::build("rustc", host));
    }

    #[test]
    fn test_cargo_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .path("cargo")
                .render_steps(), @r"
        [build] rustc 0 <host> -> cargo 1 <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustdoc 1 <host>
        [build] rustdoc 0 <host>
        [test] rustc 0 <host> -> cargo 1 <host>
        ");
    }

    #[test]
    fn test_cargo_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .path("cargo")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> cargo 2 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustdoc 2 <host>
        [build] rustdoc 1 <host>
        [test] rustc 1 <host> -> cargo 2 <host>
        ");
    }

    #[test]
    fn test_cargotest() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .path("cargotest")
                .render_steps(), @r"
        [build] rustc 0 <host> -> cargo 1 <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> CargoTest 1 <host>
        [build] rustdoc 1 <host>
        [test] cargotest 1 <host>
        ");
    }

    #[test]
    fn test_tier_check() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .path("tier-check")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [test] tier-check <host>
        ");
    }

    // Differential snapshots for `./x test run-make` run `./x test run-make-cargo`: only
    // `run-make-cargo` should build an in-tree cargo, running `./x test run-make` should not.
    #[test]
    fn test_run_make_no_cargo() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .path("run-make")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> RunMakeSupport 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> Compiletest 1 <host>
        [build] rustdoc 1 <host>
        [test] compiletest-run-make 1 <host>
        ");
    }

    #[test]
    fn test_run_make_cargo_builds_cargo() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("test")
                .path("run-make-cargo")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> RunMakeSupport 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> Compiletest 1 <host>
        [build] rustc 0 <host> -> cargo 1 <host>
        [build] rustdoc 1 <host>
        [test] compiletest-run-make-cargo 1 <host>
        ");
    }

    #[test]
    fn doc_all() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 0 <host>
        [doc] rustc 0 <host> -> standalone 1 <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 0 <host> -> error-index 1 <host>
        [doc] rustc 0 <host> -> error-index 1 <host>
        [doc] nomicon (book) <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 0 <host> -> releases 1 <host>
        ");
    }

    #[test]
    fn doc_library() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("library")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        ");
    }

    #[test]
    fn doc_cargo_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("cargo")
                .render_steps(), @r"
        [build] rustdoc 0 <host>
        [doc] rustc 0 <host> -> Cargo 1 <host>
        ");
    }
    #[test]
    fn doc_cargo_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("cargo")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> Cargo 2 <host>
        ");
    }

    #[test]
    fn doc_core() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("core")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[core]
        ");
    }

    #[test]
    fn doc_core_no_std_target() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("core")
                .override_target_no_std(&host_target())
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[core]
        ");
    }

    #[test]
    fn doc_library_no_std_target() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("library")
                .override_target_no_std(&host_target())
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,core]
        ");
    }

    #[test]
    fn doc_library_no_std_target_cross_compile() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("library")
                .targets(&[TEST_TRIPLE_1])
                .override_target_no_std(TEST_TRIPLE_1)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> std 1 <target1> crates=[alloc,core]
        ");
    }

    #[test]
    #[should_panic]
    fn doc_compiler_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("doc").path("compiler").stage(0).run();
    }

    #[test]
    fn doc_compiler_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("compiler")
                .stage(1)
                .render_steps(), @r"
        [build] rustdoc 0 <host>
        [build] llvm <host>
        [doc] rustc 0 <host> -> rustc 1 <host>
        ");
    }

    #[test]
    fn doc_compiler_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("compiler")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> rustc 2 <host>
        ");
    }

    #[test]
    #[should_panic]
    fn doc_compiletest_stage_0() {
        let ctx = TestCtx::new();
        ctx.config("doc").path("src/tools/compiletest").stage(0).run();
    }

    #[test]
    fn doc_compiletest_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("src/tools/compiletest")
                .stage(1)
                .render_steps(), @r"
        [build] rustdoc 0 <host>
        [doc] rustc 0 <host> -> Compiletest 1 <host>
        ");
    }

    #[test]
    fn doc_compiletest_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("src/tools/compiletest")
                .stage(2)
                .render_steps(), @r"
        [build] rustdoc 0 <host>
        [doc] rustc 0 <host> -> Compiletest 1 <host>
        ");
    }

    // Reference should be auto-bumped to stage 2.
    #[test]
    fn doc_reference() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("doc")
                .path("reference")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        ");
    }

    #[test]
    fn clippy_ci() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("clippy")
                .path("ci")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> clippy-driver 1 <host>
        [build] rustc 0 <host> -> cargo-clippy 1 <host>
        [clippy] rustc 1 <host> -> bootstrap 2 <host>
        [clippy] rustc 1 <host> -> std 1 <host>
        [clippy] rustc 1 <host> -> rustc 2 <host>
        [check] rustc 1 <host> -> rustc 2 <host>
        [clippy] rustc 1 <host> -> rustc_codegen_gcc 2 <host>
        ");
    }

    #[test]
    fn clippy_compiler_stage1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("clippy")
                .path("compiler")
                .render_steps(), @r"
        [build] llvm <host>
        [clippy] rustc 0 <host> -> rustc 1 <host>
        ");
    }

    #[test]
    fn clippy_compiler_stage2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("clippy")
                .path("compiler")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 0 <host> -> clippy-driver 1 <host>
        [build] rustc 0 <host> -> cargo-clippy 1 <host>
        [clippy] rustc 1 <host> -> rustc 2 <host>
        ");
    }

    #[test]
    fn clippy_std_stage1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("clippy")
                .path("std")
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> clippy-driver 1 <host>
        [build] rustc 0 <host> -> cargo-clippy 1 <host>
        [clippy] rustc 1 <host> -> std 1 <host>
        ");
    }

    #[test]
    fn clippy_std_stage2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("clippy")
                .path("std")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> clippy-driver 2 <host>
        [build] rustc 1 <host> -> cargo-clippy 2 <host>
        [clippy] rustc 2 <host> -> std 2 <host>
        ");
    }

    #[test]
    fn clippy_miri_stage1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("clippy")
                .path("miri")
                .stage(1)
                .render_steps(), @r"
        [build] llvm <host>
        [check] rustc 0 <host> -> rustc 1 <host>
        [clippy] rustc 0 <host> -> miri 1 <host>
        ");
    }

    #[test]
    fn clippy_miri_stage2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("clippy")
                .path("miri")
                .stage(2)
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [check] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 0 <host> -> clippy-driver 1 <host>
        [build] rustc 0 <host> -> cargo-clippy 1 <host>
        [clippy] rustc 1 <host> -> miri 2 <host>
        ");
    }

    #[test]
    fn clippy_bootstrap() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("clippy")
                .path("bootstrap")
                .render_steps(), @"[clippy] rustc 0 <host> -> bootstrap 1 <host>");
    }

    #[test]
    fn install_extended() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("install")
                .args(&[
                    // Using backslashes fails with `--set`
                    "--set", &format!("install.prefix={}", ctx.dir().display()).replace("\\", "/"),
                    "--set", &format!("install.sysconfdir={}", ctx.dir().display()).replace("\\", "/"),
                    "--set", "build.extended=true"
                ])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> WasmComponentLd 1 <host>
        [build] rustc 0 <host> -> UnstableBookGen 1 <host>
        [build] rustc 0 <host> -> Rustbook 1 <host>
        [doc] unstable-book (book) <host>
        [build] rustc 1 <host> -> std 1 <host>
        [doc] book (book) <host>
        [doc] book/first-edition (book) <host>
        [doc] book/second-edition (book) <host>
        [doc] book/2018-edition (book) <host>
        [build] rustdoc 1 <host>
        [doc] rustc 1 <host> -> standalone 2 <host>
        [doc] rustc 1 <host> -> std 1 <host> crates=[alloc,compiler_builtins,core,panic_abort,panic_unwind,proc_macro,rustc-std-workspace-core,std,std_detect,sysroot,test,unwind]
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> WasmComponentLd 2 <host>
        [build] rustc 1 <host> -> error-index 2 <host>
        [doc] rustc 1 <host> -> error-index 2 <host>
        [doc] nomicon (book) <host>
        [doc] rustc 1 <host> -> reference (book) 2 <host>
        [doc] rustdoc (book) <host>
        [doc] rust-by-example (book) <host>
        [build] rustc 0 <host> -> LintDocs 1 <host>
        [doc] rustc (book) <host>
        [doc] cargo (book) <host>
        [doc] clippy (book) <host>
        [doc] embedded-book (book) <host>
        [doc] edition-guide (book) <host>
        [doc] style-guide (book) <host>
        [doc] rustc 1 <host> -> releases 2 <host>
        [build] rustc 0 <host> -> RustInstaller 1 <host>
        [dist] docs <host>
        [dist] rustc 1 <host> -> std 1 <host>
        [build] rustdoc 2 <host>
        [build] rustc 1 <host> -> rust-analyzer-proc-macro-srv 2 <host>
        [build] rustc 0 <host> -> GenerateCopyright 1 <host>
        [dist] rustc <host>
        [build] rustc 1 <host> -> cargo 2 <host>
        [dist] rustc 1 <host> -> cargo 2 <host>
        [build] rustc 1 <host> -> rust-analyzer 2 <host>
        [dist] rustc 1 <host> -> rust-analyzer 2 <host>
        [build] rustc 1 <host> -> rustfmt 2 <host>
        [build] rustc 1 <host> -> cargo-fmt 2 <host>
        [dist] rustc 1 <host> -> rustfmt 2 <host>
        [build] rustc 1 <host> -> clippy-driver 2 <host>
        [build] rustc 1 <host> -> cargo-clippy 2 <host>
        [dist] rustc 1 <host> -> clippy 2 <host>
        [build] rustc 1 <host> -> miri 2 <host>
        [build] rustc 1 <host> -> cargo-miri 2 <host>
        [dist] rustc 1 <host> -> miri 2 <host>
        [dist] src <>
        ");
    }

    // Check that `x run miri --target FOO` actually builds miri for the host.
    #[test]
    fn run_miri() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("run")
                .path("miri")
                .stage(1)
                .targets(&[TEST_TRIPLE_1])
                .render_steps(), @r"
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> miri 1 <host>
        [build] rustc 0 <host> -> cargo-miri 1 <host>
        [run] rustc 0 <host> -> miri 1 <target1>
        ");
    }
}

struct ExecutedSteps {
    steps: Vec<ExecutedStep>,
}

impl ExecutedSteps {
    fn render(&self) -> String {
        self.render_with(RenderConfig::default())
    }

    fn render_with(&self, config: RenderConfig) -> String {
        render_steps(&self.steps, config)
    }

    #[track_caller]
    fn assert_contains<M: Into<StepMetadata>>(&self, metadata: M) {
        let metadata = metadata.into();
        if !self.contains(&metadata) {
            panic!(
                "Metadata `{}` ({metadata:?}) not found in executed steps:\n{}",
                render_metadata(&metadata, &RenderConfig::default()),
                self.render()
            );
        }
    }

    /// Try to match metadata by similarity, it does not need to match exactly.
    /// Stages (and built_by compiler) do not need to match, but name, target and
    /// kind has to match.
    #[track_caller]
    fn assert_contains_fuzzy<M: Into<StepMetadata>>(&self, metadata: M) {
        let metadata = metadata.into();
        if !self.contains_fuzzy(&metadata) {
            panic!(
                "Metadata `{}` ({metadata:?}) not found in executed steps:\n{}",
                render_metadata(&metadata, &RenderConfig::default()),
                self.render()
            );
        }
    }

    #[track_caller]
    fn assert_not_contains<M: Into<StepMetadata>>(&self, metadata: M) {
        let metadata = metadata.into();
        if self.contains(&metadata) {
            panic!(
                "Metadata `{}` ({metadata:?}) found in executed steps (it should not be there):\n{}",
                render_metadata(&metadata, &RenderConfig::default()),
                self.render()
            );
        }
    }

    /// Make sure that no metadata matches the given `func`.
    #[track_caller]
    fn assert_no_match<F>(&self, func: F)
    where
        F: Fn(StepMetadata) -> bool,
    {
        for metadata in self.steps.iter().filter_map(|s| s.metadata.clone()) {
            if func(metadata.clone()) {
                panic!(
                    "Metadata {metadata:?} was found, even though it should have not been present"
                );
            }
        }
    }

    fn contains(&self, metadata: &StepMetadata) -> bool {
        self.steps
            .iter()
            .filter_map(|s| s.metadata.as_ref())
            .any(|executed_metadata| executed_metadata == metadata)
    }

    fn contains_fuzzy(&self, metadata: &StepMetadata) -> bool {
        self.steps
            .iter()
            .filter_map(|s| s.metadata.as_ref())
            .any(|executed_metadata| fuzzy_metadata_eq(executed_metadata, metadata))
    }
}

fn fuzzy_metadata_eq(executed: &StepMetadata, to_match: &StepMetadata) -> bool {
    let StepMetadata { name, kind, target, built_by: _, stage: _, metadata } = executed;
    *name == to_match.name && *kind == to_match.kind && *target == to_match.target
}

impl<S: Step> From<S> for StepMetadata {
    fn from(step: S) -> Self {
        step.metadata().expect("step has no metadata")
    }
}

impl ConfigBuilder {
    fn run(self) -> Cache {
        let config = self.create_config();

        let kind = config.cmd.kind();
        let build = Build::new(config);
        let builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(kind), &builder.paths);
        builder.cache
    }

    fn get_steps(self) -> ExecutedSteps {
        let cache = self.run();
        ExecutedSteps { steps: cache.into_executed_steps() }
    }

    fn render_steps(self) -> String {
        self.get_steps().render()
    }
}

struct RenderConfig {
    normalize_host: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self { normalize_host: true }
    }
}

/// Renders the executed bootstrap steps for usage in snapshot tests with insta.
/// Only renders certain important steps.
/// Each value in `steps` should be a tuple of (Step, step output).
///
/// The arrow in the rendered output (`X -> Y`) means `X builds Y`.
/// This is similar to the output printed by bootstrap to stdout, but here it is
/// generated purely for the purpose of tests.
fn render_steps(steps: &[ExecutedStep], config: RenderConfig) -> String {
    steps
        .iter()
        .filter_map(|step| {
            use std::fmt::Write;

            let Some(metadata) = &step.metadata else {
                return None;
            };

            Some(render_metadata(&metadata, &config))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_metadata(metadata: &StepMetadata, config: &RenderConfig) -> String {
    let mut record = format!("[{}] ", metadata.kind.as_str());
    if let Some(compiler) = metadata.built_by {
        write!(record, "{} -> ", render_compiler(compiler, config));
    }
    let stage = metadata.get_stage().map(|stage| format!("{stage} ")).unwrap_or_default();
    write!(record, "{} {stage}<{}>", metadata.name, normalize_target(metadata.target, config));
    if let Some(metadata) = &metadata.metadata {
        write!(record, " {metadata}");
    }
    record
}

fn normalize_target(target: TargetSelection, config: &RenderConfig) -> String {
    let mut target = target.to_string();
    if config.normalize_host {
        target = target.replace(&host_target(), "host");
    }
    target.replace(TEST_TRIPLE_1, "target1").replace(TEST_TRIPLE_2, "target2")
}

fn render_compiler(compiler: Compiler, config: &RenderConfig) -> String {
    format!("rustc {} <{}>", compiler.stage, normalize_target(compiler.host, config))
}

fn host_target() -> String {
    get_host_target().to_string()
}
