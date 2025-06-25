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
use crate::utils::tests::ConfigBuilder;
use crate::utils::tests::git::{GitCtx, git_test};

static TEST_TRIPLE_1: &str = "i686-unknown-haiku";
static TEST_TRIPLE_2: &str = "i686-unknown-hurd-gnu";
static TEST_TRIPLE_3: &str = "i686-unknown-netbsd";

fn configure(cmd: &str, host: &[&str], target: &[&str]) -> Config {
    configure_with_args(&[cmd], host, target)
}

fn configure_with_args(cmd: &[&str], host: &[&str], target: &[&str]) -> Config {
    let cmd = cmd.iter().copied().map(String::from).collect::<Vec<_>>();
    let mut config = Config::parse(Flags::parse(&cmd));
    // don't save toolstates
    config.save_toolstates = None;
    config.set_dry_run(DryRun::SelfCheck);

    // Ignore most submodules, since we don't need them for a dry run, and the
    // tests run much faster without them.
    //
    // The src/doc/book submodule is needed because TheBook step tries to
    // access files even during a dry-run (may want to consider just skipping
    // that in a dry run).
    let submodule_build = Build::new(Config {
        // don't include LLVM, so CI doesn't require ninja/cmake to be installed
        rust_codegen_backends: vec![],
        ..Config::parse(Flags::parse(&["check".to_owned()]))
    });
    submodule_build.require_submodule("src/doc/book", None);
    config.submodules = Some(false);

    config.ninja_in_file = false;
    // try to avoid spurious failures in dist where we create/delete each others file
    // HACK: rather than pull in `tempdir`, use the one that cargo has conveniently created for us
    let dir = Path::new(env!("OUT_DIR"))
        .join("tmp-rustbuild-tests")
        .join(&thread::current().name().unwrap_or("unknown").replace(":", "-"));
    t!(fs::create_dir_all(&dir));
    config.out = dir;
    config.host_target = TargetSelection::from_user(TEST_TRIPLE_1);
    config.hosts = host.iter().map(|s| TargetSelection::from_user(s)).collect();
    config.targets = target.iter().map(|s| TargetSelection::from_user(s)).collect();
    config
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

mod defaults {
    use pretty_assertions::assert_eq;

    use super::{TEST_TRIPLE_1, TEST_TRIPLE_2, configure, first, run_build};
    use crate::Config;
    use crate::core::builder::*;

    #[test]
    fn doc_default() {
        let mut config = configure("doc", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);
        config.compiler_docs = true;
        config.cmd = Subcommand::Doc { open: false, json: false };
        let mut cache = run_build(&[], config);
        let a = TargetSelection::from_user(TEST_TRIPLE_1);

        // error_index_generator uses stage 0 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(first(cache.all::<doc::ErrorIndex>()), &[doc::ErrorIndex { target: a },]);
        assert_eq!(
            first(cache.all::<tool::ErrorIndex>()),
            &[tool::ErrorIndex { compiler: Compiler::new(1, a) }]
        );
        // docs should be built with the stage0 compiler, not with the stage0 artifacts.
        // recall that rustdoc is off-by-one: `stage` is the compiler rustdoc is _linked_ to,
        // not the one it was built by.
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            &[tool::Rustdoc { compiler: Compiler::new(1, a) },]
        );
    }
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

    #[test]
    fn doc_ci() {
        let mut config = configure(&[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);
        config.compiler_docs = true;
        config.cmd = Subcommand::Doc { open: false, json: false };
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Doc), &[]);
        let a = TargetSelection::from_user(TEST_TRIPLE_1);

        // error_index_generator uses stage 1 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(
            first(builder.cache.all::<doc::ErrorIndex>()),
            &[doc::ErrorIndex { target: a },]
        );
        assert_eq!(
            first(builder.cache.all::<tool::ErrorIndex>()),
            &[tool::ErrorIndex { compiler: Compiler::new(1, a) }]
        );
        // This is actually stage 1, but Rustdoc::run swaps out the compiler with
        // stage minus 1 if --stage is not 0. Very confusing!
        assert_eq!(
            first(builder.cache.all::<tool::Rustdoc>()),
            &[tool::Rustdoc { compiler: Compiler::new(2, a) },]
        );
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
    .llvm_config
    .clone();
    let actual = drop_win_disk_prefix_if_present(actual);
    assert_eq!(expected, actual);

    let actual = prebuilt_llvm_config(&builder, builder.config.host_target, false)
        .llvm_result()
        .llvm_config
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
        .llvm_config
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
            .llvm_config
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

#[test]
fn test_get_tool_rustc_compiler() {
    let mut config = configure("build", &[], &[]);
    config.download_rustc_commit = None;
    let build = Build::new(config);
    let builder = Builder::new(&build);

    let target_triple_1 = TargetSelection::from_user(TEST_TRIPLE_1);

    let compiler = Compiler::new(2, target_triple_1);
    let expected = Compiler::new(1, target_triple_1);
    let actual = tool::get_tool_rustc_compiler(&builder, compiler);
    assert_eq!(expected, actual);

    let compiler = Compiler::new(1, target_triple_1);
    let expected = Compiler::new(0, target_triple_1);
    let actual = tool::get_tool_rustc_compiler(&builder, compiler);
    assert_eq!(expected, actual);

    let mut config = configure("build", &[], &[]);
    config.download_rustc_commit = Some("".to_owned());
    let build = Build::new(config);
    let builder = Builder::new(&build);

    let compiler = Compiler::new(1, target_triple_1);
    let expected = Compiler::new(1, target_triple_1);
    let actual = tool::get_tool_rustc_compiler(&builder, compiler);
    assert_eq!(expected, actual);
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
        TEST_TRIPLE_1, TEST_TRIPLE_2, TEST_TRIPLE_3, configure, configure_with_args, first,
        host_target, render_steps, run_build,
    };
    use crate::core::builder::{Builder, Kind, StepDescription, StepMetadata};
    use crate::core::config::TargetSelection;
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
        [build] rustdoc 0 <host>
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
        [build] rustdoc 1 <host>
        [build] llvm <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustdoc 1 <target1>
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
    fn build_bootstrap_tool_no_explicit_stage() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("opt-dist")
                .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist <host>");
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
                .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist <host>");
    }

    #[test]
    fn build_bootstrap_tool_stage_2() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("opt-dist")
                .stage(2)
                .render_steps(), @"[build] rustc 0 <host> -> OptimizedDist <host>");
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
        [build] rustdoc 0 <host>
        [doc] std 1 <host>
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
        [build] rustc 2 <host> -> std 2 <target2>
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
        [build] rustc 0 <host> -> UnstableBookGen <host>
        [build] rustc 0 <host> -> Rustbook <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustdoc 1 <host>
        [doc] std 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> LintDocs <host>
        [build] rustc 0 <host> -> RustInstaller <host>
        [dist] docs <host>
        [doc] std 2 <host>
        [dist] mingw <host>
        [build] rustc 0 <host> -> GenerateCopyright <host>
        [dist] rustc <host>
        [dist] rustc 1 <host> -> std <host>
        [dist] src <>
        "
        );
    }

    #[test]
    fn dist_extended() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .args(&["--set", "build.extended=true"])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen <host>
        [build] rustc 0 <host> -> Rustbook <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> WasmComponentLd <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> WasmComponentLd <host>
        [build] rustdoc 1 <host>
        [doc] std 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> LintDocs <host>
        [build] rustc 0 <host> -> RustInstaller <host>
        [dist] docs <host>
        [doc] std 2 <host>
        [dist] mingw <host>
        [build] rustc 0 <host> -> GenerateCopyright <host>
        [dist] rustc <host>
        [dist] rustc 1 <host> -> std <host>
        [dist] src <>
        [build] rustc 0 <host> -> rustfmt 1 <host>
        [build] rustc 0 <host> -> cargo-fmt 1 <host>
        [build] rustc 0 <host> -> clippy-driver 1 <host>
        [build] rustc 0 <host> -> cargo-clippy 1 <host>
        [build] rustc 0 <host> -> miri 1 <host>
        [build] rustc 0 <host> -> cargo-miri 1 <host>
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
        [build] rustc 0 <host> -> UnstableBookGen <host>
        [build] rustc 0 <host> -> Rustbook <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustdoc 1 <host>
        [doc] std 2 <host>
        [doc] std 2 <target1>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> LintDocs <host>
        [build] rustc 0 <host> -> RustInstaller <host>
        [dist] docs <host>
        [dist] docs <target1>
        [doc] std 2 <host>
        [doc] std 2 <target1>
        [dist] mingw <host>
        [dist] mingw <target1>
        [build] rustc 0 <host> -> GenerateCopyright <host>
        [dist] rustc <host>
        [dist] rustc 1 <host> -> std <host>
        [build] rustc 2 <host> -> std 2 <target1>
        [dist] rustc 2 <host> -> std <target1>
        [dist] src <>
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
        [build] rustc 0 <host> -> UnstableBookGen <host>
        [build] rustc 0 <host> -> Rustbook <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustdoc 1 <host>
        [doc] std 2 <host>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> LintDocs <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 2 <host> -> std 2 <target1>
        [build] rustc 0 <host> -> RustInstaller <host>
        [dist] docs <host>
        [doc] std 2 <host>
        [dist] mingw <host>
        [build] rustc 0 <host> -> GenerateCopyright <host>
        [dist] rustc <host>
        [build] llvm <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustdoc 1 <target1>
        [dist] rustc <target1>
        [dist] rustc 1 <host> -> std <host>
        [dist] src <>
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
        [build] rustc 0 <host> -> UnstableBookGen <host>
        [build] rustc 0 <host> -> Rustbook <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustdoc 1 <host>
        [doc] std 2 <host>
        [doc] std 2 <target1>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> LintDocs <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 2 <host> -> std 2 <target1>
        [build] rustc 0 <host> -> RustInstaller <host>
        [dist] docs <host>
        [dist] docs <target1>
        [doc] std 2 <host>
        [doc] std 2 <target1>
        [dist] mingw <host>
        [dist] mingw <target1>
        [build] rustc 0 <host> -> GenerateCopyright <host>
        [dist] rustc <host>
        [build] llvm <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustdoc 1 <target1>
        [dist] rustc <target1>
        [dist] rustc 1 <host> -> std <host>
        [dist] rustc 1 <host> -> std <target1>
        [dist] src <>
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
        [build] rustc 0 <host> -> UnstableBookGen <host>
        [build] rustc 0 <host> -> Rustbook <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustdoc 1 <host>
        [doc] std 2 <target1>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 0 <host> -> RustInstaller <host>
        [dist] docs <target1>
        [doc] std 2 <target1>
        [dist] mingw <target1>
        [build] rustc 2 <host> -> std 2 <target1>
        [dist] rustc 2 <host> -> std <target1>
        ");
    }

    /// This also serves as an important regression test for <https://github.com/rust-lang/rust/issues/138123>
    /// and <https://github.com/rust-lang/rust/issues/138004>.
    #[test]
    fn dist_all_cross() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx
                .config("dist")
                .hosts(&[TEST_TRIPLE_1])
                .targets(&[TEST_TRIPLE_1])
                .args(&["--set", "rust.channel=nightly", "--set", "build.extended=true"])
                .render_steps(), @r"
        [build] rustc 0 <host> -> UnstableBookGen <host>
        [build] rustc 0 <host> -> Rustbook <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> WasmComponentLd <host>
        [build] rustc 1 <host> -> std 1 <host>
        [build] rustc 1 <host> -> rustc 2 <host>
        [build] rustc 1 <host> -> WasmComponentLd <host>
        [build] rustdoc 1 <host>
        [doc] std 2 <target1>
        [build] rustc 2 <host> -> std 2 <host>
        [build] rustc 1 <host> -> std 1 <target1>
        [build] rustc 2 <host> -> std 2 <target1>
        [build] rustc 0 <host> -> LintDocs <host>
        [build] rustc 0 <host> -> RustInstaller <host>
        [dist] docs <target1>
        [doc] std 2 <target1>
        [dist] mingw <target1>
        [build] llvm <target1>
        [build] rustc 1 <host> -> rustc 2 <target1>
        [build] rustc 1 <host> -> WasmComponentLd <target1>
        [build] rustdoc 1 <target1>
        [build] rustc 0 <host> -> GenerateCopyright <host>
        [dist] rustc <target1>
        [dist] rustc 1 <host> -> std <target1>
        [dist] src <>
        [build] rustc 0 <host> -> rustfmt 1 <target1>
        [build] rustc 0 <host> -> cargo-fmt 1 <target1>
        [build] rustc 0 <host> -> clippy-driver 1 <target1>
        [build] rustc 0 <host> -> cargo-clippy 1 <target1>
        [build] rustc 0 <host> -> miri 1 <target1>
        [build] rustc 0 <host> -> cargo-miri 1 <target1>
        ");
    }

    #[test]
    fn test_exclude() {
        let ctx = TestCtx::new();
        let steps = ctx.config("test").args(&["--skip", "src/tools/tidy"]).get_steps();

        let host = TargetSelection::from_user(&host_target());
        steps.assert_contains(StepMetadata::test("RustdocUi", host));
        steps.assert_not_contains(test::Tidy);
    }

    #[test]
    fn test_exclude_kind() {
        let ctx = TestCtx::new();
        let host = TargetSelection::from_user(&host_target());

        let get_steps = |args: &[&str]| ctx.config("test").args(args).get_steps();

        // Ensure our test is valid, and `test::Rustc` would be run without the exclude.
        get_steps(&[]).assert_contains(StepMetadata::test("CrateLibrustc", host));

        let steps = get_steps(&["--skip", "compiler/rustc_data_structures"]);

        // Ensure tests for rustc are not skipped.
        steps.assert_contains(StepMetadata::test("CrateLibrustc", host));
        steps.assert_contains_fuzzy(StepMetadata::build("rustc", host));
    }
}

struct ExecutedSteps {
    steps: Vec<ExecutedStep>,
}

impl ExecutedSteps {
    fn render(&self) -> String {
        render_steps(&self.steps)
    }

    #[track_caller]
    fn assert_contains<M: Into<StepMetadata>>(&self, metadata: M) {
        let metadata = metadata.into();
        if !self.contains(&metadata) {
            panic!(
                "Metadata `{}` ({metadata:?}) not found in executed steps:\n{}",
                render_metadata(&metadata),
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
                render_metadata(&metadata),
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
                render_metadata(&metadata),
                self.render()
            );
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
    let StepMetadata { name, kind, target, built_by: _, stage: _ } = executed;
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

/// Renders the executed bootstrap steps for usage in snapshot tests with insta.
/// Only renders certain important steps.
/// Each value in `steps` should be a tuple of (Step, step output).
///
/// The arrow in the rendered output (`X -> Y`) means `X builds Y`.
/// This is similar to the output printed by bootstrap to stdout, but here it is
/// generated purely for the purpose of tests.
fn render_steps(steps: &[ExecutedStep]) -> String {
    steps
        .iter()
        .filter_map(|step| {
            use std::fmt::Write;

            let Some(metadata) = &step.metadata else {
                return None;
            };

            Some(render_metadata(&metadata))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_metadata(metadata: &StepMetadata) -> String {
    let mut record = format!("[{}] ", metadata.kind.as_str());
    if let Some(compiler) = metadata.built_by {
        write!(record, "{} -> ", render_compiler(compiler));
    }
    let stage = if let Some(stage) = metadata.stage { format!("{stage} ") } else { "".to_string() };
    write!(record, "{} {stage}<{}>", metadata.name, normalize_target(metadata.target));
    record
}

fn normalize_target(target: TargetSelection) -> String {
    target
        .to_string()
        .replace(&host_target(), "host")
        .replace(TEST_TRIPLE_1, "target1")
        .replace(TEST_TRIPLE_2, "target2")
}

fn render_compiler(compiler: Compiler) -> String {
    format!("rustc {} <{}>", compiler.stage, normalize_target(compiler.host))
}

fn host_target() -> String {
    get_host_target().to_string()
}
