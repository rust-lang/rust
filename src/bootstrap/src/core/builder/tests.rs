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
fn test_exclude() {
    let mut config = configure("test", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);
    config.skip = vec!["src/tools/tidy".into()];
    let cache = run_build(&[], config);

    // Ensure we have really excluded tidy
    assert!(!cache.contains::<test::Tidy>());

    // Ensure other tests are not affected.
    assert!(cache.contains::<test::RustdocUi>());
}

#[test]
fn test_exclude_kind() {
    let path = PathBuf::from("compiler/rustc_data_structures");

    let mut config = configure("test", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);
    // Ensure our test is valid, and `test::Rustc` would be run without the exclude.
    assert!(run_build(&[], config.clone()).contains::<test::CrateLibrustc>());
    // Ensure tests for rustc are not skipped.
    config.skip = vec![path.clone()];
    assert!(run_build(&[], config.clone()).contains::<test::CrateLibrustc>());
    // Ensure builds for rustc are not skipped.
    assert!(run_build(&[], config).contains::<compile::Rustc>());
}

/// Ensure that if someone passes both a single crate and `library`, all library crates get built.
#[test]
fn alias_and_path_for_library() {
    let mut cache = run_build(
        &["library".into(), "core".into()],
        configure("build", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]),
    );
    assert_eq!(
        first(cache.all::<compile::Std>()),
        &[
            std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
            std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1)
        ]
    );

    let mut cache = run_build(
        &["library".into(), "core".into()],
        configure("doc", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]),
    );
    assert_eq!(
        first(cache.all::<doc::Std>()),
        &[doc_std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1)]
    );
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
    fn build_default() {
        let mut cache = run_build(&[], configure("build", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]));

        let a = TargetSelection::from_user(TEST_TRIPLE_1);
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
            ]
        );
        assert!(!cache.all::<compile::Assemble>().is_empty());
        // Make sure rustdoc is only built once.
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            // Recall that rustdoc stages are off-by-one
            // - this is the compiler it's _linked_ to, not built with.
            &[tool::Rustdoc { compiler: Compiler::new(1, a) }],
        );
        assert_eq!(
            first(cache.all::<compile::Rustc>()),
            &[rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0)],
        );
    }

    #[test]
    fn build_stage_0() {
        let config = Config { stage: 0, ..configure("build", &[TEST_TRIPLE_1], &[TEST_TRIPLE_1]) };
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user(TEST_TRIPLE_1);
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0)]
        );
        assert!(!cache.all::<compile::Assemble>().is_empty());
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            // This is the beta rustdoc.
            // Add an assert here to make sure this is the only rustdoc built.
            &[tool::Rustdoc { compiler: Compiler::new(0, a) }],
        );
        assert!(cache.all::<compile::Rustc>().is_empty());
    }

    #[test]
    fn build_cross_compile() {
        let config = Config {
            stage: 1,
            ..configure("build", &[TEST_TRIPLE_1, TEST_TRIPLE_2], &[TEST_TRIPLE_1, TEST_TRIPLE_2])
        };
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user(TEST_TRIPLE_1);
        let b = TargetSelection::from_user(TEST_TRIPLE_2);

        // Ideally, this build wouldn't actually have `target: a`
        // rustdoc/rustcc/std here (the user only requested a host=B build, so
        // there's not really a need for us to build for target A in this case
        // (since we're producing stage 1 libraries/binaries).  But currently
        // bootstrap is just a bit buggy here; this should be fixed though.
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 0),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 1),
            ]
        );
        assert_eq!(
            first(cache.all::<compile::Assemble>()),
            &[
                compile::Assemble { target_compiler: Compiler::new(0, a) },
                compile::Assemble { target_compiler: Compiler::new(1, a) },
                compile::Assemble { target_compiler: Compiler::new(1, b) },
            ]
        );
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            &[
                tool::Rustdoc { compiler: Compiler::new(1, a) },
                tool::Rustdoc { compiler: Compiler::new(1, b) },
            ],
        );
        assert_eq!(
            first(cache.all::<compile::Rustc>()),
            &[
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 0),
            ]
        );
    }

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
    fn dist_baseline() {
        let mut cache = run_build(&[], configure(&[TEST_TRIPLE_1], &[TEST_TRIPLE_1]));

        let a = TargetSelection::from_user(TEST_TRIPLE_1);

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: a },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: a },]);
        assert_eq!(
            first(cache.all::<dist::Rustc>()),
            &[dist::Rustc { compiler: Compiler::new(2, a) },]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[dist::Std { compiler: Compiler::new(1, a), target: a },]
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
        // Make sure rustdoc is only built once.
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            &[tool::Rustdoc { compiler: Compiler::new(2, a) },]
        );
    }

    #[test]
    fn dist_with_targets() {
        let mut cache =
            run_build(&[], configure(&[TEST_TRIPLE_1], &[TEST_TRIPLE_1, TEST_TRIPLE_2]));

        let a = TargetSelection::from_user(TEST_TRIPLE_1);
        let b = TargetSelection::from_user(TEST_TRIPLE_2);

        assert_eq!(
            first(cache.all::<dist::Docs>()),
            &[dist::Docs { host: a }, dist::Docs { host: b },]
        );
        assert_eq!(
            first(cache.all::<dist::Mingw>()),
            &[dist::Mingw { host: a }, dist::Mingw { host: b },]
        );
        assert_eq!(
            first(cache.all::<dist::Rustc>()),
            &[dist::Rustc { compiler: Compiler::new(2, a) },]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler::new(1, a), target: a },
                dist::Std { compiler: Compiler::new(2, a), target: b },
            ]
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_with_hosts() {
        let mut cache = run_build(
            &[],
            configure(&[TEST_TRIPLE_1, TEST_TRIPLE_2], &[TEST_TRIPLE_1, TEST_TRIPLE_2]),
        );

        let a = TargetSelection::from_user(TEST_TRIPLE_1);
        let b = TargetSelection::from_user(TEST_TRIPLE_2);

        assert_eq!(
            first(cache.all::<dist::Docs>()),
            &[dist::Docs { host: a }, dist::Docs { host: b },]
        );
        assert_eq!(
            first(cache.all::<dist::Mingw>()),
            &[dist::Mingw { host: a }, dist::Mingw { host: b },]
        );
        assert_eq!(
            first(cache.all::<dist::Rustc>()),
            &[
                dist::Rustc { compiler: Compiler::new(2, a) },
                dist::Rustc { compiler: Compiler::new(2, b) },
            ]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler::new(1, a), target: a },
                dist::Std { compiler: Compiler::new(1, a), target: b },
            ]
        );
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 2),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 1),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 2),
            ],
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_only_cross_host() {
        let b = TargetSelection::from_user(TEST_TRIPLE_2);
        let mut config =
            configure(&[TEST_TRIPLE_1, TEST_TRIPLE_2], &[TEST_TRIPLE_1, TEST_TRIPLE_2]);
        config.docs = false;
        config.extended = true;
        config.hosts = vec![b];
        let mut cache = run_build(&[], config);

        assert_eq!(
            first(cache.all::<dist::Rustc>()),
            &[dist::Rustc { compiler: Compiler::new(2, b) },]
        );
        assert_eq!(
            first(cache.all::<compile::Rustc>()),
            &[
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 1),
            ]
        );
    }

    #[test]
    fn dist_with_targets_and_hosts() {
        let mut cache = run_build(
            &[],
            configure(
                &[TEST_TRIPLE_1, TEST_TRIPLE_2],
                &[TEST_TRIPLE_1, TEST_TRIPLE_2, TEST_TRIPLE_3],
            ),
        );

        let a = TargetSelection::from_user(TEST_TRIPLE_1);
        let b = TargetSelection::from_user(TEST_TRIPLE_2);
        let c = TargetSelection::from_user(TEST_TRIPLE_3);

        assert_eq!(
            first(cache.all::<dist::Docs>()),
            &[dist::Docs { host: a }, dist::Docs { host: b }, dist::Docs { host: c },]
        );
        assert_eq!(
            first(cache.all::<dist::Mingw>()),
            &[dist::Mingw { host: a }, dist::Mingw { host: b }, dist::Mingw { host: c },]
        );
        assert_eq!(
            first(cache.all::<dist::Rustc>()),
            &[
                dist::Rustc { compiler: Compiler::new(2, a) },
                dist::Rustc { compiler: Compiler::new(2, b) },
            ]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler::new(1, a), target: a },
                dist::Std { compiler: Compiler::new(1, a), target: b },
                dist::Std { compiler: Compiler::new(2, a), target: c },
            ]
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_with_empty_host() {
        let config = configure(&[], &[TEST_TRIPLE_3]);
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user(TEST_TRIPLE_1);
        let c = TargetSelection::from_user(TEST_TRIPLE_3);

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: c },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: c },]);
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[dist::Std { compiler: Compiler::new(2, a), target: c },]
        );
    }

    #[test]
    fn dist_with_same_targets_and_hosts() {
        let mut cache = run_build(
            &[],
            configure(&[TEST_TRIPLE_1, TEST_TRIPLE_2], &[TEST_TRIPLE_1, TEST_TRIPLE_2]),
        );

        let a = TargetSelection::from_user(TEST_TRIPLE_1);
        let b = TargetSelection::from_user(TEST_TRIPLE_2);

        assert_eq!(
            first(cache.all::<dist::Docs>()),
            &[dist::Docs { host: a }, dist::Docs { host: b },]
        );
        assert_eq!(
            first(cache.all::<dist::Mingw>()),
            &[dist::Mingw { host: a }, dist::Mingw { host: b },]
        );
        assert_eq!(
            first(cache.all::<dist::Rustc>()),
            &[
                dist::Rustc { compiler: Compiler::new(2, a) },
                dist::Rustc { compiler: Compiler::new(2, b) },
            ]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler::new(1, a), target: a },
                dist::Std { compiler: Compiler::new(1, a), target: b },
            ]
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 2),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 1),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 2),
            ]
        );
        assert_eq!(
            first(cache.all::<compile::Assemble>()),
            &[
                compile::Assemble { target_compiler: Compiler::new(0, a) },
                compile::Assemble { target_compiler: Compiler::new(1, a) },
                compile::Assemble { target_compiler: Compiler::new(2, a) },
                compile::Assemble { target_compiler: Compiler::new(2, b) },
            ]
        );
    }

    /// This also serves as an important regression test for <https://github.com/rust-lang/rust/issues/138123>
    /// and <https://github.com/rust-lang/rust/issues/138004>.
    #[test]
    fn dist_all_cross() {
        let cmd_args =
            &["dist", "--stage", "2", "--dry-run", "--config=/does/not/exist"].map(str::to_owned);
        let config_str = r#"
            [rust]
            channel = "nightly"

            [build]
            extended = true

            build = "i686-unknown-haiku"
            host = ["i686-unknown-netbsd"]
            target = ["i686-unknown-netbsd"]
        "#;
        let config = Config::parse_inner(Flags::parse(cmd_args), |&_| toml::from_str(config_str));
        let mut cache = run_build(&[], config);

        // Stage 2 `compile::Rustc` should **NEVER** be cached here.
        assert_eq!(
            first(cache.all::<compile::Rustc>()),
            &[
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_3, stage = 1),
            ]
        );
    }

    #[test]
    fn build_all() {
        let build = Build::new(configure(
            &[TEST_TRIPLE_1, TEST_TRIPLE_2],
            &[TEST_TRIPLE_1, TEST_TRIPLE_2, TEST_TRIPLE_3],
        ));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(
            &Builder::get_step_descriptions(Kind::Build),
            &["compiler/rustc".into(), "library".into()],
        );

        assert_eq!(builder.config.stage, 2);

        // `compile::Rustc` includes one-stage-off compiler information as the target compiler
        // artifacts get copied from there to the target stage sysroot.
        // For example, `stage2/bin/rustc` gets copied from the `stage1-rustc` build directory.
        assert_eq!(
            first(builder.cache.all::<compile::Rustc>()),
            &[
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 1),
            ]
        );

        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 2),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 1),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_2, stage = 2),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_3, stage = 2),
            ]
        );

        assert_eq!(
            first(builder.cache.all::<compile::Assemble>()),
            &[
                compile::Assemble {
                    target_compiler: Compiler::new(0, TargetSelection::from_user(TEST_TRIPLE_1),)
                },
                compile::Assemble {
                    target_compiler: Compiler::new(1, TargetSelection::from_user(TEST_TRIPLE_1),)
                },
                compile::Assemble {
                    target_compiler: Compiler::new(2, TargetSelection::from_user(TEST_TRIPLE_1),)
                },
                compile::Assemble {
                    target_compiler: Compiler::new(2, TargetSelection::from_user(TEST_TRIPLE_2),)
                },
            ]
        );
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
    fn build_with_empty_host() {
        let config = configure(&[], &[TEST_TRIPLE_3]);
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[]);

        let a = TargetSelection::from_user(TEST_TRIPLE_1);

        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
                std!(TEST_TRIPLE_1 => TEST_TRIPLE_3, stage = 2),
            ]
        );
        assert_eq!(
            first(builder.cache.all::<compile::Assemble>()),
            &[
                compile::Assemble { target_compiler: Compiler::new(0, a) },
                compile::Assemble { target_compiler: Compiler::new(1, a) },
                compile::Assemble { target_compiler: Compiler::new(2, a) },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<compile::Rustc>()),
            &[
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 0),
                rustc!(TEST_TRIPLE_1 => TEST_TRIPLE_1, stage = 1),
            ]
        );
    }

    #[test]
    fn test_with_no_doc_stage0() {
        let mut config = configure(&[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);
        config.stage = 0;
        config.paths = vec!["library/std".into()];
        config.cmd = Subcommand::Test {
            test_args: vec![],
            compiletest_rustc_args: vec![],
            no_fail_fast: false,
            no_doc: true,
            doc: false,
            bless: false,
            force_rerun: false,
            compare_mode: None,
            rustfix_coverage: false,
            pass: None,
            run: None,
            only_modified: false,
            extra_checks: None,
            no_capture: false,
        };

        let build = Build::new(config);
        let mut builder = Builder::new(&build);

        let host = TargetSelection::from_user(TEST_TRIPLE_1);

        builder.run_step_descriptions(
            &[StepDescription::from::<test::Crate>(Kind::Test)],
            &["library/std".into()],
        );

        // Ensure we don't build any compiler artifacts.
        assert!(!builder.cache.contains::<compile::Rustc>());
        assert_eq!(
            first(builder.cache.all::<test::Crate>()),
            &[test::Crate {
                compiler: Compiler::new(0, host),
                target: host,
                mode: crate::Mode::Std,
                crates: vec!["std".to_owned()],
            },]
        );
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

    #[test]
    fn test_docs() {
        // Behavior of `x.py test` doing various documentation tests.
        let mut config = configure(&[TEST_TRIPLE_1], &[TEST_TRIPLE_1]);
        config.cmd = Subcommand::Test {
            test_args: vec![],
            compiletest_rustc_args: vec![],
            no_fail_fast: false,
            doc: true,
            no_doc: false,
            bless: false,
            force_rerun: false,
            compare_mode: None,
            rustfix_coverage: false,
            pass: None,
            run: None,
            only_modified: false,
            extra_checks: None,
            no_capture: false,
        };
        // Make sure rustfmt binary not being found isn't an error.
        config.channel = "beta".to_string();
        let build = Build::new(config);
        let mut builder = Builder::new(&build);

        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Test), &[]);
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
        // Unfortunately rustdoc is built twice. Once from stage1 for compiletest
        // (and other things), and once from stage0 for std crates. Ideally it
        // would only be built once. If someone wants to fix this, it might be
        // worth investigating if it would be possible to test std from stage1.
        // Note that the stages here are +1 than what they actually are because
        // Rustdoc::run swaps out the compiler with stage minus 1 if --stage is
        // not 0.
        //
        // The stage 0 copy is the one downloaded for bootstrapping. It is
        // (currently) needed to run "cargo test" on the linkchecker, and
        // should be relatively "free".
        assert_eq!(
            first(builder.cache.all::<tool::Rustdoc>()),
            &[
                tool::Rustdoc { compiler: Compiler::new(0, a) },
                tool::Rustdoc { compiler: Compiler::new(1, a) },
                tool::Rustdoc { compiler: Compiler::new(2, a) },
            ]
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

/// The staging tests use insta for snapshot testing.
/// See bootstrap's README on how to bless the snapshots.
mod staging {
    use crate::Build;
    use crate::core::builder::Builder;
    use crate::core::builder::tests::{
        TEST_TRIPLE_1, configure, configure_with_args, render_steps, run_build,
    };
    use crate::utils::tests::{ConfigBuilder, TestCtx};

    #[test]
    fn build_compiler_stage_1() {
        let ctx = TestCtx::new();
        insta::assert_snapshot!(
            ctx.config("build")
                .path("compiler")
                .stage(1)
                .get_steps(), @r"
        [build] rustc 0 <host> -> std 0 <host>
        [build] llvm <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        [build] rustc 0 <host> -> rustc 1 <host>
        ");
    }

    impl ConfigBuilder {
        fn get_steps(self) -> String {
            let config = self.create_config();

            let kind = config.cmd.kind();
            let build = Build::new(config);
            let builder = Builder::new(&build);
            builder.run_step_descriptions(&Builder::get_step_descriptions(kind), &builder.paths);
            render_steps(&builder.cache.into_executed_steps())
        }
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

            let mut record = format!("[{}] ", metadata.kind.as_str());
            if let Some(compiler) = metadata.built_by {
                write!(record, "{} -> ", render_compiler(compiler));
            }
            let stage =
                if let Some(stage) = metadata.stage { format!("{stage} ") } else { "".to_string() };
            write!(record, "{} {stage}<{}>", metadata.name, normalize_target(metadata.target));
            Some(record)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_target(target: TargetSelection) -> String {
    target.to_string().replace(&get_host_target().to_string(), "host")
}

fn render_compiler(compiler: Compiler) -> String {
    format!("rustc {} <{}>", compiler.stage, normalize_target(compiler.host))
}
