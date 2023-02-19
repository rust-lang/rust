use super::*;
use crate::config::{Config, DryRun, TargetSelection};
use std::thread;

fn configure(cmd: &str, host: &[&str], target: &[&str]) -> Config {
    configure_with_args(&[cmd.to_owned()], host, target)
}

fn configure_with_args(cmd: &[String], host: &[&str], target: &[&str]) -> Config {
    let mut config = Config::parse(cmd);
    // don't save toolstates
    config.save_toolstates = None;
    config.dry_run = DryRun::SelfCheck;

    // Ignore most submodules, since we don't need them for a dry run.
    // But make sure to check out the `doc` and `rust-analyzer` submodules, since some steps need them
    // just to know which commands to run.
    let submodule_build = Build::new(Config {
        // don't include LLVM, so CI doesn't require ninja/cmake to be installed
        rust_codegen_backends: vec![],
        ..Config::parse(&["check".to_owned()])
    });
    submodule_build.update_submodule(Path::new("src/doc/book"));
    submodule_build.update_submodule(Path::new("src/tools/rust-analyzer"));
    config.submodules = Some(false);

    config.ninja_in_file = false;
    // try to avoid spurious failures in dist where we create/delete each others file
    // HACK: rather than pull in `tempdir`, use the one that cargo has conveniently created for us
    let dir = Path::new(env!("OUT_DIR"))
        .join("tmp-rustbuild-tests")
        .join(&thread::current().name().unwrap_or("unknown").replace(":", "-"));
    t!(fs::create_dir_all(&dir));
    config.out = dir;
    config.build = TargetSelection::from_user("A");
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
        configure_with_args(&paths.map(String::from), &["A"], &["A"]),
    );
}

macro_rules! std {
    ($host:ident => $target:ident, stage = $stage:literal) => {
        compile::Std::new(
            Compiler { host: TargetSelection::from_user(stringify!($host)), stage: $stage },
            TargetSelection::from_user(stringify!($target)),
        )
    };
}

macro_rules! rustc {
    ($host:ident => $target:ident, stage = $stage:literal) => {
        compile::Rustc::new(
            Compiler { host: TargetSelection::from_user(stringify!($host)), stage: $stage },
            TargetSelection::from_user(stringify!($target)),
        )
    };
}

#[test]
fn test_valid() {
    // make sure multi suite paths are accepted
    check_cli(["test", "tests/ui/attr-start.rs", "tests/ui/attr-shebang.rs"]);
}

#[test]
#[should_panic]
fn test_invalid() {
    // make sure that invalid paths are caught, even when combined with valid paths
    check_cli(["test", "library/std", "x"]);
}

#[test]
fn test_intersection() {
    let set = PathSet::Set(
        ["library/core", "library/alloc", "library/std"].into_iter().map(TaskPath::parse).collect(),
    );
    let mut command_paths =
        vec![Path::new("library/core"), Path::new("library/alloc"), Path::new("library/stdarch")];
    let subset = set.intersection_removing_matches(&mut command_paths, None);
    assert_eq!(
        subset,
        PathSet::Set(["library/core", "library/alloc"].into_iter().map(TaskPath::parse).collect())
    );
    assert_eq!(command_paths, vec![Path::new("library/stdarch")]);
}

#[test]
fn test_exclude() {
    let mut config = configure("test", &["A"], &["A"]);
    config.exclude = vec![TaskPath::parse("src/tools/tidy")];
    let cache = run_build(&[], config);

    // Ensure we have really excluded tidy
    assert!(!cache.contains::<test::Tidy>());

    // Ensure other tests are not affected.
    assert!(cache.contains::<test::RustdocUi>());
}

#[test]
fn test_exclude_kind() {
    let path = PathBuf::from("src/tools/cargotest");
    let exclude = TaskPath::parse("test::src/tools/cargotest");
    assert_eq!(exclude, TaskPath { kind: Some(Kind::Test), path: path.clone() });

    let mut config = configure("test", &["A"], &["A"]);
    // Ensure our test is valid, and `test::Cargotest` would be run without the exclude.
    assert!(run_build(&[path.clone()], config.clone()).contains::<test::Cargotest>());
    // Ensure tests for cargotest are skipped.
    config.exclude = vec![exclude.clone()];
    assert!(!run_build(&[path.clone()], config).contains::<test::Cargotest>());

    // Ensure builds for cargotest are not skipped.
    let mut config = configure("build", &["A"], &["A"]);
    config.exclude = vec![exclude];
    assert!(run_build(&[path], config).contains::<tool::CargoTest>());
}

/// Ensure that if someone passes both a single crate and `library`, all library crates get built.
#[test]
fn alias_and_path_for_library() {
    let mut cache =
        run_build(&["library".into(), "core".into()], configure("build", &["A"], &["A"]));
    assert_eq!(
        first(cache.all::<compile::Std>()),
        &[std!(A => A, stage = 0), std!(A => A, stage = 1)]
    );
}

mod defaults {
    use super::{configure, first, run_build};
    use crate::builder::*;
    use crate::Config;
    use pretty_assertions::assert_eq;

    #[test]
    fn build_default() {
        let mut cache = run_build(&[], configure("build", &["A"], &["A"]));

        let a = TargetSelection::from_user("A");
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[std!(A => A, stage = 0), std!(A => A, stage = 1),]
        );
        assert!(!cache.all::<compile::Assemble>().is_empty());
        // Make sure rustdoc is only built once.
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            // Recall that rustdoc stages are off-by-one
            // - this is the compiler it's _linked_ to, not built with.
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 1 } }],
        );
        assert_eq!(first(cache.all::<compile::Rustc>()), &[rustc!(A => A, stage = 0)],);
    }

    #[test]
    fn build_stage_0() {
        let config = Config { stage: 0, ..configure("build", &["A"], &["A"]) };
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user("A");
        assert_eq!(first(cache.all::<compile::Std>()), &[std!(A => A, stage = 0)]);
        assert!(!cache.all::<compile::Assemble>().is_empty());
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            // This is the beta rustdoc.
            // Add an assert here to make sure this is the only rustdoc built.
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 0 } }],
        );
        assert!(cache.all::<compile::Rustc>().is_empty());
    }

    #[test]
    fn build_cross_compile() {
        let config = Config { stage: 1, ..configure("build", &["A", "B"], &["A", "B"]) };
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");

        // Ideally, this build wouldn't actually have `target: a`
        // rustdoc/rustcc/std here (the user only requested a host=B build, so
        // there's not really a need for us to build for target A in this case
        // (since we're producing stage 1 libraries/binaries).  But currently
        // rustbuild is just a bit buggy here; this should be fixed though.
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[
                std!(A => A, stage = 0),
                std!(A => A, stage = 1),
                std!(A => B, stage = 0),
                std!(A => B, stage = 1),
            ]
        );
        assert_eq!(
            first(cache.all::<compile::Assemble>()),
            &[
                compile::Assemble { target_compiler: Compiler { host: a, stage: 0 } },
                compile::Assemble { target_compiler: Compiler { host: a, stage: 1 } },
                compile::Assemble { target_compiler: Compiler { host: b, stage: 1 } },
            ]
        );
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            &[
                tool::Rustdoc { compiler: Compiler { host: a, stage: 1 } },
                tool::Rustdoc { compiler: Compiler { host: b, stage: 1 } },
            ],
        );
        assert_eq!(
            first(cache.all::<compile::Rustc>()),
            &[rustc!(A => A, stage = 0), rustc!(A => B, stage = 0),]
        );
    }

    #[test]
    fn doc_default() {
        let mut config = configure("doc", &["A"], &["A"]);
        config.compiler_docs = true;
        config.cmd = Subcommand::Doc { paths: Vec::new(), open: false, json: false };
        let mut cache = run_build(&[], config);
        let a = TargetSelection::from_user("A");

        // error_index_generator uses stage 0 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(first(cache.all::<doc::ErrorIndex>()), &[doc::ErrorIndex { target: a },]);
        assert_eq!(
            first(cache.all::<tool::ErrorIndex>()),
            &[tool::ErrorIndex { compiler: Compiler { host: a, stage: 0 } }]
        );
        // docs should be built with the beta compiler, not with the stage0 artifacts.
        // recall that rustdoc is off-by-one: `stage` is the compiler rustdoc is _linked_ to,
        // not the one it was built by.
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 0 } },]
        );
    }
}

mod dist {
    use super::{first, run_build, Config};
    use crate::builder::*;
    use pretty_assertions::assert_eq;

    fn configure(host: &[&str], target: &[&str]) -> Config {
        Config { stage: 2, ..super::configure("dist", host, target) }
    }

    #[test]
    fn dist_baseline() {
        let mut cache = run_build(&[], configure(&["A"], &["A"]));

        let a = TargetSelection::from_user("A");

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: a },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: a },]);
        assert_eq!(
            first(cache.all::<dist::Rustc>()),
            &[dist::Rustc { compiler: Compiler { host: a, stage: 2 } },]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },]
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
        // Make sure rustdoc is only built once.
        assert_eq!(
            first(cache.all::<tool::Rustdoc>()),
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 2 } },]
        );
    }

    #[test]
    fn dist_with_targets() {
        let mut cache = run_build(&[], configure(&["A"], &["A", "B"]));

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");

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
            &[dist::Rustc { compiler: Compiler { host: a, stage: 2 } },]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                dist::Std { compiler: Compiler { host: a, stage: 2 }, target: b },
            ]
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_with_hosts() {
        let mut cache = run_build(&[], configure(&["A", "B"], &["A", "B"]));

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");

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
                dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
                dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
            ]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
            ]
        );
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[
                std!(A => A, stage = 0),
                std!(A => A, stage = 1),
                std!(A => A, stage = 2),
                std!(A => B, stage = 1),
                std!(A => B, stage = 2),
            ],
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_only_cross_host() {
        let b = TargetSelection::from_user("B");
        let mut config = configure(&["A", "B"], &["A", "B"]);
        config.docs = false;
        config.extended = true;
        config.hosts = vec![b];
        let mut cache = run_build(&[], config);

        assert_eq!(
            first(cache.all::<dist::Rustc>()),
            &[dist::Rustc { compiler: Compiler { host: b, stage: 2 } },]
        );
        assert_eq!(
            first(cache.all::<compile::Rustc>()),
            &[rustc!(A => A, stage = 0), rustc!(A => B, stage = 1),]
        );
    }

    #[test]
    fn dist_with_targets_and_hosts() {
        let mut cache = run_build(&[], configure(&["A", "B"], &["A", "B", "C"]));

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");
        let c = TargetSelection::from_user("C");

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
                dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
                dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
            ]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
                dist::Std { compiler: Compiler { host: a, stage: 2 }, target: c },
            ]
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_with_empty_host() {
        let config = configure(&[], &["C"]);
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user("A");
        let c = TargetSelection::from_user("C");

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: c },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: c },]);
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[dist::Std { compiler: Compiler { host: a, stage: 2 }, target: c },]
        );
    }

    #[test]
    fn dist_with_same_targets_and_hosts() {
        let mut cache = run_build(&[], configure(&["A", "B"], &["A", "B"]));

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");

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
                dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
                dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
            ]
        );
        assert_eq!(
            first(cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
            ]
        );
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
        assert_eq!(
            first(cache.all::<compile::Std>()),
            &[
                std!(A => A, stage = 0),
                std!(A => A, stage = 1),
                std!(A => A, stage = 2),
                std!(A => B, stage = 1),
                std!(A => B, stage = 2),
            ]
        );
        assert_eq!(
            first(cache.all::<compile::Assemble>()),
            &[
                compile::Assemble { target_compiler: Compiler { host: a, stage: 0 } },
                compile::Assemble { target_compiler: Compiler { host: a, stage: 1 } },
                compile::Assemble { target_compiler: Compiler { host: a, stage: 2 } },
                compile::Assemble { target_compiler: Compiler { host: b, stage: 2 } },
            ]
        );
    }

    #[test]
    fn build_all() {
        let build = Build::new(configure(&["A", "B"], &["A", "B", "C"]));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(
            &Builder::get_step_descriptions(Kind::Build),
            &["compiler/rustc".into(), "library".into()],
        );

        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                std!(A => A, stage = 0),
                std!(A => A, stage = 1),
                std!(A => A, stage = 2),
                std!(A => B, stage = 1),
                std!(A => B, stage = 2),
                std!(A => C, stage = 2),
            ]
        );
        assert_eq!(builder.cache.all::<compile::Assemble>().len(), 5);
        assert_eq!(
            first(builder.cache.all::<compile::Rustc>()),
            &[
                rustc!(A => A, stage = 0),
                rustc!(A => A, stage = 1),
                rustc!(A => A, stage = 2),
                rustc!(A => B, stage = 1),
                rustc!(A => B, stage = 2),
            ]
        );
    }

    #[test]
    fn build_with_empty_host() {
        let config = configure(&[], &["C"]);
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[]);

        let a = TargetSelection::from_user("A");

        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[std!(A => A, stage = 0), std!(A => A, stage = 1), std!(A => C, stage = 2),]
        );
        assert_eq!(
            first(builder.cache.all::<compile::Assemble>()),
            &[
                compile::Assemble { target_compiler: Compiler { host: a, stage: 0 } },
                compile::Assemble { target_compiler: Compiler { host: a, stage: 1 } },
                compile::Assemble { target_compiler: Compiler { host: a, stage: 2 } },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<compile::Rustc>()),
            &[rustc!(A => A, stage = 0), rustc!(A => A, stage = 1),]
        );
    }

    #[test]
    fn test_with_no_doc_stage0() {
        let mut config = configure(&["A"], &["A"]);
        config.stage = 0;
        config.cmd = Subcommand::Test {
            paths: vec!["library/std".into()],
            test_args: vec![],
            rustc_args: vec![],
            fail_fast: true,
            doc_tests: DocTests::No,
            bless: false,
            force_rerun: false,
            compare_mode: None,
            rustfix_coverage: false,
            pass: None,
            run: None,
            only_modified: false,
        };

        let build = Build::new(config);
        let mut builder = Builder::new(&build);

        let host = TargetSelection::from_user("A");

        builder.run_step_descriptions(
            &[StepDescription::from::<test::Crate>(Kind::Test)],
            &["library/std".into()],
        );

        // Ensure we don't build any compiler artifacts.
        assert!(!builder.cache.contains::<compile::Rustc>());
        assert_eq!(
            first(builder.cache.all::<test::Crate>()),
            &[test::Crate {
                compiler: Compiler { host, stage: 0 },
                target: host,
                mode: Mode::Std,
                test_kind: test::TestKind::Test,
                crates: vec![INTERNER.intern_str("std")],
            },]
        );
    }

    #[test]
    fn doc_ci() {
        let mut config = configure(&["A"], &["A"]);
        config.compiler_docs = true;
        config.cmd = Subcommand::Doc { paths: Vec::new(), open: false, json: false };
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Doc), &[]);
        let a = TargetSelection::from_user("A");

        // error_index_generator uses stage 1 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(
            first(builder.cache.all::<doc::ErrorIndex>()),
            &[doc::ErrorIndex { target: a },]
        );
        assert_eq!(
            first(builder.cache.all::<tool::ErrorIndex>()),
            &[tool::ErrorIndex { compiler: Compiler { host: a, stage: 1 } }]
        );
        // This is actually stage 1, but Rustdoc::run swaps out the compiler with
        // stage minus 1 if --stage is not 0. Very confusing!
        assert_eq!(
            first(builder.cache.all::<tool::Rustdoc>()),
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 2 } },]
        );
    }

    #[test]
    fn test_docs() {
        // Behavior of `x.py test` doing various documentation tests.
        let mut config = configure(&["A"], &["A"]);
        config.cmd = Subcommand::Test {
            paths: vec![],
            test_args: vec![],
            rustc_args: vec![],
            fail_fast: true,
            doc_tests: DocTests::Yes,
            bless: false,
            force_rerun: false,
            compare_mode: None,
            rustfix_coverage: false,
            pass: None,
            run: None,
            only_modified: false,
        };
        // Make sure rustfmt binary not being found isn't an error.
        config.channel = "beta".to_string();
        let build = Build::new(config);
        let mut builder = Builder::new(&build);

        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Test), &[]);
        let a = TargetSelection::from_user("A");

        // error_index_generator uses stage 1 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(
            first(builder.cache.all::<doc::ErrorIndex>()),
            &[doc::ErrorIndex { target: a },]
        );
        assert_eq!(
            first(builder.cache.all::<tool::ErrorIndex>()),
            &[tool::ErrorIndex { compiler: Compiler { host: a, stage: 1 } }]
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
                tool::Rustdoc { compiler: Compiler { host: a, stage: 0 } },
                tool::Rustdoc { compiler: Compiler { host: a, stage: 1 } },
                tool::Rustdoc { compiler: Compiler { host: a, stage: 2 } },
            ]
        );
    }
}
