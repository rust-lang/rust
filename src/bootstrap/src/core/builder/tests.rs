use std::thread;

use super::*;
use crate::Flags;
use crate::core::build_steps::doc::DocumentationFormat;
use crate::core::config::Config;

fn configure(cmd: &str, host: &[&str], target: &[&str]) -> Config {
    configure_with_args(&[cmd.to_owned()], host, target)
}

fn configure_with_args(cmd: &[String], host: &[&str], target: &[&str]) -> Config {
    let mut config = Config::parse(Flags::parse(cmd));
    // don't save toolstates
    config.save_toolstates = None;
    config.dry_run = DryRun::SelfCheck;

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
    config.build = TargetSelection::from_user("A-A");
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
        configure_with_args(&paths.map(String::from), &["A-A"], &["A-A"]),
    );
}

macro_rules! std {
    ($host:ident => $target:ident, stage = $stage:literal) => {
        compile::Std::new(
            Compiler {
                host: TargetSelection::from_user(concat!(
                    stringify!($host),
                    "-",
                    stringify!($host)
                )),
                stage: $stage,
            },
            TargetSelection::from_user(concat!(stringify!($target), "-", stringify!($target))),
        )
    };
}

macro_rules! doc_std {
    ($host:ident => $target:ident, stage = $stage:literal) => {{
        doc::Std::new(
            $stage,
            TargetSelection::from_user(concat!(stringify!($target), "-", stringify!($target))),
            DocumentationFormat::Html,
        )
    }};
}

macro_rules! rustc {
    ($host:ident => $target:ident, stage = $stage:literal) => {
        compile::Rustc::new(
            Compiler {
                host: TargetSelection::from_user(concat!(
                    stringify!($host),
                    "-",
                    stringify!($host)
                )),
                stage: $stage,
            },
            TargetSelection::from_user(concat!(stringify!($target), "-", stringify!($target))),
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
    let set = |paths: &[&str]| {
        PathSet::Set(paths.into_iter().map(|p| TaskPath { path: p.into(), kind: None }).collect())
    };
    let library_set = set(&["library/core", "library/alloc", "library/std"]);
    let mut command_paths = vec![
        PathBuf::from("library/core"),
        PathBuf::from("library/alloc"),
        PathBuf::from("library/stdarch"),
    ];
    let subset = library_set.intersection_removing_matches(&mut command_paths, Kind::Build);
    assert_eq!(subset, set(&["library/core", "library/alloc"]),);
    assert_eq!(command_paths, vec![PathBuf::from("library/stdarch")]);
}

#[test]
fn validate_path_remap() {
    let build = Build::new(configure("test", &["A-A"], &["A-A"]));

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
    let build = Build::new(configure("test", &["A-A"], &["A-A"]));

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
    let mut config = configure("test", &["A-A"], &["A-A"]);
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

    let mut config = configure("test", &["A-A"], &["A-A"]);
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
    let mut cache =
        run_build(&["library".into(), "core".into()], configure("build", &["A-A"], &["A-A"]));
    assert_eq!(first(cache.all::<compile::Std>()), &[
        std!(A => A, stage = 0),
        std!(A => A, stage = 1)
    ]);

    let mut cache =
        run_build(&["library".into(), "core".into()], configure("doc", &["A-A"], &["A-A"]));
    assert_eq!(first(cache.all::<doc::Std>()), &[doc_std!(A => A, stage = 0)]);
}

#[test]
fn ci_rustc_if_unchanged_logic() {
    let config = Config::parse_inner(
        Flags::parse(&[
            "build".to_owned(),
            "--dry-run".to_owned(),
            "--set=rust.download-rustc='if-unchanged'".to_owned(),
        ]),
        |&_| Ok(Default::default()),
    );

    let build = Build::new(config.clone());
    let builder = Builder::new(&build);

    if config.out.exists() {
        fs::remove_dir_all(&config.out).unwrap();
    }

    builder.run_step_descriptions(&Builder::get_step_descriptions(config.cmd.kind()), &[]);

    // Make sure "if-unchanged" logic doesn't try to use CI rustc while there are changes
    // in compiler and/or library.
    if config.download_rustc_commit.is_some() {
        let has_changes =
            config.last_modified_commit(&["compiler", "library"], "download-rustc", true).is_none();

        assert!(
            !has_changes,
            "CI-rustc can't be used with 'if-unchanged' while there are changes in compiler and/or library."
        );
    }
}

mod defaults {
    use pretty_assertions::assert_eq;

    use super::{configure, first, run_build};
    use crate::Config;
    use crate::core::builder::*;

    #[test]
    fn build_default() {
        let mut cache = run_build(&[], configure("build", &["A-A"], &["A-A"]));

        let a = TargetSelection::from_user("A-A");
        assert_eq!(first(cache.all::<compile::Std>()), &[
            std!(A => A, stage = 0),
            std!(A => A, stage = 1),
        ]);
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
        let config = Config { stage: 0, ..configure("build", &["A-A"], &["A-A"]) };
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user("A-A");
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
        let config = Config { stage: 1, ..configure("build", &["A-A", "B-B"], &["A-A", "B-B"]) };
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user("A-A");
        let b = TargetSelection::from_user("B-B");

        // Ideally, this build wouldn't actually have `target: a`
        // rustdoc/rustcc/std here (the user only requested a host=B build, so
        // there's not really a need for us to build for target A in this case
        // (since we're producing stage 1 libraries/binaries).  But currently
        // bootstrap is just a bit buggy here; this should be fixed though.
        assert_eq!(first(cache.all::<compile::Std>()), &[
            std!(A => A, stage = 0),
            std!(A => A, stage = 1),
            std!(A => B, stage = 0),
            std!(A => B, stage = 1),
        ]);
        assert_eq!(first(cache.all::<compile::Assemble>()), &[
            compile::Assemble { target_compiler: Compiler { host: a, stage: 0 } },
            compile::Assemble { target_compiler: Compiler { host: a, stage: 1 } },
            compile::Assemble { target_compiler: Compiler { host: b, stage: 1 } },
        ]);
        assert_eq!(first(cache.all::<tool::Rustdoc>()), &[
            tool::Rustdoc { compiler: Compiler { host: a, stage: 1 } },
            tool::Rustdoc { compiler: Compiler { host: b, stage: 1 } },
        ],);
        assert_eq!(first(cache.all::<compile::Rustc>()), &[
            rustc!(A => A, stage = 0),
            rustc!(A => B, stage = 0),
        ]);
    }

    #[test]
    fn doc_default() {
        let mut config = configure("doc", &["A-A"], &["A-A"]);
        config.compiler_docs = true;
        config.cmd = Subcommand::Doc { open: false, json: false };
        let mut cache = run_build(&[], config);
        let a = TargetSelection::from_user("A-A");

        // error_index_generator uses stage 0 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(first(cache.all::<doc::ErrorIndex>()), &[doc::ErrorIndex { target: a },]);
        assert_eq!(first(cache.all::<tool::ErrorIndex>()), &[tool::ErrorIndex {
            compiler: Compiler { host: a, stage: 0 }
        }]);
        // docs should be built with the beta compiler, not with the stage0 artifacts.
        // recall that rustdoc is off-by-one: `stage` is the compiler rustdoc is _linked_ to,
        // not the one it was built by.
        assert_eq!(first(cache.all::<tool::Rustdoc>()), &[tool::Rustdoc {
            compiler: Compiler { host: a, stage: 0 }
        },]);
    }
}

mod dist {
    use pretty_assertions::assert_eq;

    use super::{Config, first, run_build};
    use crate::core::builder::*;

    fn configure(host: &[&str], target: &[&str]) -> Config {
        Config { stage: 2, ..super::configure("dist", host, target) }
    }

    #[test]
    fn dist_baseline() {
        let mut cache = run_build(&[], configure(&["A-A"], &["A-A"]));

        let a = TargetSelection::from_user("A-A");

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: a },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: a },]);
        assert_eq!(first(cache.all::<dist::Rustc>()), &[dist::Rustc {
            compiler: Compiler { host: a, stage: 2 }
        },]);
        assert_eq!(first(cache.all::<dist::Std>()), &[dist::Std {
            compiler: Compiler { host: a, stage: 1 },
            target: a
        },]);
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
        // Make sure rustdoc is only built once.
        assert_eq!(first(cache.all::<tool::Rustdoc>()), &[tool::Rustdoc {
            compiler: Compiler { host: a, stage: 2 }
        },]);
    }

    #[test]
    fn dist_with_targets() {
        let mut cache = run_build(&[], configure(&["A-A"], &["A-A", "B-B"]));

        let a = TargetSelection::from_user("A-A");
        let b = TargetSelection::from_user("B-B");

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: a }, dist::Docs {
            host: b
        },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: a }, dist::Mingw {
            host: b
        },]);
        assert_eq!(first(cache.all::<dist::Rustc>()), &[dist::Rustc {
            compiler: Compiler { host: a, stage: 2 }
        },]);
        assert_eq!(first(cache.all::<dist::Std>()), &[
            dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
            dist::Std { compiler: Compiler { host: a, stage: 2 }, target: b },
        ]);
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_with_hosts() {
        let mut cache = run_build(&[], configure(&["A-A", "B-B"], &["A-A", "B-B"]));

        let a = TargetSelection::from_user("A-A");
        let b = TargetSelection::from_user("B-B");

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: a }, dist::Docs {
            host: b
        },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: a }, dist::Mingw {
            host: b
        },]);
        assert_eq!(first(cache.all::<dist::Rustc>()), &[
            dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
            dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
        ]);
        assert_eq!(first(cache.all::<dist::Std>()), &[
            dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
            dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
        ]);
        assert_eq!(first(cache.all::<compile::Std>()), &[
            std!(A => A, stage = 0),
            std!(A => A, stage = 1),
            std!(A => A, stage = 2),
            std!(A => B, stage = 1),
            std!(A => B, stage = 2),
        ],);
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_only_cross_host() {
        let b = TargetSelection::from_user("B-B");
        let mut config = configure(&["A-A", "B-B"], &["A-A", "B-B"]);
        config.docs = false;
        config.extended = true;
        config.hosts = vec![b];
        let mut cache = run_build(&[], config);

        assert_eq!(first(cache.all::<dist::Rustc>()), &[dist::Rustc {
            compiler: Compiler { host: b, stage: 2 }
        },]);
        assert_eq!(first(cache.all::<compile::Rustc>()), &[
            rustc!(A => A, stage = 0),
            rustc!(A => B, stage = 1),
        ]);
    }

    #[test]
    fn dist_with_targets_and_hosts() {
        let mut cache = run_build(&[], configure(&["A-A", "B-B"], &["A-A", "B-B", "C-C"]));

        let a = TargetSelection::from_user("A-A");
        let b = TargetSelection::from_user("B-B");
        let c = TargetSelection::from_user("C-C");

        assert_eq!(first(cache.all::<dist::Docs>()), &[
            dist::Docs { host: a },
            dist::Docs { host: b },
            dist::Docs { host: c },
        ]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[
            dist::Mingw { host: a },
            dist::Mingw { host: b },
            dist::Mingw { host: c },
        ]);
        assert_eq!(first(cache.all::<dist::Rustc>()), &[
            dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
            dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
        ]);
        assert_eq!(first(cache.all::<dist::Std>()), &[
            dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
            dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
            dist::Std { compiler: Compiler { host: a, stage: 2 }, target: c },
        ]);
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_with_empty_host() {
        let config = configure(&[], &["C-C"]);
        let mut cache = run_build(&[], config);

        let a = TargetSelection::from_user("A-A");
        let c = TargetSelection::from_user("C-C");

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: c },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: c },]);
        assert_eq!(first(cache.all::<dist::Std>()), &[dist::Std {
            compiler: Compiler { host: a, stage: 2 },
            target: c
        },]);
    }

    #[test]
    fn dist_with_same_targets_and_hosts() {
        let mut cache = run_build(&[], configure(&["A-A", "B-B"], &["A-A", "B-B"]));

        let a = TargetSelection::from_user("A-A");
        let b = TargetSelection::from_user("B-B");

        assert_eq!(first(cache.all::<dist::Docs>()), &[dist::Docs { host: a }, dist::Docs {
            host: b
        },]);
        assert_eq!(first(cache.all::<dist::Mingw>()), &[dist::Mingw { host: a }, dist::Mingw {
            host: b
        },]);
        assert_eq!(first(cache.all::<dist::Rustc>()), &[
            dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
            dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
        ]);
        assert_eq!(first(cache.all::<dist::Std>()), &[
            dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
            dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
        ]);
        assert_eq!(first(cache.all::<dist::Src>()), &[dist::Src]);
        assert_eq!(first(cache.all::<compile::Std>()), &[
            std!(A => A, stage = 0),
            std!(A => A, stage = 1),
            std!(A => A, stage = 2),
            std!(A => B, stage = 1),
            std!(A => B, stage = 2),
        ]);
        assert_eq!(first(cache.all::<compile::Assemble>()), &[
            compile::Assemble { target_compiler: Compiler { host: a, stage: 0 } },
            compile::Assemble { target_compiler: Compiler { host: a, stage: 1 } },
            compile::Assemble { target_compiler: Compiler { host: a, stage: 2 } },
            compile::Assemble { target_compiler: Compiler { host: b, stage: 2 } },
        ]);
    }

    #[test]
    fn build_all() {
        let build = Build::new(configure(&["A-A", "B-B"], &["A-A", "B-B", "C-C"]));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[
            "compiler/rustc".into(),
            "library".into(),
        ]);

        assert_eq!(first(builder.cache.all::<compile::Std>()), &[
            std!(A => A, stage = 0),
            std!(A => A, stage = 1),
            std!(A => A, stage = 2),
            std!(A => B, stage = 1),
            std!(A => B, stage = 2),
            std!(A => C, stage = 2),
        ]);
        assert_eq!(builder.cache.all::<compile::Assemble>().len(), 5);
        assert_eq!(first(builder.cache.all::<compile::Rustc>()), &[
            rustc!(A => A, stage = 0),
            rustc!(A => A, stage = 1),
            rustc!(A => A, stage = 2),
            rustc!(A => B, stage = 1),
            rustc!(A => B, stage = 2),
        ]);
    }

    #[test]
    fn llvm_out_behaviour() {
        let mut config = configure(&["A-A"], &["B-B"]);
        config.llvm_from_ci = true;
        let build = Build::new(config.clone());

        let target = TargetSelection::from_user("A-A");
        assert!(build.llvm_out(target).ends_with("ci-llvm"));
        let target = TargetSelection::from_user("B-B");
        assert!(build.llvm_out(target).ends_with("llvm"));

        config.llvm_from_ci = false;
        let build = Build::new(config.clone());
        let target = TargetSelection::from_user("A-A");
        assert!(build.llvm_out(target).ends_with("llvm"));
    }

    #[test]
    fn build_with_empty_host() {
        let config = configure(&[], &["C-C"]);
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[]);

        let a = TargetSelection::from_user("A-A");

        assert_eq!(first(builder.cache.all::<compile::Std>()), &[
            std!(A => A, stage = 0),
            std!(A => A, stage = 1),
            std!(A => C, stage = 2),
        ]);
        assert_eq!(first(builder.cache.all::<compile::Assemble>()), &[
            compile::Assemble { target_compiler: Compiler { host: a, stage: 0 } },
            compile::Assemble { target_compiler: Compiler { host: a, stage: 1 } },
            compile::Assemble { target_compiler: Compiler { host: a, stage: 2 } },
        ]);
        assert_eq!(first(builder.cache.all::<compile::Rustc>()), &[
            rustc!(A => A, stage = 0),
            rustc!(A => A, stage = 1),
        ]);
    }

    #[test]
    fn test_with_no_doc_stage0() {
        let mut config = configure(&["A-A"], &["A-A"]);
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
        };

        let build = Build::new(config);
        let mut builder = Builder::new(&build);

        let host = TargetSelection::from_user("A-A");

        builder.run_step_descriptions(&[StepDescription::from::<test::Crate>(Kind::Test)], &[
            "library/std".into(),
        ]);

        // Ensure we don't build any compiler artifacts.
        assert!(!builder.cache.contains::<compile::Rustc>());
        assert_eq!(first(builder.cache.all::<test::Crate>()), &[test::Crate {
            compiler: Compiler { host, stage: 0 },
            target: host,
            mode: Mode::Std,
            crates: vec!["std".to_owned()],
        },]);
    }

    #[test]
    fn doc_ci() {
        let mut config = configure(&["A-A"], &["A-A"]);
        config.compiler_docs = true;
        config.cmd = Subcommand::Doc { open: false, json: false };
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Doc), &[]);
        let a = TargetSelection::from_user("A-A");

        // error_index_generator uses stage 1 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(
            first(builder.cache.all::<doc::ErrorIndex>()),
            &[doc::ErrorIndex { target: a },]
        );
        assert_eq!(first(builder.cache.all::<tool::ErrorIndex>()), &[tool::ErrorIndex {
            compiler: Compiler { host: a, stage: 1 }
        }]);
        // This is actually stage 1, but Rustdoc::run swaps out the compiler with
        // stage minus 1 if --stage is not 0. Very confusing!
        assert_eq!(first(builder.cache.all::<tool::Rustdoc>()), &[tool::Rustdoc {
            compiler: Compiler { host: a, stage: 2 }
        },]);
    }

    #[test]
    fn test_docs() {
        // Behavior of `x.py test` doing various documentation tests.
        let mut config = configure(&["A-A"], &["A-A"]);
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
        };
        // Make sure rustfmt binary not being found isn't an error.
        config.channel = "beta".to_string();
        let build = Build::new(config);
        let mut builder = Builder::new(&build);

        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Test), &[]);
        let a = TargetSelection::from_user("A-A");

        // error_index_generator uses stage 1 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(
            first(builder.cache.all::<doc::ErrorIndex>()),
            &[doc::ErrorIndex { target: a },]
        );
        assert_eq!(first(builder.cache.all::<tool::ErrorIndex>()), &[tool::ErrorIndex {
            compiler: Compiler { host: a, stage: 1 }
        }]);
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
        assert_eq!(first(builder.cache.all::<tool::Rustdoc>()), &[
            tool::Rustdoc { compiler: Compiler { host: a, stage: 0 } },
            tool::Rustdoc { compiler: Compiler { host: a, stage: 1 } },
            tool::Rustdoc { compiler: Compiler { host: a, stage: 2 } },
        ]);
    }
}
