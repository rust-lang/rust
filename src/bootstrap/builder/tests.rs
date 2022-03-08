use super::*;
use crate::config::{Config, TargetSelection};
use std::thread;

fn configure(cmd: &str, host: &[&str], target: &[&str]) -> Config {
    let mut config = Config::parse(&[cmd.to_owned()]);
    // don't save toolstates
    config.save_toolstates = None;
    config.dry_run = true;
    config.ninja_in_file = false;
    // try to avoid spurious failures in dist where we create/delete each others file
    config.out = PathBuf::from(env::var_os("BOOTSTRAP_OUTPUT_DIRECTORY").unwrap());
    config.initial_rustc = PathBuf::from(env::var_os("RUSTC").unwrap());
    config.initial_cargo = PathBuf::from(env::var_os("BOOTSTRAP_INITIAL_CARGO").unwrap());
    let dir = config
        .out
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

mod defaults {
    use super::{configure, first};
    use crate::builder::*;
    use crate::Config;
    use pretty_assertions::assert_eq;

    #[test]
    fn build_default() {
        let build = Build::new(configure("build", &["A"], &["A"]));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[]);

        let a = TargetSelection::from_user("A");
        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                compile::Std { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
            ]
        );
        assert!(!builder.cache.all::<compile::Assemble>().is_empty());
        // Make sure rustdoc is only built once.
        assert_eq!(
            first(builder.cache.all::<tool::Rustdoc>()),
            // Recall that rustdoc stages are off-by-one
            // - this is the compiler it's _linked_ to, not built with.
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 1 } }],
        );
        assert_eq!(
            first(builder.cache.all::<compile::Rustc>()),
            &[compile::Rustc { compiler: Compiler { host: a, stage: 0 }, target: a },]
        );
    }

    #[test]
    fn build_stage_0() {
        let config = Config { stage: 0, ..configure("build", &["A"], &["A"]) };
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[]);

        let a = TargetSelection::from_user("A");
        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[compile::Std { compiler: Compiler { host: a, stage: 0 }, target: a },]
        );
        assert!(!builder.cache.all::<compile::Assemble>().is_empty());
        assert_eq!(
            first(builder.cache.all::<tool::Rustdoc>()),
            // This is the beta rustdoc.
            // Add an assert here to make sure this is the only rustdoc built.
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 0 } }],
        );
        assert!(builder.cache.all::<compile::Rustc>().is_empty());
    }

    #[test]
    fn build_cross_compile() {
        let config = Config { stage: 1, ..configure("build", &["A", "B"], &["A", "B"]) };
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[]);

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");

        // Ideally, this build wouldn't actually have `target: a`
        // rustdoc/rustcc/std here (the user only requested a host=B build, so
        // there's not really a need for us to build for target A in this case
        // (since we're producing stage 1 libraries/binaries).  But currently
        // rustbuild is just a bit buggy here; this should be fixed though.
        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                compile::Std { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 0 }, target: b },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<compile::Assemble>()),
            &[
                compile::Assemble { target_compiler: Compiler { host: a, stage: 0 } },
                compile::Assemble { target_compiler: Compiler { host: a, stage: 1 } },
                compile::Assemble { target_compiler: Compiler { host: b, stage: 1 } },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<tool::Rustdoc>()),
            &[
                tool::Rustdoc { compiler: Compiler { host: a, stage: 1 } },
                tool::Rustdoc { compiler: Compiler { host: b, stage: 1 } },
            ],
        );
        assert_eq!(
            first(builder.cache.all::<compile::Rustc>()),
            &[
                compile::Rustc { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Rustc { compiler: Compiler { host: a, stage: 0 }, target: b },
            ]
        );
    }

    #[test]
    fn doc_default() {
        let mut config = configure("doc", &["A"], &["A"]);
        config.compiler_docs = true;
        config.cmd = Subcommand::Doc { paths: Vec::new(), open: false };
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Doc), &[]);
        let a = TargetSelection::from_user("A");

        // error_index_generator uses stage 0 to share rustdoc artifacts with the
        // rustdoc tool.
        assert_eq!(
            first(builder.cache.all::<doc::ErrorIndex>()),
            &[doc::ErrorIndex { target: a },]
        );
        assert_eq!(
            first(builder.cache.all::<tool::ErrorIndex>()),
            &[tool::ErrorIndex { compiler: Compiler { host: a, stage: 0 } }]
        );
        // docs should be built with the beta compiler, not with the stage0 artifacts.
        // recall that rustdoc is off-by-one: `stage` is the compiler rustdoc is _linked_ to,
        // not the one it was built by.
        assert_eq!(
            first(builder.cache.all::<tool::Rustdoc>()),
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 0 } },]
        );
    }
}

mod dist {
    use super::{first, Config};
    use crate::builder::*;
    use pretty_assertions::assert_eq;

    fn configure(host: &[&str], target: &[&str]) -> Config {
        Config { stage: 2, ..super::configure("dist", host, target) }
    }

    #[test]
    fn dist_baseline() {
        let build = Build::new(configure(&["A"], &["A"]));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

        let a = TargetSelection::from_user("A");

        assert_eq!(first(builder.cache.all::<dist::Docs>()), &[dist::Docs { host: a },]);
        assert_eq!(first(builder.cache.all::<dist::Mingw>()), &[dist::Mingw { host: a },]);
        assert_eq!(
            first(builder.cache.all::<dist::Rustc>()),
            &[dist::Rustc { compiler: Compiler { host: a, stage: 2 } },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Std>()),
            &[dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },]
        );
        assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
        // Make sure rustdoc is only built once.
        assert_eq!(
            first(builder.cache.all::<tool::Rustdoc>()),
            &[tool::Rustdoc { compiler: Compiler { host: a, stage: 2 } },]
        );
    }

    #[test]
    fn dist_with_targets() {
        let build = Build::new(configure(&["A"], &["A", "B"]));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");

        assert_eq!(
            first(builder.cache.all::<dist::Docs>()),
            &[dist::Docs { host: a }, dist::Docs { host: b },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Mingw>()),
            &[dist::Mingw { host: a }, dist::Mingw { host: b },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Rustc>()),
            &[dist::Rustc { compiler: Compiler { host: a, stage: 2 } },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                dist::Std { compiler: Compiler { host: a, stage: 2 }, target: b },
            ]
        );
        assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_with_hosts() {
        let build = Build::new(configure(&["A", "B"], &["A", "B"]));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");

        assert_eq!(
            first(builder.cache.all::<dist::Docs>()),
            &[dist::Docs { host: a }, dist::Docs { host: b },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Mingw>()),
            &[dist::Mingw { host: a }, dist::Mingw { host: b },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Rustc>()),
            &[
                dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
                dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                compile::Std { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 2 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
                compile::Std { compiler: Compiler { host: a, stage: 2 }, target: b },
            ],
        );
        assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_only_cross_host() {
        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");
        let mut build = Build::new(configure(&["A", "B"], &["A", "B"]));
        build.config.docs = false;
        build.config.extended = true;
        build.hosts = vec![b];
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

        assert_eq!(
            first(builder.cache.all::<dist::Rustc>()),
            &[dist::Rustc { compiler: Compiler { host: b, stage: 2 } },]
        );
        assert_eq!(
            first(builder.cache.all::<compile::Rustc>()),
            &[
                compile::Rustc { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Rustc { compiler: Compiler { host: a, stage: 1 }, target: b },
            ]
        );
    }

    #[test]
    fn dist_with_targets_and_hosts() {
        let build = Build::new(configure(&["A", "B"], &["A", "B", "C"]));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");
        let c = TargetSelection::from_user("C");

        assert_eq!(
            first(builder.cache.all::<dist::Docs>()),
            &[dist::Docs { host: a }, dist::Docs { host: b }, dist::Docs { host: c },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Mingw>()),
            &[dist::Mingw { host: a }, dist::Mingw { host: b }, dist::Mingw { host: c },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Rustc>()),
            &[
                dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
                dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
                dist::Std { compiler: Compiler { host: a, stage: 2 }, target: c },
            ]
        );
        assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
    }

    #[test]
    fn dist_with_empty_host() {
        let config = configure(&[], &["C"]);
        let build = Build::new(config);
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

        let a = TargetSelection::from_user("A");
        let c = TargetSelection::from_user("C");

        assert_eq!(first(builder.cache.all::<dist::Docs>()), &[dist::Docs { host: c },]);
        assert_eq!(first(builder.cache.all::<dist::Mingw>()), &[dist::Mingw { host: c },]);
        assert_eq!(
            first(builder.cache.all::<dist::Std>()),
            &[dist::Std { compiler: Compiler { host: a, stage: 2 }, target: c },]
        );
    }

    #[test]
    fn dist_with_same_targets_and_hosts() {
        let build = Build::new(configure(&["A", "B"], &["A", "B"]));
        let mut builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");

        assert_eq!(
            first(builder.cache.all::<dist::Docs>()),
            &[dist::Docs { host: a }, dist::Docs { host: b },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Mingw>()),
            &[dist::Mingw { host: a }, dist::Mingw { host: b },]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Rustc>()),
            &[
                dist::Rustc { compiler: Compiler { host: a, stage: 2 } },
                dist::Rustc { compiler: Compiler { host: b, stage: 2 } },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<dist::Std>()),
            &[
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                dist::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
            ]
        );
        assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                compile::Std { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 2 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
                compile::Std { compiler: Compiler { host: a, stage: 2 }, target: b },
            ]
        );
        assert_eq!(
            first(builder.cache.all::<compile::Assemble>()),
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
            &["compiler/rustc".into(), "library/std".into()],
        );

        let a = TargetSelection::from_user("A");
        let b = TargetSelection::from_user("B");
        let c = TargetSelection::from_user("C");

        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                compile::Std { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 2 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: b },
                compile::Std { compiler: Compiler { host: a, stage: 2 }, target: b },
                compile::Std { compiler: Compiler { host: a, stage: 2 }, target: c },
            ]
        );
        assert!(!builder.cache.all::<compile::Assemble>().is_empty());
        assert_eq!(
            first(builder.cache.all::<compile::Rustc>()),
            &[
                compile::Rustc { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Rustc { compiler: Compiler { host: a, stage: 1 }, target: a },
                compile::Rustc { compiler: Compiler { host: a, stage: 2 }, target: a },
                compile::Rustc { compiler: Compiler { host: a, stage: 1 }, target: b },
                compile::Rustc { compiler: Compiler { host: a, stage: 2 }, target: b },
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
        let c = TargetSelection::from_user("C");

        assert_eq!(
            first(builder.cache.all::<compile::Std>()),
            &[
                compile::Std { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 1 }, target: a },
                compile::Std { compiler: Compiler { host: a, stage: 2 }, target: c },
            ]
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
            &[
                compile::Rustc { compiler: Compiler { host: a, stage: 0 }, target: a },
                compile::Rustc { compiler: Compiler { host: a, stage: 1 }, target: a },
            ]
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
                krate: INTERNER.intern_str("std"),
            },]
        );
    }

    #[test]
    fn test_exclude() {
        let mut config = configure(&["A"], &["A"]);
        config.exclude = vec![TaskPath::parse("src/tools/tidy")];
        config.cmd = Subcommand::Test {
            paths: Vec::new(),
            test_args: Vec::new(),
            rustc_args: Vec::new(),
            fail_fast: true,
            doc_tests: DocTests::No,
            bless: false,
            force_rerun: false,
            compare_mode: None,
            rustfix_coverage: false,
            pass: None,
            run: None,
        };

        let build = Build::new(config);
        let builder = Builder::new(&build);
        builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Test), &[]);

        // Ensure we have really excluded tidy
        assert!(!builder.cache.contains::<test::Tidy>());

        // Ensure other tests are not affected.
        assert!(builder.cache.contains::<test::RustdocUi>());
    }

    #[test]
    fn doc_ci() {
        let mut config = configure(&["A"], &["A"]);
        config.compiler_docs = true;
        config.cmd = Subcommand::Doc { paths: Vec::new(), open: false };
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
