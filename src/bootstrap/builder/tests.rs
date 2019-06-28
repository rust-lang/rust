use super::*;
use crate::config::Config;
use std::thread;

use pretty_assertions::assert_eq;

fn configure(host: &[&str], target: &[&str]) -> Config {
    let mut config = Config::default_opts();
    // don't save toolstates
    config.save_toolstates = None;
    config.skip_only_host_steps = false;
    config.dry_run = true;
    // try to avoid spurious failures in dist where we create/delete each others file
    let dir = config.out.join("tmp-rustbuild-tests").join(
        &thread::current()
            .name()
            .unwrap_or("unknown")
            .replace(":", "-"),
    );
    t!(fs::create_dir_all(&dir));
    config.out = dir;
    config.build = INTERNER.intern_str("A");
    config.hosts = vec![config.build]
        .clone()
        .into_iter()
        .chain(host.iter().map(|s| INTERNER.intern_str(s)))
        .collect::<Vec<_>>();
    config.targets = config
        .hosts
        .clone()
        .into_iter()
        .chain(target.iter().map(|s| INTERNER.intern_str(s)))
        .collect::<Vec<_>>();
    config
}

fn first<A, B>(v: Vec<(A, B)>) -> Vec<A> {
    v.into_iter().map(|(a, _)| a).collect::<Vec<_>>()
}

#[test]
fn dist_baseline() {
    let build = Build::new(configure(&[], &[]));
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

    let a = INTERNER.intern_str("A");

    assert_eq!(
        first(builder.cache.all::<dist::Docs>()),
        &[dist::Docs { host: a },]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Mingw>()),
        &[dist::Mingw { host: a },]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Rustc>()),
        &[dist::Rustc {
            compiler: Compiler { host: a, stage: 2 }
        },]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Std>()),
        &[dist::Std {
            compiler: Compiler { host: a, stage: 1 },
            target: a,
        },]
    );
    assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
}

#[test]
fn dist_with_targets() {
    let build = Build::new(configure(&[], &["B"]));
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

    let a = INTERNER.intern_str("A");
    let b = INTERNER.intern_str("B");

    assert_eq!(
        first(builder.cache.all::<dist::Docs>()),
        &[
            dist::Docs { host: a },
            dist::Docs { host: b },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Mingw>()),
        &[dist::Mingw { host: a }, dist::Mingw { host: b },]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Rustc>()),
        &[dist::Rustc {
            compiler: Compiler { host: a, stage: 2 }
        },]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Std>()),
        &[
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            dist::Std {
                compiler: Compiler { host: a, stage: 2 },
                target: b,
            },
        ]
    );
    assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
}

#[test]
fn dist_with_hosts() {
    let build = Build::new(configure(&["B"], &[]));
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

    let a = INTERNER.intern_str("A");
    let b = INTERNER.intern_str("B");

    assert_eq!(
        first(builder.cache.all::<dist::Docs>()),
        &[
            dist::Docs { host: a },
            dist::Docs { host: b },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Mingw>()),
        &[dist::Mingw { host: a }, dist::Mingw { host: b },]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Rustc>()),
        &[
            dist::Rustc {
                compiler: Compiler { host: a, stage: 2 }
            },
            dist::Rustc {
                compiler: Compiler { host: b, stage: 2 }
            },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Std>()),
        &[
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
        ]
    );
    assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
}

#[test]
fn dist_only_cross_host() {
    let a = INTERNER.intern_str("A");
    let b = INTERNER.intern_str("B");
    let mut build = Build::new(configure(&["B"], &[]));
    build.config.docs = false;
    build.config.extended = true;
    build.hosts = vec![b];
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

    assert_eq!(
        first(builder.cache.all::<dist::Rustc>()),
        &[
            dist::Rustc {
                compiler: Compiler { host: b, stage: 2 }
            },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<compile::Rustc>()),
        &[
            compile::Rustc {
                compiler: Compiler { host: a, stage: 0 },
                target: a,
            },
            compile::Rustc {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
        ]
    );
}

#[test]
fn dist_with_targets_and_hosts() {
    let build = Build::new(configure(&["B"], &["C"]));
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

    let a = INTERNER.intern_str("A");
    let b = INTERNER.intern_str("B");
    let c = INTERNER.intern_str("C");

    assert_eq!(
        first(builder.cache.all::<dist::Docs>()),
        &[
            dist::Docs { host: a },
            dist::Docs { host: b },
            dist::Docs { host: c },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Mingw>()),
        &[
            dist::Mingw { host: a },
            dist::Mingw { host: b },
            dist::Mingw { host: c },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Rustc>()),
        &[
            dist::Rustc {
                compiler: Compiler { host: a, stage: 2 }
            },
            dist::Rustc {
                compiler: Compiler { host: b, stage: 2 }
            },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Std>()),
        &[
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
            dist::Std {
                compiler: Compiler { host: a, stage: 2 },
                target: c,
            },
        ]
    );
    assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
}

#[test]
fn dist_with_target_flag() {
    let mut config = configure(&["B"], &["C"]);
    config.skip_only_host_steps = true; // as-if --target=C was passed
    let build = Build::new(config);
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

    let a = INTERNER.intern_str("A");
    let b = INTERNER.intern_str("B");
    let c = INTERNER.intern_str("C");

    assert_eq!(
        first(builder.cache.all::<dist::Docs>()),
        &[
            dist::Docs { host: a },
            dist::Docs { host: b },
            dist::Docs { host: c },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Mingw>()),
        &[
            dist::Mingw { host: a },
            dist::Mingw { host: b },
            dist::Mingw { host: c },
        ]
    );
    assert_eq!(first(builder.cache.all::<dist::Rustc>()), &[]);
    assert_eq!(
        first(builder.cache.all::<dist::Std>()),
        &[
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
            dist::Std {
                compiler: Compiler { host: a, stage: 2 },
                target: c,
            },
        ]
    );
    assert_eq!(first(builder.cache.all::<dist::Src>()), &[]);
}

#[test]
fn dist_with_same_targets_and_hosts() {
    let build = Build::new(configure(&["B"], &["B"]));
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Dist), &[]);

    let a = INTERNER.intern_str("A");
    let b = INTERNER.intern_str("B");

    assert_eq!(
        first(builder.cache.all::<dist::Docs>()),
        &[
            dist::Docs { host: a },
            dist::Docs { host: b },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Mingw>()),
        &[dist::Mingw { host: a }, dist::Mingw { host: b },]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Rustc>()),
        &[
            dist::Rustc {
                compiler: Compiler { host: a, stage: 2 }
            },
            dist::Rustc {
                compiler: Compiler { host: b, stage: 2 }
            },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<dist::Std>()),
        &[
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            dist::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
        ]
    );
    assert_eq!(first(builder.cache.all::<dist::Src>()), &[dist::Src]);
    assert_eq!(
        first(builder.cache.all::<compile::Std>()),
        &[
            compile::Std {
                compiler: Compiler { host: a, stage: 0 },
                target: a,
            },
            compile::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            compile::Std {
                compiler: Compiler { host: a, stage: 2 },
                target: a,
            },
            compile::Std {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<compile::Test>()),
        &[
            compile::Test {
                compiler: Compiler { host: a, stage: 0 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 2 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<compile::Assemble>()),
        &[
            compile::Assemble {
                target_compiler: Compiler { host: a, stage: 0 },
            },
            compile::Assemble {
                target_compiler: Compiler { host: a, stage: 1 },
            },
            compile::Assemble {
                target_compiler: Compiler { host: a, stage: 2 },
            },
            compile::Assemble {
                target_compiler: Compiler { host: b, stage: 2 },
            },
        ]
    );
}

#[test]
fn build_default() {
    let build = Build::new(configure(&["B"], &["C"]));
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[]);

    let a = INTERNER.intern_str("A");
    let b = INTERNER.intern_str("B");
    let c = INTERNER.intern_str("C");

    assert!(!builder.cache.all::<compile::Std>().is_empty());
    assert!(!builder.cache.all::<compile::Assemble>().is_empty());
    assert_eq!(
        first(builder.cache.all::<compile::Rustc>()),
        &[
            compile::Rustc {
                compiler: Compiler { host: a, stage: 0 },
                target: a,
            },
            compile::Rustc {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            compile::Rustc {
                compiler: Compiler { host: a, stage: 2 },
                target: a,
            },
            compile::Rustc {
                compiler: Compiler { host: b, stage: 2 },
                target: a,
            },
            compile::Rustc {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
            compile::Rustc {
                compiler: Compiler { host: a, stage: 2 },
                target: b,
            },
            compile::Rustc {
                compiler: Compiler { host: b, stage: 2 },
                target: b,
            },
        ]
    );

    assert_eq!(
        first(builder.cache.all::<compile::Test>()),
        &[
            compile::Test {
                compiler: Compiler { host: a, stage: 0 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 2 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: b, stage: 2 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 2 },
                target: b,
            },
            compile::Test {
                compiler: Compiler { host: b, stage: 2 },
                target: b,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 2 },
                target: c,
            },
            compile::Test {
                compiler: Compiler { host: b, stage: 2 },
                target: c,
            },
        ]
    );
}

#[test]
fn build_with_target_flag() {
    let mut config = configure(&["B"], &["C"]);
    config.skip_only_host_steps = true;
    let build = Build::new(config);
    let mut builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Build), &[]);

    let a = INTERNER.intern_str("A");
    let b = INTERNER.intern_str("B");
    let c = INTERNER.intern_str("C");

    assert!(!builder.cache.all::<compile::Std>().is_empty());
    assert_eq!(
        first(builder.cache.all::<compile::Assemble>()),
        &[
            compile::Assemble {
                target_compiler: Compiler { host: a, stage: 0 },
            },
            compile::Assemble {
                target_compiler: Compiler { host: a, stage: 1 },
            },
            compile::Assemble {
                target_compiler: Compiler { host: a, stage: 2 },
            },
            compile::Assemble {
                target_compiler: Compiler { host: b, stage: 2 },
            },
        ]
    );
    assert_eq!(
        first(builder.cache.all::<compile::Rustc>()),
        &[
            compile::Rustc {
                compiler: Compiler { host: a, stage: 0 },
                target: a,
            },
            compile::Rustc {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            compile::Rustc {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
        ]
    );

    assert_eq!(
        first(builder.cache.all::<compile::Test>()),
        &[
            compile::Test {
                compiler: Compiler { host: a, stage: 0 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 1 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 2 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: b, stage: 2 },
                target: a,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 1 },
                target: b,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 2 },
                target: b,
            },
            compile::Test {
                compiler: Compiler { host: b, stage: 2 },
                target: b,
            },
            compile::Test {
                compiler: Compiler { host: a, stage: 2 },
                target: c,
            },
            compile::Test {
                compiler: Compiler { host: b, stage: 2 },
                target: c,
            },
        ]
    );
}

#[test]
fn test_with_no_doc_stage0() {
    let mut config = configure(&[], &[]);
    config.stage = Some(0);
    config.cmd = Subcommand::Test {
        paths: vec!["src/libstd".into()],
        test_args: vec![],
        rustc_args: vec![],
        fail_fast: true,
        doc_tests: DocTests::No,
        bless: false,
        compare_mode: None,
        rustfix_coverage: false,
        pass: None,
    };

    let build = Build::new(config);
    let mut builder = Builder::new(&build);

    let host = INTERNER.intern_str("A");

    builder.run_step_descriptions(
        &[StepDescription::from::<test::Crate>()],
        &["src/libstd".into()],
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
    let mut config = configure(&[], &[]);
    config.exclude = vec![
        "src/test/run-pass".into(),
        "src/tools/tidy".into(),
    ];
    config.cmd = Subcommand::Test {
        paths: Vec::new(),
        test_args: Vec::new(),
        rustc_args: Vec::new(),
        fail_fast: true,
        doc_tests: DocTests::No,
        bless: false,
        compare_mode: None,
        rustfix_coverage: false,
        pass: None,
    };

    let build = Build::new(config);
    let builder = Builder::new(&build);
    builder.run_step_descriptions(&Builder::get_step_descriptions(Kind::Test), &[]);

    // Ensure we have really excluded run-pass & tidy
    assert!(!builder.cache.contains::<test::RunPass>());
    assert!(!builder.cache.contains::<test::Tidy>());

    // Ensure other tests are not affected.
    assert!(builder.cache.contains::<test::RunPassFullDeps>());
    assert!(builder.cache.contains::<test::RustdocUi>());
}
