//! Implementation of compiling the compiler and standard library, in "check"-based modes.

use crate::builder::{Builder, Kind, RunConfig, ShouldRun, Step};
use crate::cache::Interned;
use crate::compile::{add_to_sysroot, run_cargo, rustc_cargo, rustc_cargo_env, std_cargo};
use crate::config::TargetSelection;
use crate::tool::{prepare_tool_cargo, SourceType};
use crate::INTERNER;
use crate::{Compiler, Mode, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: TargetSelection,
}

/// Returns args for the subcommand itself (not for cargo)
fn args(builder: &Builder<'_>) -> Vec<String> {
    fn strings<'a>(arr: &'a [&str]) -> impl Iterator<Item = String> + 'a {
        arr.iter().copied().map(String::from)
    }

    if let Subcommand::Clippy { fix, allow, deny, warn, forbid, .. } = &builder.config.cmd {
        // disable the most spammy clippy lints
        let ignored_lints = vec![
            "many_single_char_names", // there are a lot in stdarch
            "collapsible_if",
            "type_complexity",
            "missing_safety_doc", // almost 3K warnings
            "too_many_arguments",
            "needless_lifetimes", // people want to keep the lifetimes
            "wrong_self_convention",
        ];
        let mut args = vec![];
        if *fix {
            #[rustfmt::skip]
            args.extend(strings(&[
                "--fix", "-Zunstable-options",
                // FIXME: currently, `--fix` gives an error while checking tests for libtest,
                // possibly because libtest is not yet built in the sysroot.
                // As a workaround, avoid checking tests and benches when passed --fix.
                "--lib", "--bins", "--examples",
            ]));
        }
        args.extend(strings(&["--", "--cap-lints", "warn"]));
        args.extend(ignored_lints.iter().map(|lint| format!("-Aclippy::{}", lint)));
        let mut clippy_lint_levels: Vec<String> = Vec::new();
        allow.iter().for_each(|v| clippy_lint_levels.push(format!("-A{}", v)));
        deny.iter().for_each(|v| clippy_lint_levels.push(format!("-D{}", v)));
        warn.iter().for_each(|v| clippy_lint_levels.push(format!("-W{}", v)));
        forbid.iter().for_each(|v| clippy_lint_levels.push(format!("-F{}", v)));
        args.extend(clippy_lint_levels);
        args.extend(builder.config.free_args.clone());
        args
    } else {
        builder.config.free_args.clone()
    }
}

fn cargo_subcommand(kind: Kind) -> &'static str {
    match kind {
        Kind::Check => "check",
        Kind::Clippy => "clippy",
        Kind::Fix => "fix",
        _ => unreachable!(),
    }
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("sysroot").path("library")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Std { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.update_submodule(&Path::new("library").join("stdarch"));

        let target = self.target;
        let compiler = builder.compiler(builder.top_stage, builder.config.build);

        let mut cargo = builder.cargo(
            compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            cargo_subcommand(builder.kind),
        );
        std_cargo(builder, target, compiler.stage, &mut cargo);
        if matches!(builder.config.cmd, Subcommand::Fix { .. }) {
            // By default, cargo tries to fix all targets. Tell it not to fix tests until we've added `test` to the sysroot.
            cargo.arg("--lib");
        }

        let _guard = builder.msg_check("library artifacts", target);
        run_cargo(
            builder,
            cargo,
            args(builder),
            &libstd_stamp(builder, compiler, target),
            vec![],
            true,
            false,
        );

        // We skip populating the sysroot in non-zero stage because that'll lead
        // to rlib/rmeta conflicts if std gets built during this session.
        if compiler.stage == 0 {
            let libdir = builder.sysroot_libdir(compiler, target);
            let hostdir = builder.sysroot_libdir(compiler, compiler.host);
            add_to_sysroot(&builder, &libdir, &hostdir, &libstd_stamp(builder, compiler, target));
        }

        // don't run on std twice with x.py clippy
        if builder.kind == Kind::Clippy {
            return;
        }

        // Then run cargo again, once we've put the rmeta files for the library
        // crates into the sysroot. This is needed because e.g., core's tests
        // depend on `libtest` -- Cargo presumes it will exist, but it doesn't
        // since we initialize with an empty sysroot.
        //
        // Currently only the "libtest" tree of crates does this.
        let mut cargo = builder.cargo(
            compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            cargo_subcommand(builder.kind),
        );

        // If we're not in stage 0, tests and examples will fail to compile
        // from `core` definitions being loaded from two different `libcore`
        // .rmeta and .rlib files.
        if compiler.stage == 0 {
            cargo.arg("--all-targets");
        }

        std_cargo(builder, target, compiler.stage, &mut cargo);

        // Explicitly pass -p for all dependencies krates -- this will force cargo
        // to also check the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in builder.in_tree_crates("test", Some(target)) {
            cargo.arg("-p").arg(krate.name);
        }

        let _guard = builder.msg_check("library test/bench/example targets", target);
        run_cargo(
            builder,
            cargo,
            args(builder),
            &libstd_test_stamp(builder, compiler, target),
            vec![],
            true,
            false,
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub target: TargetSelection,
}

impl Step for Rustc {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("rustc-main").path("compiler")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustc { target: run.target });
    }

    /// Builds the compiler.
    ///
    /// This will build the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(builder.top_stage, builder.config.build);
        let target = self.target;

        if compiler.stage != 0 {
            // If we're not in stage 0, then we won't have a std from the beta
            // compiler around. That means we need to make sure there's one in
            // the sysroot for the compiler to find. Otherwise, we're going to
            // fail when building crates that need to generate code (e.g., build
            // scripts and their dependencies).
            builder.ensure(crate::compile::Std::new(compiler, compiler.host));
            builder.ensure(crate::compile::Std::new(compiler, target));
        } else {
            builder.ensure(Std { target });
        }

        let mut cargo = builder.cargo(
            compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            cargo_subcommand(builder.kind),
        );
        rustc_cargo(builder, &mut cargo, target, compiler.stage);

        // For ./x.py clippy, don't run with --all-targets because
        // linting tests and benchmarks can produce very noisy results
        if builder.kind != Kind::Clippy {
            cargo.arg("--all-targets");
        }

        // Explicitly pass -p for all compiler krates -- this will force cargo
        // to also check the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in builder.in_tree_crates("rustc-main", Some(target)) {
            cargo.arg("-p").arg(krate.name);
        }

        let _guard = builder.msg_check("compiler artifacts", target);
        run_cargo(
            builder,
            cargo,
            args(builder),
            &librustc_stamp(builder, compiler, target),
            vec![],
            true,
            false,
        );

        let libdir = builder.sysroot_libdir(compiler, target);
        let hostdir = builder.sysroot_libdir(compiler, compiler.host);
        add_to_sysroot(&builder, &libdir, &hostdir, &librustc_stamp(builder, compiler, target));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CodegenBackend {
    pub target: TargetSelection,
    pub backend: Interned<String>,
}

impl Step for CodegenBackend {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["compiler/rustc_codegen_cranelift", "compiler/rustc_codegen_gcc"])
    }

    fn make_run(run: RunConfig<'_>) {
        for &backend in &[INTERNER.intern_str("cranelift"), INTERNER.intern_str("gcc")] {
            run.builder.ensure(CodegenBackend { target: run.target, backend });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(builder.top_stage, builder.config.build);
        let target = self.target;
        let backend = self.backend;

        builder.ensure(Rustc { target });

        let mut cargo = builder.cargo(
            compiler,
            Mode::Codegen,
            SourceType::InTree,
            target,
            cargo_subcommand(builder.kind),
        );
        cargo
            .arg("--manifest-path")
            .arg(builder.src.join(format!("compiler/rustc_codegen_{}/Cargo.toml", backend)));
        rustc_cargo_env(builder, &mut cargo, target, compiler.stage);

        let _guard = builder.msg_check(&backend, target);

        run_cargo(
            builder,
            cargo,
            args(builder),
            &codegen_backend_stamp(builder, compiler, target, backend),
            vec![],
            true,
            false,
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct RustAnalyzer {
    pub target: TargetSelection,
}

impl Step for RustAnalyzer {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rust-analyzer")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustAnalyzer { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(builder.top_stage, builder.config.build);
        let target = self.target;

        builder.ensure(Std { target });

        let mut cargo = prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolStd,
            target,
            cargo_subcommand(builder.kind),
            "src/tools/rust-analyzer",
            SourceType::InTree,
            &["rust-analyzer/in-rust-tree".to_owned()],
        );

        cargo.allow_features(crate::tool::RustAnalyzer::ALLOW_FEATURES);

        // For ./x.py clippy, don't check those targets because
        // linting tests and benchmarks can produce very noisy results
        if builder.kind != Kind::Clippy {
            // can't use `--all-targets` because `--examples` doesn't work well
            cargo.arg("--bins");
            cargo.arg("--tests");
            cargo.arg("--benches");
        }

        let _guard = builder.msg_check("rust-analyzer artifacts", target);
        run_cargo(
            builder,
            cargo,
            args(builder),
            &stamp(builder, compiler, target),
            vec![],
            true,
            false,
        );

        /// Cargo's output path in a given stage, compiled by a particular
        /// compiler for the specified target.
        fn stamp(builder: &Builder<'_>, compiler: Compiler, target: TargetSelection) -> PathBuf {
            builder.cargo_out(compiler, Mode::ToolStd, target).join(".rust-analyzer-check.stamp")
        }
    }
}

macro_rules! tool_check_step {
    ($name:ident, $path:literal, $($alias:literal, )* $source_type:path $(, $default:literal )?) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            pub target: TargetSelection,
        }

        impl Step for $name {
            type Output = ();
            const ONLY_HOSTS: bool = true;
            // don't ever check out-of-tree tools by default, they'll fail when toolstate is broken
            const DEFAULT: bool = matches!($source_type, SourceType::InTree) $( && $default )?;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.paths(&[ $path, $($alias),* ])
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name { target: run.target });
            }

            fn run(self, builder: &Builder<'_>) {
                let compiler = builder.compiler(builder.top_stage, builder.config.build);
                let target = self.target;

                builder.ensure(Rustc { target });

                let mut cargo = prepare_tool_cargo(
                    builder,
                    compiler,
                    Mode::ToolRustc,
                    target,
                    cargo_subcommand(builder.kind),
                    $path,
                    $source_type,
                    &[],
                );

                // For ./x.py clippy, don't run with --all-targets because
                // linting tests and benchmarks can produce very noisy results
                if builder.kind != Kind::Clippy {
                    cargo.arg("--all-targets");
                }

                // Enable internal lints for clippy and rustdoc
                // NOTE: this doesn't enable lints for any other tools unless they explicitly add `#![warn(rustc::internal)]`
                // See https://github.com/rust-lang/rust/pull/80573#issuecomment-754010776
                cargo.rustflag("-Zunstable-options");
                let _guard = builder.msg_check(&concat!(stringify!($name), " artifacts").to_lowercase(), target);
                run_cargo(
                    builder,
                    cargo,
                    args(builder),
                    &stamp(builder, compiler, target),
                    vec![],
                    true,
                    false,
                );

                /// Cargo's output path in a given stage, compiled by a particular
                /// compiler for the specified target.
                fn stamp(
                    builder: &Builder<'_>,
                    compiler: Compiler,
                    target: TargetSelection,
                ) -> PathBuf {
                    builder
                        .cargo_out(compiler, Mode::ToolRustc, target)
                        .join(format!(".{}-check.stamp", stringify!($name).to_lowercase()))
                }
            }
        }
    };
}

tool_check_step!(Rustdoc, "src/tools/rustdoc", "src/librustdoc", SourceType::InTree);
// Clippy, miri and Rustfmt are hybrids. They are external tools, but use a git subtree instead
// of a submodule. Since the SourceType only drives the deny-warnings
// behavior, treat it as in-tree so that any new warnings in clippy will be
// rejected.
tool_check_step!(Clippy, "src/tools/clippy", SourceType::InTree);
tool_check_step!(Miri, "src/tools/miri", SourceType::InTree);
tool_check_step!(CargoMiri, "src/tools/miri/cargo-miri", SourceType::InTree);
tool_check_step!(Rls, "src/tools/rls", SourceType::InTree);
tool_check_step!(Rustfmt, "src/tools/rustfmt", SourceType::InTree);
tool_check_step!(MiroptTestTools, "src/tools/miropt-test-tools", SourceType::InTree);

tool_check_step!(Bootstrap, "src/bootstrap", SourceType::InTree, false);

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
fn libstd_stamp(builder: &Builder<'_>, compiler: Compiler, target: TargetSelection) -> PathBuf {
    builder.cargo_out(compiler, Mode::Std, target).join(".libstd-check.stamp")
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
fn libstd_test_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
) -> PathBuf {
    builder.cargo_out(compiler, Mode::Std, target).join(".libstd-check-test.stamp")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// compiler for the specified target.
fn librustc_stamp(builder: &Builder<'_>, compiler: Compiler, target: TargetSelection) -> PathBuf {
    builder.cargo_out(compiler, Mode::Rustc, target).join(".librustc-check.stamp")
}

/// Cargo's output path for librustc_codegen_llvm in a given stage, compiled by a particular
/// compiler for the specified target and backend.
fn codegen_backend_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
    backend: Interned<String>,
) -> PathBuf {
    builder
        .cargo_out(compiler, Mode::Codegen, target)
        .join(format!(".librustc_codegen_{}-check.stamp", backend))
}
