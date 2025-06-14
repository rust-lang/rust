//! Implementation of compiling the compiler and standard library, in "check"-based modes.

use crate::core::build_steps::compile;
use crate::core::build_steps::compile::{
    add_to_sysroot, run_cargo, rustc_cargo, rustc_cargo_env, std_cargo, std_crates_for_run_make,
};
use crate::core::build_steps::tool::{COMPILETEST_ALLOW_FEATURES, SourceType, prepare_tool_cargo};
use crate::core::builder::{
    self, Alias, Builder, Kind, RunConfig, ShouldRun, Step, crate_description,
};
use crate::core::config::TargetSelection;
use crate::utils::build_stamp::{self, BuildStamp};
use crate::{Mode, Subcommand};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: TargetSelection,
    /// Whether to build only a subset of crates.
    ///
    /// This shouldn't be used from other steps; see the comment on [`compile::Rustc`].
    ///
    /// [`compile::Rustc`]: crate::core::build_steps::compile::Rustc
    crates: Vec<String>,
    /// Override `Builder::kind` on cargo invocations.
    ///
    /// By default, `Builder::kind` is propagated as the subcommand to the cargo invocations.
    /// However, there are cases when this is not desirable. For example, when running `x clippy $tool_name`,
    /// passing `Builder::kind` to cargo invocations would run clippy on the entire compiler and library,
    /// which is not useful if we only want to lint a few crates with specific rules.
    override_build_kind: Option<Kind>,
    /// Never use this from outside calls. It is intended for internal use only within `check::Std::make_run`
    /// and `check::Std::run`.
    custom_stage: Option<u32>,
}

impl Std {
    const CRATE_OR_DEPS: &[&str] = &["sysroot", "coretests", "alloctests"];

    pub fn new(target: TargetSelection) -> Self {
        Self { target, crates: vec![], override_build_kind: None, custom_stage: None }
    }

    pub fn build_kind(mut self, kind: Option<Kind>) -> Self {
        self.override_build_kind = kind;
        self
    }
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let mut run = run;
        for c in Std::CRATE_OR_DEPS {
            run = run.crate_or_deps(c);
        }

        run.path("library")
    }

    fn make_run(run: RunConfig<'_>) {
        let crates = std_crates_for_run_make(&run);

        let stage = if run.builder.config.is_explicit_stage() || run.builder.top_stage >= 1 {
            run.builder.top_stage
        } else {
            1
        };

        run.builder.ensure(Std {
            target: run.target,
            crates,
            override_build_kind: None,
            custom_stage: Some(stage),
        });
    }

    fn run(self, builder: &Builder<'_>) {
        if !builder.download_rustc() && builder.config.skip_std_check_if_no_download_rustc {
            eprintln!(
                "WARNING: `--skip-std-check-if-no-download-rustc` flag was passed and `rust.download-rustc` is not available. Skipping."
            );
            return;
        }

        builder.require_submodule("library/stdarch", None);

        let stage = self.custom_stage.unwrap_or(builder.top_stage);

        let target = self.target;
        let compiler = builder.compiler(stage, builder.config.host_target);

        if stage == 0 {
            let mut is_explicitly_called =
                builder.paths.iter().any(|p| p.starts_with("library") || p.starts_with("std"));

            if !is_explicitly_called {
                for c in Std::CRATE_OR_DEPS {
                    is_explicitly_called = builder.paths.iter().any(|p| p.starts_with(c));
                }
            }

            if is_explicitly_called {
                eprintln!("WARNING: stage 0 std is precompiled and does nothing during `x check`.");
            }

            // Reuse the stage0 libstd
            builder.ensure(compile::Std::new(compiler, target));
            return;
        }

        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            self.override_build_kind.unwrap_or(builder.kind),
        );

        std_cargo(builder, target, compiler.stage, &mut cargo);
        if matches!(builder.config.cmd, Subcommand::Fix) {
            // By default, cargo tries to fix all targets. Tell it not to fix tests until we've added `test` to the sysroot.
            cargo.arg("--lib");
        }

        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard = builder.msg_check(
            format_args!("library artifacts{}", crate_description(&self.crates)),
            target,
            Some(stage),
        );

        let stamp = build_stamp::libstd_stamp(builder, compiler, target).with_prefix("check");
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);

        // We skip populating the sysroot in non-zero stage because that'll lead
        // to rlib/rmeta conflicts if std gets built during this session.
        if compiler.stage == 0 {
            let libdir = builder.sysroot_target_libdir(compiler, target);
            let hostdir = builder.sysroot_target_libdir(compiler, compiler.host);
            add_to_sysroot(builder, &libdir, &hostdir, &stamp);
        }
        drop(_guard);

        // don't run on std twice with x.py clippy
        // don't check test dependencies if we haven't built libtest
        if builder.kind == Kind::Clippy || !self.crates.iter().any(|krate| krate == "test") {
            return;
        }

        // Then run cargo again, once we've put the rmeta files for the library
        // crates into the sysroot. This is needed because e.g., core's tests
        // depend on `libtest` -- Cargo presumes it will exist, but it doesn't
        // since we initialize with an empty sysroot.
        //
        // Currently only the "libtest" tree of crates does this.
        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            self.override_build_kind.unwrap_or(builder.kind),
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
        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let stamp = build_stamp::libstd_stamp(builder, compiler, target).with_prefix("check-test");
        let _guard = builder.msg_check("library test/bench/example targets", target, Some(stage));
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub target: TargetSelection,
    /// Whether to build only a subset of crates.
    ///
    /// This shouldn't be used from other steps; see the comment on [`compile::Rustc`].
    ///
    /// [`compile::Rustc`]: crate::core::build_steps::compile::Rustc
    crates: Vec<String>,
    /// Override `Builder::kind` on cargo invocations.
    ///
    /// By default, `Builder::kind` is propagated as the subcommand to the cargo invocations.
    /// However, there are cases when this is not desirable. For example, when running `x clippy $tool_name`,
    /// passing `Builder::kind` to cargo invocations would run clippy on the entire compiler and library,
    /// which is not useful if we only want to lint a few crates with specific rules.
    override_build_kind: Option<Kind>,
}

impl Rustc {
    pub fn new(target: TargetSelection, builder: &Builder<'_>) -> Self {
        let crates = builder
            .in_tree_crates("rustc-main", Some(target))
            .into_iter()
            .map(|krate| krate.name.to_string())
            .collect();
        Self { target, crates, override_build_kind: None }
    }

    pub fn build_kind(mut self, build_kind: Option<Kind>) -> Self {
        self.override_build_kind = build_kind;
        self
    }
}

impl Step for Rustc {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("rustc-main").path("compiler")
    }

    fn make_run(run: RunConfig<'_>) {
        let crates = run.make_run_crates(Alias::Compiler);
        run.builder.ensure(Rustc { target: run.target, crates, override_build_kind: None });
    }

    /// Builds the compiler.
    ///
    /// This will build the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(builder.top_stage, builder.config.host_target);
        let target = self.target;

        if compiler.stage != 0 {
            // If we're not in stage 0, then we won't have a std from the beta
            // compiler around. That means we need to make sure there's one in
            // the sysroot for the compiler to find. Otherwise, we're going to
            // fail when building crates that need to generate code (e.g., build
            // scripts and their dependencies).
            builder.ensure(crate::core::build_steps::compile::Std::new(compiler, compiler.host));
            builder.ensure(crate::core::build_steps::compile::Std::new(compiler, target));
        } else {
            builder.ensure(Std::new(target).build_kind(self.override_build_kind));
        }

        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            self.override_build_kind.unwrap_or(builder.kind),
        );

        rustc_cargo(builder, &mut cargo, target, &compiler, &self.crates);

        // For ./x.py clippy, don't run with --all-targets because
        // linting tests and benchmarks can produce very noisy results
        if builder.kind != Kind::Clippy {
            cargo.arg("--all-targets");
        }

        // Explicitly pass -p for all compiler crates -- this will force cargo
        // to also check the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard = builder.msg_check(
            format_args!("compiler artifacts{}", crate_description(&self.crates)),
            target,
            None,
        );

        let stamp = build_stamp::librustc_stamp(builder, compiler, target).with_prefix("check");

        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);

        let libdir = builder.sysroot_target_libdir(compiler, target);
        let hostdir = builder.sysroot_target_libdir(compiler, compiler.host);
        add_to_sysroot(builder, &libdir, &hostdir, &stamp);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodegenBackend {
    pub target: TargetSelection,
    pub backend: &'static str,
}

impl Step for CodegenBackend {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["compiler/rustc_codegen_cranelift", "compiler/rustc_codegen_gcc"])
    }

    fn make_run(run: RunConfig<'_>) {
        for &backend in &["cranelift", "gcc"] {
            run.builder.ensure(CodegenBackend { target: run.target, backend });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        // FIXME: remove once https://github.com/rust-lang/rust/issues/112393 is resolved
        if builder.build.config.vendor && self.backend == "gcc" {
            println!("Skipping checking of `rustc_codegen_gcc` with vendoring enabled.");
            return;
        }

        let compiler = builder.compiler(builder.top_stage, builder.config.host_target);
        let target = self.target;
        let backend = self.backend;

        builder.ensure(Rustc::new(target, builder));

        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Codegen,
            SourceType::InTree,
            target,
            builder.kind,
        );

        cargo
            .arg("--manifest-path")
            .arg(builder.src.join(format!("compiler/rustc_codegen_{backend}/Cargo.toml")));
        rustc_cargo_env(builder, &mut cargo, target, compiler.stage);

        let _guard = builder.msg_check(backend, target, None);

        let stamp = build_stamp::codegen_backend_stamp(builder, compiler, target, backend)
            .with_prefix("check");

        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RustAnalyzer {
    pub target: TargetSelection,
}

impl Step for RustAnalyzer {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/rust-analyzer").default_condition(
            builder
                .config
                .tools
                .as_ref()
                .is_none_or(|tools| tools.iter().any(|tool| tool == "rust-analyzer")),
        )
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustAnalyzer { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = builder.compiler(builder.top_stage, builder.config.host_target);
        let target = self.target;

        builder.ensure(Rustc::new(target, builder));

        let mut cargo = prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            target,
            builder.kind,
            "src/tools/rust-analyzer",
            SourceType::InTree,
            &["in-rust-tree".to_owned()],
        );

        cargo.allow_features(crate::core::build_steps::tool::RustAnalyzer::ALLOW_FEATURES);

        // For ./x.py clippy, don't check those targets because
        // linting tests and benchmarks can produce very noisy results
        if builder.kind != Kind::Clippy {
            // can't use `--all-targets` because `--examples` doesn't work well
            cargo.arg("--bins");
            cargo.arg("--tests");
            cargo.arg("--benches");
        }

        // Cargo's output path in a given stage, compiled by a particular
        // compiler for the specified target.
        let stamp = BuildStamp::new(&builder.cargo_out(compiler, Mode::ToolRustc, target))
            .with_prefix("rust-analyzer-check");

        let _guard = builder.msg_check("rust-analyzer artifacts", target, None);
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }
}

/// Compiletest is implicitly "checked" when it gets built in order to run tests,
/// so this is mainly for people working on compiletest to run locally.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Compiletest {
    pub target: TargetSelection,
}

impl Step for Compiletest {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/compiletest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Compiletest { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let mode = if builder.config.compiletest_use_stage0_libtest {
            Mode::ToolBootstrap
        } else {
            Mode::ToolStd
        };

        let compiler = builder.compiler(
            if mode == Mode::ToolBootstrap { 0 } else { builder.top_stage },
            builder.config.host_target,
        );

        if mode != Mode::ToolBootstrap {
            builder.ensure(Rustc::new(self.target, builder));
        }

        let mut cargo = prepare_tool_cargo(
            builder,
            compiler,
            mode,
            self.target,
            builder.kind,
            "src/tools/compiletest",
            SourceType::InTree,
            &[],
        );

        cargo.allow_features(COMPILETEST_ALLOW_FEATURES);

        // For ./x.py clippy, don't run with --all-targets because
        // linting tests and benchmarks can produce very noisy results
        if builder.kind != Kind::Clippy {
            cargo.arg("--all-targets");
        }

        let stamp = BuildStamp::new(&builder.cargo_out(compiler, mode, self.target))
            .with_prefix("compiletest-check");

        let _guard = builder.msg_check("compiletest artifacts", self.target, None);
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }
}

macro_rules! tool_check_step {
    (
        $name:ident {
            // The part of this path after the final '/' is also used as a display name.
            path: $path:literal
            $(, alt_path: $alt_path:literal )*
            $(, default: $default:literal )?
            $( , )?
        }
    ) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            pub target: TargetSelection,
        }

        impl Step for $name {
            type Output = ();
            const ONLY_HOSTS: bool = true;
            /// Most of the tool-checks using this macro are run by default.
            const DEFAULT: bool = true $( && $default )?;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.paths(&[ $path, $( $alt_path ),* ])
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name { target: run.target });
            }

            fn run(self, builder: &Builder<'_>) {
                let Self { target } = self;
                run_tool_check_step(builder, target, stringify!($name), $path);
            }
        }
    }
}

/// Used by the implementation of `Step::run` in `tool_check_step!`.
fn run_tool_check_step(
    builder: &Builder<'_>,
    target: TargetSelection,
    step_type_name: &str,
    path: &str,
) {
    let display_name = path.rsplit('/').next().unwrap();
    let compiler = builder.compiler(builder.top_stage, builder.config.host_target);

    builder.ensure(Rustc::new(target, builder));

    let mut cargo = prepare_tool_cargo(
        builder,
        compiler,
        Mode::ToolRustc,
        target,
        builder.kind,
        path,
        // Currently, all of the tools that use this macro/function are in-tree.
        // If support for out-of-tree tools is re-added in the future, those
        // steps should probably be marked non-default so that the default
        // checks aren't affected by toolstate being broken.
        SourceType::InTree,
        &[],
    );

    // For ./x.py clippy, don't run with --all-targets because
    // linting tests and benchmarks can produce very noisy results
    if builder.kind != Kind::Clippy {
        cargo.arg("--all-targets");
    }

    let stamp = BuildStamp::new(&builder.cargo_out(compiler, Mode::ToolRustc, target))
        .with_prefix(&format!("{}-check", step_type_name.to_lowercase()));

    let _guard = builder.msg_check(format!("{display_name} artifacts"), target, None);
    run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
}

tool_check_step!(Rustdoc { path: "src/tools/rustdoc", alt_path: "src/librustdoc" });
// Clippy, miri and Rustfmt are hybrids. They are external tools, but use a git subtree instead
// of a submodule. Since the SourceType only drives the deny-warnings
// behavior, treat it as in-tree so that any new warnings in clippy will be
// rejected.
tool_check_step!(Clippy { path: "src/tools/clippy" });
tool_check_step!(Miri { path: "src/tools/miri" });
tool_check_step!(CargoMiri { path: "src/tools/miri/cargo-miri" });
tool_check_step!(Rustfmt { path: "src/tools/rustfmt" });
tool_check_step!(MiroptTestTools { path: "src/tools/miropt-test-tools" });
tool_check_step!(TestFloatParse { path: "src/tools/test-float-parse" });
tool_check_step!(FeaturesStatusDump { path: "src/tools/features-status-dump" });

tool_check_step!(Bootstrap { path: "src/bootstrap", default: false });

// `run-make-support` will be built as part of suitable run-make compiletest test steps, but support
// check to make it easier to work on.
tool_check_step!(RunMakeSupport { path: "src/tools/run-make-support", default: false });

/// Check step for the `coverage-dump` bootstrap tool. The coverage-dump tool
/// is used internally by coverage tests.
///
/// FIXME(Zalathar): This is temporarily separate from the other tool check
/// steps so that it can use the stage 0 compiler instead of `top_stage`,
/// without introducing conflicts with the stage 0 redesign (#119899).
///
/// After the stage 0 redesign lands, we can look into using the stage 0
/// compiler to check all bootstrap tools (#139170).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CoverageDump;

impl CoverageDump {
    const PATH: &str = "src/tools/coverage-dump";
}

impl Step for CoverageDump {
    type Output = ();

    /// Most contributors won't care about coverage-dump, so don't make their
    /// check builds slower unless they opt in and check it explicitly.
    const DEFAULT: bool = false;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path(Self::PATH)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Self {});
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        // Make sure we haven't forgotten any fields, if there are any.
        let Self {} = self;
        let display_name = "coverage-dump";
        let host = builder.config.host_target;
        let target = host;
        let mode = Mode::ToolBootstrap;

        let compiler = builder.compiler(0, host);
        let cargo = prepare_tool_cargo(
            builder,
            compiler,
            mode,
            target,
            builder.kind,
            Self::PATH,
            SourceType::InTree,
            &[],
        );

        let stamp = BuildStamp::new(&builder.cargo_out(compiler, mode, target))
            .with_prefix(&format!("{display_name}-check"));

        let _guard = builder.msg_tool(
            builder.kind,
            mode,
            display_name,
            compiler.stage,
            &compiler.host,
            &target,
        );
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }
}
