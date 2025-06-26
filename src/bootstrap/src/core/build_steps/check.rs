//! Implementation of compiling the compiler and standard library, in "check"-based modes.

use build_helper::exit;

use crate::core::build_steps::compile::{
    add_to_sysroot, run_cargo, rustc_cargo, rustc_cargo_env, std_cargo, std_crates_for_run_make,
};
use crate::core::build_steps::tool::{COMPILETEST_ALLOW_FEATURES, SourceType, prepare_tool_cargo};
use crate::core::builder::{
    self, Alias, Builder, Kind, RunConfig, ShouldRun, Step, StepMetadata, crate_description,
};
use crate::core::config::TargetSelection;
use crate::utils::build_stamp::{self, BuildStamp};
use crate::{Compiler, Mode, Subcommand};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    /// Compiler that will check this std.
    pub build_compiler: Compiler,
    pub target: TargetSelection,
    /// Whether to build only a subset of crates.
    ///
    /// This shouldn't be used from other steps; see the comment on [`compile::Rustc`].
    ///
    /// [`compile::Rustc`]: crate::core::build_steps::compile::Rustc
    crates: Vec<String>,
}

impl Std {
    const CRATE_OR_DEPS: &[&str] = &["sysroot", "coretests", "alloctests"];

    pub fn new(build_compiler: Compiler, target: TargetSelection) -> Self {
        Self { build_compiler, target, crates: vec![] }
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
        run.builder.ensure(Std {
            build_compiler: run.builder.compiler(run.builder.top_stage, run.target),
            target: run.target,
            crates,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        if !builder.download_rustc() && builder.config.skip_std_check_if_no_download_rustc {
            eprintln!(
                "WARNING: `--skip-std-check-if-no-download-rustc` flag was passed and `rust.download-rustc` is not available. Skipping."
            );
            return;
        }

        let build_compiler = self.build_compiler;
        let stage = build_compiler.stage;
        let target = self.target;

        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            Kind::Check,
        );

        std_cargo(builder, target, stage, &mut cargo);
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

        let stamp = build_stamp::libstd_stamp(builder, build_compiler, target).with_prefix("check");
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);

        drop(_guard);

        // don't check test dependencies if we haven't built libtest
        if !self.crates.iter().any(|krate| krate == "test") {
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
            build_compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            Kind::Check,
        );

        std_cargo(builder, target, build_compiler.stage, &mut cargo);

        // Explicitly pass -p for all dependencies krates -- this will force cargo
        // to also check the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let stamp =
            build_stamp::libstd_stamp(builder, build_compiler, target).with_prefix("check-test");
        let _guard = builder.msg_check("library test/bench/example targets", target, Some(stage));
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::check("std", self.target).built_by(self.build_compiler))
    }
}

fn default_compiler_for_checking_rustc(builder: &Builder<'_>) -> Compiler {
    // When checking the stage N compiler, we want to do it with the stage N-1 compiler,
    builder.compiler(builder.top_stage - 1, builder.config.host_target)
}

/// Checks rustc using `build_compiler` and copies the built
/// .rmeta files into the sysroot of `build_compiler`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    /// Compiler that will check this rustc.
    pub build_compiler: Compiler,
    pub target: TargetSelection,
    /// Whether to build only a subset of crates.
    ///
    /// This shouldn't be used from other steps; see the comment on [`compile::Rustc`].
    ///
    /// [`compile::Rustc`]: crate::core::build_steps::compile::Rustc
    crates: Vec<String>,
}

impl Rustc {
    pub fn new(builder: &Builder<'_>, build_compiler: Compiler, target: TargetSelection) -> Self {
        let crates = builder
            .in_tree_crates("rustc-main", Some(target))
            .into_iter()
            .map(|krate| krate.name.to_string())
            .collect();
        Self { build_compiler, target, crates }
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
        run.builder.ensure(Rustc {
            target: run.target,
            build_compiler: default_compiler_for_checking_rustc(run.builder),
            crates,
        });
    }

    /// Check the compiler.
    ///
    /// This will check the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder<'_>) {
        if builder.top_stage < 2 && builder.config.host_target != self.target {
            eprintln!("Cannot do a cross-compilation check on stage 1, use stage 2");
            exit!(1);
        }

        let build_compiler = self.build_compiler;
        let target = self.target;

        // Build host std for compiling build scripts
        builder.std(build_compiler, build_compiler.host);

        // Build target std so that the checked rustc can link to it during the check
        // FIXME: maybe we can a way to only do a check of std here?
        // But for that we would have to copy the stdlib rmetas to the sysroot of the build
        // compiler, which conflicts with std rlibs, if we also build std.
        builder.std(build_compiler, target);

        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            Kind::Check,
        );

        rustc_cargo(builder, &mut cargo, target, &build_compiler, &self.crates);

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

        let stamp =
            build_stamp::librustc_stamp(builder, build_compiler, target).with_prefix("check");

        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);

        let libdir = builder.sysroot_target_libdir(build_compiler, target);
        let hostdir = builder.sysroot_target_libdir(build_compiler, build_compiler.host);
        add_to_sysroot(builder, &libdir, &hostdir, &stamp);
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::check("rustc", self.target).built_by(self.build_compiler))
    }
}

/// Prepares a build compiler sysroot that will check a `Mode::ToolRustc` tool.
/// Also checks rustc using this compiler, to prepare .rmetas that the tool will link to.
fn prepare_compiler_for_tool_rustc(builder: &Builder<'_>, target: TargetSelection) -> Compiler {
    // When we check tool stage N, we check it with compiler stage N-1
    let build_compiler = builder.compiler(builder.top_stage - 1, builder.config.host_target);
    builder.ensure(Rustc::new(builder, build_compiler, target));
    build_compiler
}

/// Checks a single codegen backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodegenBackend {
    pub build_compiler: Compiler,
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
        // FIXME: only check the backend(s) that were actually selected in run.paths
        let build_compiler = prepare_compiler_for_tool_rustc(run.builder, run.target);
        for &backend in &["cranelift", "gcc"] {
            run.builder.ensure(CodegenBackend { build_compiler, target: run.target, backend });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        // FIXME: remove once https://github.com/rust-lang/rust/issues/112393 is resolved
        if builder.build.config.vendor && self.backend == "gcc" {
            println!("Skipping checking of `rustc_codegen_gcc` with vendoring enabled.");
            return;
        }

        let build_compiler = self.build_compiler;
        let target = self.target;
        let backend = self.backend;

        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
            Mode::Codegen,
            SourceType::InTree,
            target,
            builder.kind,
        );

        cargo
            .arg("--manifest-path")
            .arg(builder.src.join(format!("compiler/rustc_codegen_{backend}/Cargo.toml")));
        rustc_cargo_env(builder, &mut cargo, target, build_compiler.stage);

        let _guard = builder.msg_check(&format!("rustc_codegen_{backend}"), target, None);

        let stamp = build_stamp::codegen_backend_stamp(builder, build_compiler, target, backend)
            .with_prefix("check");

        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::check(self.backend, self.target).built_by(self.build_compiler))
    }
}

/// Checks Rust analyzer that links to .rmetas from a checked rustc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RustAnalyzer {
    pub build_compiler: Compiler,
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
        let build_compiler = prepare_compiler_for_tool_rustc(run.builder, run.target);
        run.builder.ensure(RustAnalyzer { build_compiler, target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let build_compiler = self.build_compiler;
        let target = self.target;

        let mut cargo = prepare_tool_cargo(
            builder,
            build_compiler,
            Mode::ToolRustc,
            target,
            builder.kind,
            "src/tools/rust-analyzer",
            SourceType::InTree,
            &["in-rust-tree".to_owned()],
        );

        cargo.allow_features(crate::core::build_steps::tool::RustAnalyzer::ALLOW_FEATURES);

        cargo.arg("--bins");
        cargo.arg("--tests");
        cargo.arg("--benches");

        // Cargo's output path in a given stage, compiled by a particular
        // compiler for the specified target.
        let stamp = BuildStamp::new(&builder.cargo_out(build_compiler, Mode::ToolRustc, target))
            .with_prefix("rust-analyzer-check");

        let _guard = builder.msg_check("rust-analyzer artifacts", target, None);
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::check("rust-analyzer", self.target).built_by(self.build_compiler))
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
            builder.std(compiler, self.target);
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

        cargo.arg("--all-targets");

        let stamp = BuildStamp::new(&builder.cargo_out(compiler, mode, self.target))
            .with_prefix("compiletest-check");

        let _guard = builder.msg_check("compiletest artifacts", self.target, None);
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::check("compiletest", self.target))
    }
}

macro_rules! tool_check_step {
    (
        $name:ident {
            // The part of this path after the final '/' is also used as a display name.
            path: $path:literal
            $(, alt_path: $alt_path:literal )*
            , mode: $mode:path
            $(, default: $default:literal )?
            $( , )?
        }
    ) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            pub build_compiler: Compiler,
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
                let host = run.builder.config.host_target;
                let target = run.target;
                let build_compiler = match $mode {
                    Mode::ToolBootstrap => run.builder.compiler(0, host),
                    Mode::ToolStd => {
                        // A small number of tools rely on in-tree standard
                        // library crates (e.g. compiletest needs libtest).
                        let build_compiler = run.builder.compiler(run.builder.top_stage, host);
                        run.builder.std(build_compiler, host);
                        run.builder.std(build_compiler, target);
                        build_compiler
                    }
                    Mode::ToolRustc => {
                        prepare_compiler_for_tool_rustc(run.builder, target)
                    }
                    _ => panic!("unexpected mode for tool check step: {:?}", $mode),
                };
                run.builder.ensure($name { target, build_compiler });
            }

            fn run(self, builder: &Builder<'_>) {
                let Self { target, build_compiler } = self;
                run_tool_check_step(builder, build_compiler, target, $path, $mode);
            }

            fn metadata(&self) -> Option<StepMetadata> {
                Some(StepMetadata::check(stringify!($name), self.target).built_by(self.build_compiler))
            }
        }
    }
}

/// Used by the implementation of `Step::run` in `tool_check_step!`.
fn run_tool_check_step(
    builder: &Builder<'_>,
    build_compiler: Compiler,
    target: TargetSelection,
    path: &str,
    mode: Mode,
) {
    let display_name = path.rsplit('/').next().unwrap();

    let mut cargo = prepare_tool_cargo(
        builder,
        build_compiler,
        mode,
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

    // FIXME: check bootstrap doesn't currently work with --all-targets
    cargo.arg("--all-targets");

    let stamp = BuildStamp::new(&builder.cargo_out(build_compiler, mode, target))
        .with_prefix(&format!("{display_name}-check"));

    let stage = match mode {
        // Mode::ToolRustc is included here because of how msg_sysroot_tool prints stages
        Mode::Std | Mode::ToolRustc => build_compiler.stage,
        _ => build_compiler.stage + 1,
    };

    let _guard =
        builder.msg_tool(builder.kind, mode, display_name, stage, &build_compiler.host, &target);
    run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
}

tool_check_step!(Rustdoc {
    path: "src/tools/rustdoc",
    alt_path: "src/librustdoc",
    mode: Mode::ToolRustc
});
// Clippy, miri and Rustfmt are hybrids. They are external tools, but use a git subtree instead
// of a submodule. Since the SourceType only drives the deny-warnings
// behavior, treat it as in-tree so that any new warnings in clippy will be
// rejected.
tool_check_step!(Clippy { path: "src/tools/clippy", mode: Mode::ToolRustc });
tool_check_step!(Miri { path: "src/tools/miri", mode: Mode::ToolRustc });
tool_check_step!(CargoMiri { path: "src/tools/miri/cargo-miri", mode: Mode::ToolRustc });
tool_check_step!(Rustfmt { path: "src/tools/rustfmt", mode: Mode::ToolRustc });
tool_check_step!(MiroptTestTools {
    path: "src/tools/miropt-test-tools",
    mode: Mode::ToolBootstrap
});
// We want to test the local std
tool_check_step!(TestFloatParse { path: "src/tools/test-float-parse", mode: Mode::ToolStd });
tool_check_step!(FeaturesStatusDump {
    path: "src/tools/features-status-dump",
    mode: Mode::ToolBootstrap
});

tool_check_step!(Bootstrap { path: "src/bootstrap", mode: Mode::ToolBootstrap, default: false });

// `run-make-support` will be built as part of suitable run-make compiletest test steps, but support
// check to make it easier to work on.
tool_check_step!(RunMakeSupport {
    path: "src/tools/run-make-support",
    mode: Mode::ToolBootstrap,
    default: false
});

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
