//! Implementation of compiling the compiler and standard library, in "check"-based modes.

use crate::core::build_steps::compile::{
    add_to_sysroot, run_cargo, rustc_cargo, rustc_cargo_env, std_cargo, std_crates_for_run_make,
};
use crate::core::build_steps::tool;
use crate::core::build_steps::tool::{
    COMPILETEST_ALLOW_FEATURES, SourceType, ToolTargetBuildMode, get_tool_target_compiler,
    prepare_tool_cargo,
};
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
        if !run.builder.download_rustc() && run.builder.config.skip_std_check_if_no_download_rustc {
            eprintln!(
                "WARNING: `--skip-std-check-if-no-download-rustc` flag was passed and `rust.download-rustc` is not available. Skipping."
            );
            return;
        }

        if run.builder.config.compile_time_deps {
            // libstd doesn't have any important build scripts and can't have any proc macros
            return;
        }

        let crates = std_crates_for_run_make(&run);
        run.builder.ensure(Std {
            build_compiler: prepare_compiler_for_check(run.builder, run.target, Mode::Std),
            target: run.target,
            crates,
        });
    }

    fn run(self, builder: &Builder<'_>) {
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
            build_compiler: prepare_compiler_for_check(run.builder, run.target, Mode::Rustc),
            crates,
        });
    }

    /// Check the compiler.
    ///
    /// This will check the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    ///
    /// If we check a stage 2 compiler, we will have to first build a stage 1 compiler to check it.
    fn run(self, builder: &Builder<'_>) {
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

/// Prepares a compiler that will check something with the given `mode`.
fn prepare_compiler_for_check(
    builder: &Builder<'_>,
    target: TargetSelection,
    mode: Mode,
) -> Compiler {
    let host = builder.host_target;

    match mode {
        Mode::ToolBootstrap => builder.compiler(0, host),
        Mode::ToolTarget => get_tool_target_compiler(builder, ToolTargetBuildMode::Build(target)),
        Mode::ToolStd => {
            if builder.config.compile_time_deps {
                // When --compile-time-deps is passed, we can't use any rustc
                // other than the bootstrap compiler. Luckily build scripts and
                // proc macros for tools are unlikely to need nightly.
                return builder.compiler(0, host);
            }

            // These tools require the local standard library to be checked
            let build_compiler = builder.compiler(builder.top_stage, host);

            // We need to build the host stdlib to check the tool itself.
            // We need to build the target stdlib so that the tool can link to it.
            builder.std(build_compiler, host);
            // We could only check this library in theory, but `check::Std` doesn't copy rmetas
            // into `build_compiler`'s sysroot to avoid clashes with `.rlibs`, so we build it
            // instead.
            builder.std(build_compiler, target);
            build_compiler
        }
        Mode::ToolRustc | Mode::Codegen => {
            // FIXME: this is a hack, see description of Mode::Rustc below
            let stage = if host == target { builder.top_stage - 1 } else { builder.top_stage };
            // When checking tool stage N, we check it with compiler stage N-1
            let build_compiler = builder.compiler(stage, host);
            builder.ensure(Rustc::new(builder, build_compiler, target));
            build_compiler
        }
        Mode::Rustc => {
            // This is a horrible hack, because we actually change the compiler stage numbering
            // here. If you do `x check --stage 1 --host FOO`, we build stage 1 host rustc,
            // and use that to check stage 1 FOO rustc (which actually makes that stage 2 FOO
            // rustc).
            //
            // FIXME: remove this and either fix cross-compilation check on stage 2 (which has a
            // myriad of other problems) or disable cross-checking on stage 1.
            let stage = if host == target { builder.top_stage - 1 } else { builder.top_stage };
            builder.compiler(stage, host)
        }
        Mode::Std => {
            // When checking std stage N, we want to do it with the stage N compiler
            // Note: we don't need to build the host stdlib here, because when compiling std, the
            // stage 0 stdlib is used to compile build scripts and proc macros.
            builder.compiler(builder.top_stage, host)
        }
    }
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
        let build_compiler = prepare_compiler_for_check(run.builder, run.target, Mode::Codegen);
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

        let _guard = builder.msg_check(format!("rustc_codegen_{backend}"), target, None);

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
        let build_compiler = prepare_compiler_for_check(run.builder, run.target, Mode::ToolRustc);
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
        let build_compiler = prepare_compiler_for_check(builder, self.target, mode);

        let mut cargo = prepare_tool_cargo(
            builder,
            build_compiler,
            mode,
            self.target,
            builder.kind,
            "src/tools/compiletest",
            SourceType::InTree,
            &[],
        );

        cargo.allow_features(COMPILETEST_ALLOW_FEATURES);

        cargo.arg("--all-targets");

        let stamp = BuildStamp::new(&builder.cargo_out(build_compiler, mode, self.target))
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
            $(, allow_features: $allow_features:expr )?
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
                let target = run.target;
                let build_compiler = prepare_compiler_for_check(run.builder, target, $mode);

                // It doesn't make sense to cross-check bootstrap tools
                if $mode == Mode::ToolBootstrap && target != run.builder.host_target {
                    println!("WARNING: not checking bootstrap tool {} for target {target} as it is a bootstrap (host-only) tool", stringify!($path));
                    return;
                };

                run.builder.ensure($name { target, build_compiler });
            }

            fn run(self, builder: &Builder<'_>) {
                let Self { target, build_compiler } = self;
                let allow_features = {
                    let mut _value = "";
                    $( _value = $allow_features; )?
                    _value
                };
                run_tool_check_step(builder, build_compiler, target, $path, $mode, allow_features);
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
    allow_features: &str,
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
    cargo.allow_features(allow_features);

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
tool_check_step!(TestFloatParse {
    path: "src/tools/test-float-parse",
    mode: Mode::ToolStd,
    allow_features: tool::TestFloatParse::ALLOW_FEATURES
});
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

tool_check_step!(CoverageDump {
    path: "src/tools/coverage-dump",
    mode: Mode::ToolBootstrap,
    default: false
});
