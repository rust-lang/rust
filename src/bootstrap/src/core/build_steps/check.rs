//! Implementation of compiling the compiler and standard library, in "check"-based modes.

use std::fs;
use std::path::{Path, PathBuf};

use crate::core::build_steps::compile::{
    add_to_sysroot, run_cargo, rustc_cargo, rustc_cargo_env, std_cargo, std_crates_for_run_make,
};
use crate::core::build_steps::tool;
use crate::core::build_steps::tool::{
    COMPILETEST_ALLOW_FEATURES, SourceType, TEST_FLOAT_PARSE_ALLOW_FEATURES, ToolTargetBuildMode,
    get_tool_target_compiler, prepare_tool_cargo,
};
use crate::core::builder::{
    self, Alias, Builder, Cargo, Kind, RunConfig, ShouldRun, Step, StepMetadata, crate_description,
};
use crate::core::config::TargetSelection;
use crate::utils::build_stamp::{self, BuildStamp};
use crate::{CodegenBackendKind, Compiler, Mode, Subcommand, t};

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
}

impl Step for Std {
    type Output = BuildStamp;
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
            build_compiler: prepare_compiler_for_check(run.builder, run.target, Mode::Std)
                .build_compiler(),
            target: run.target,
            crates,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let build_compiler = self.build_compiler;
        let target = self.target;

        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
            Mode::Std,
            SourceType::InTree,
            target,
            Kind::Check,
        );

        std_cargo(builder, target, &mut cargo);
        if matches!(builder.config.cmd, Subcommand::Fix) {
            // By default, cargo tries to fix all targets. Tell it not to fix tests until we've added `test` to the sysroot.
            cargo.arg("--lib");
        }

        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard = builder.msg(
            Kind::Check,
            format_args!("library artifacts{}", crate_description(&self.crates)),
            Mode::Std,
            build_compiler,
            target,
        );

        let check_stamp =
            build_stamp::libstd_stamp(builder, build_compiler, target).with_prefix("check");
        run_cargo(
            builder,
            cargo,
            builder.config.free_args.clone(),
            &check_stamp,
            vec![],
            true,
            false,
        );

        drop(_guard);

        // don't check test dependencies if we haven't built libtest
        if !self.crates.iter().any(|krate| krate == "test") {
            return check_stamp;
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

        std_cargo(builder, target, &mut cargo);

        // Explicitly pass -p for all dependencies krates -- this will force cargo
        // to also check the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let stamp =
            build_stamp::libstd_stamp(builder, build_compiler, target).with_prefix("check-test");
        let _guard = builder.msg(
            Kind::Check,
            "library test/bench/example targets",
            Mode::Std,
            build_compiler,
            target,
        );
        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
        check_stamp
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::check("std", self.target).built_by(self.build_compiler))
    }
}

/// Represents a proof that rustc was **checked**.
/// Contains directories with .rmeta files generated by checking rustc for a specific
/// target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RmetaSysroot {
    host_dir: PathBuf,
    target_dir: PathBuf,
}

impl RmetaSysroot {
    /// Copy rmeta artifacts from the given `stamp` into a sysroot located at `directory`.
    fn from_stamp(
        builder: &Builder<'_>,
        stamp: BuildStamp,
        target: TargetSelection,
        directory: &Path,
    ) -> Self {
        let host_dir = directory.join("host");
        let target_dir = directory.join(target);
        let _ = fs::remove_dir_all(directory);
        t!(fs::create_dir_all(directory));
        add_to_sysroot(builder, &target_dir, &host_dir, &stamp);

        Self { host_dir, target_dir }
    }

    /// Configure the given cargo invocation so that the compiled crate will be able to use
    /// rustc .rmeta artifacts that were previously generated.
    fn configure_cargo(&self, cargo: &mut Cargo) {
        cargo.append_to_env(
            "RUSTC_ADDITIONAL_SYSROOT_PATHS",
            format!("{},{}", self.host_dir.to_str().unwrap(), self.target_dir.to_str().unwrap()),
            ",",
        );
    }
}

/// Checks rustc using the given `build_compiler` for the given `target`, and produces
/// a sysroot in the build directory that stores the generated .rmeta files.
///
/// This step exists so that we can store the generated .rmeta artifacts into a separate
/// directory, instead of copying them into the sysroot of `build_compiler`, which would
/// "pollute" it (that is especially problematic for the external stage0 rustc).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrepareRustcRmetaSysroot {
    build_compiler: CompilerForCheck,
    target: TargetSelection,
}

impl PrepareRustcRmetaSysroot {
    fn new(build_compiler: CompilerForCheck, target: TargetSelection) -> Self {
        Self { build_compiler, target }
    }
}

impl Step for PrepareRustcRmetaSysroot {
    type Output = RmetaSysroot;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        // Check rustc
        let stamp = builder.ensure(Rustc::from_build_compiler(
            self.build_compiler.clone(),
            self.target,
            vec![],
        ));

        let build_compiler = self.build_compiler.build_compiler();

        // Copy the generated rmeta artifacts to a separate directory
        let dir = builder
            .out
            .join(build_compiler.host)
            .join(format!("stage{}-rustc-rmeta-artifacts", build_compiler.stage + 1));
        RmetaSysroot::from_stamp(builder, stamp, self.target, &dir)
    }
}

/// Checks std using the given `build_compiler` for the given `target`, and produces
/// a sysroot in the build directory that stores the generated .rmeta files.
///
/// This step exists so that we can store the generated .rmeta artifacts into a separate
/// directory, instead of copying them into the sysroot of `build_compiler`, which would
/// "pollute" it (that is especially problematic for the external stage0 rustc).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrepareStdRmetaSysroot {
    build_compiler: Compiler,
    target: TargetSelection,
}

impl PrepareStdRmetaSysroot {
    fn new(build_compiler: Compiler, target: TargetSelection) -> Self {
        Self { build_compiler, target }
    }
}

impl Step for PrepareStdRmetaSysroot {
    type Output = RmetaSysroot;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        // Check std
        let stamp = builder.ensure(Std {
            build_compiler: self.build_compiler,
            target: self.target,
            crates: vec![],
        });

        // Copy the generated rmeta artifacts to a separate directory
        let dir = builder
            .out
            .join(self.build_compiler.host)
            .join(format!("stage{}-std-rmeta-artifacts", self.build_compiler.stage));

        RmetaSysroot::from_stamp(builder, stamp, self.target, &dir)
    }
}

/// Checks rustc using `build_compiler`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    /// Compiler that will check this rustc.
    pub build_compiler: CompilerForCheck,
    pub target: TargetSelection,
    /// Whether to build only a subset of crates.
    ///
    /// This shouldn't be used from other steps; see the comment on [`compile::Rustc`].
    ///
    /// [`compile::Rustc`]: crate::core::build_steps::compile::Rustc
    crates: Vec<String>,
}

impl Rustc {
    pub fn new(builder: &Builder<'_>, target: TargetSelection, crates: Vec<String>) -> Self {
        let build_compiler = prepare_compiler_for_check(builder, target, Mode::Rustc);
        Self::from_build_compiler(build_compiler, target, crates)
    }

    fn from_build_compiler(
        build_compiler: CompilerForCheck,
        target: TargetSelection,
        crates: Vec<String>,
    ) -> Self {
        Self { build_compiler, target, crates }
    }
}

impl Step for Rustc {
    type Output = BuildStamp;
    const IS_HOST: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("rustc-main").path("compiler")
    }

    fn make_run(run: RunConfig<'_>) {
        let crates = run.make_run_crates(Alias::Compiler);
        run.builder.ensure(Rustc::new(run.builder, run.target, crates));
    }

    /// Check the compiler.
    ///
    /// This will check the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    ///
    /// If we check a stage 2 compiler, we will have to first build a stage 1 compiler to check it.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let build_compiler = self.build_compiler.build_compiler;
        let target = self.target;

        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            Kind::Check,
        );

        rustc_cargo(builder, &mut cargo, target, &build_compiler, &self.crates);
        self.build_compiler.configure_cargo(&mut cargo);

        // Explicitly pass -p for all compiler crates -- this will force cargo
        // to also check the tests/benches/examples for these crates, rather
        // than just the leaf crate.
        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        let _guard = builder.msg(
            Kind::Check,
            format_args!("compiler artifacts{}", crate_description(&self.crates)),
            Mode::Rustc,
            self.build_compiler.build_compiler(),
            target,
        );

        let stamp =
            build_stamp::librustc_stamp(builder, build_compiler, target).with_prefix("check");

        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);

        stamp
    }

    fn metadata(&self) -> Option<StepMetadata> {
        let metadata = StepMetadata::check("rustc", self.target)
            .built_by(self.build_compiler.build_compiler());
        let metadata = if self.crates.is_empty() {
            metadata
        } else {
            metadata.with_metadata(format!("({} crates)", self.crates.len()))
        };
        Some(metadata)
    }
}

/// Represents a compiler that can check something.
///
/// If the compiler was created for `Mode::ToolRustcPrivate` or `Mode::Codegen`, it will also contain
/// .rmeta artifacts from rustc that was already checked using `build_compiler`.
///
/// All steps that use this struct in a "general way" (i.e. they don't know exactly what kind of
/// thing is being built) should call `configure_cargo` to ensure that the rmeta artifacts are
/// properly linked, if present.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompilerForCheck {
    build_compiler: Compiler,
    rustc_rmeta_sysroot: Option<RmetaSysroot>,
    std_rmeta_sysroot: Option<RmetaSysroot>,
}

impl CompilerForCheck {
    pub fn build_compiler(&self) -> Compiler {
        self.build_compiler
    }

    /// If there are any rustc rmeta artifacts available, configure the Cargo invocation
    /// so that the artifact being built can find them.
    pub fn configure_cargo(&self, cargo: &mut Cargo) {
        if let Some(sysroot) = &self.rustc_rmeta_sysroot {
            sysroot.configure_cargo(cargo);
        }
        if let Some(sysroot) = &self.std_rmeta_sysroot {
            sysroot.configure_cargo(cargo);
        }
    }
}

/// Prepare the standard library for checking something (that requires stdlib) using
/// `build_compiler`.
fn prepare_std(
    builder: &Builder<'_>,
    build_compiler: Compiler,
    target: TargetSelection,
) -> Option<RmetaSysroot> {
    // We need to build the host stdlib even if we only check, to compile build scripts and proc
    // macros
    builder.std(build_compiler, builder.host_target);

    // If we're cross-compiling, we generate the rmeta files for the given target
    // This check has to be here, because if we generate both .so and .rmeta files, rustc will fail,
    // as it will have multiple candidates for linking.
    if builder.host_target != target {
        Some(builder.ensure(PrepareStdRmetaSysroot::new(build_compiler, target)))
    } else {
        None
    }
}

/// Prepares a compiler that will check something with the given `mode`.
pub fn prepare_compiler_for_check(
    builder: &Builder<'_>,
    target: TargetSelection,
    mode: Mode,
) -> CompilerForCheck {
    let host = builder.host_target;

    let mut rustc_rmeta_sysroot = None;
    let mut std_rmeta_sysroot = None;
    let build_compiler = match mode {
        Mode::ToolBootstrap => builder.compiler(0, host),
        // We could also only check std here and use `prepare_std`, but `ToolTarget` is currently
        // only used for running in-tree Clippy on bootstrap tools, so it does not seem worth it to
        // optimize it. Therefore, here we build std for the target, instead of just checking it.
        Mode::ToolTarget => get_tool_target_compiler(builder, ToolTargetBuildMode::Build(target)),
        Mode::ToolStd => {
            if builder.config.compile_time_deps {
                // When --compile-time-deps is passed, we can't use any rustc
                // other than the bootstrap compiler. Luckily build scripts and
                // proc macros for tools are unlikely to need nightly.
                builder.compiler(0, host)
            } else {
                // These tools require the local standard library to be checked
                let build_compiler = builder.compiler(builder.top_stage, host);
                std_rmeta_sysroot = prepare_std(builder, build_compiler, target);
                build_compiler
            }
        }
        Mode::ToolRustcPrivate | Mode::Codegen => {
            // Check Rustc to produce the required rmeta artifacts for rustc_private, and then
            // return the build compiler that was used to check rustc.
            // We do not need to check examples/tests/etc. of Rustc for rustc_private, so we pass
            // an empty set of crates, which will avoid using `cargo -p`.
            let compiler_for_rustc = prepare_compiler_for_check(builder, target, Mode::Rustc);
            rustc_rmeta_sysroot = Some(
                builder.ensure(PrepareRustcRmetaSysroot::new(compiler_for_rustc.clone(), target)),
            );
            let build_compiler = compiler_for_rustc.build_compiler();

            // To check a rustc_private tool, we also need to check std that it will link to
            std_rmeta_sysroot = prepare_std(builder, build_compiler, target);
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
            let build_compiler = builder.compiler(stage, host);

            // To check rustc, we need to check std that it will link to
            std_rmeta_sysroot = prepare_std(builder, build_compiler, target);
            build_compiler
        }
        Mode::Std => {
            // When checking std stage N, we want to do it with the stage N compiler
            // Note: we don't need to build the host stdlib here, because when compiling std, the
            // stage 0 stdlib is used to compile build scripts and proc macros.
            builder.compiler(builder.top_stage, host)
        }
    };
    CompilerForCheck { build_compiler, rustc_rmeta_sysroot, std_rmeta_sysroot }
}

/// Check the Cranelift codegen backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CraneliftCodegenBackend {
    build_compiler: CompilerForCheck,
    target: TargetSelection,
}

impl Step for CraneliftCodegenBackend {
    type Output = ();

    const IS_HOST: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rustc_codegen_cranelift").alias("cg_clif")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CraneliftCodegenBackend {
            build_compiler: prepare_compiler_for_check(run.builder, run.target, Mode::Codegen),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let build_compiler = self.build_compiler.build_compiler();
        let target = self.target;

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
            .arg(builder.src.join("compiler/rustc_codegen_cranelift/Cargo.toml"));
        rustc_cargo_env(builder, &mut cargo, target);
        self.build_compiler.configure_cargo(&mut cargo);

        let _guard = builder.msg(
            Kind::Check,
            "rustc_codegen_cranelift",
            Mode::Codegen,
            build_compiler,
            target,
        );

        let stamp = build_stamp::codegen_backend_stamp(
            builder,
            build_compiler,
            target,
            &CodegenBackendKind::Cranelift,
        )
        .with_prefix("check");

        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(
            StepMetadata::check("rustc_codegen_cranelift", self.target)
                .built_by(self.build_compiler.build_compiler()),
        )
    }
}

/// Check the GCC codegen backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GccCodegenBackend {
    build_compiler: CompilerForCheck,
    target: TargetSelection,
}

impl Step for GccCodegenBackend {
    type Output = ();

    const IS_HOST: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rustc_codegen_gcc").alias("cg_gcc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(GccCodegenBackend {
            build_compiler: prepare_compiler_for_check(run.builder, run.target, Mode::Codegen),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        // FIXME: remove once https://github.com/rust-lang/rust/issues/112393 is resolved
        if builder.build.config.vendor {
            println!("Skipping checking of `rustc_codegen_gcc` with vendoring enabled.");
            return;
        }

        let build_compiler = self.build_compiler.build_compiler();
        let target = self.target;

        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
            Mode::Codegen,
            SourceType::InTree,
            target,
            builder.kind,
        );

        cargo.arg("--manifest-path").arg(builder.src.join("compiler/rustc_codegen_gcc/Cargo.toml"));
        rustc_cargo_env(builder, &mut cargo, target);
        self.build_compiler.configure_cargo(&mut cargo);

        let _guard =
            builder.msg(Kind::Check, "rustc_codegen_gcc", Mode::Codegen, build_compiler, target);

        let stamp = build_stamp::codegen_backend_stamp(
            builder,
            build_compiler,
            target,
            &CodegenBackendKind::Gcc,
        )
        .with_prefix("check");

        run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(
            StepMetadata::check("rustc_codegen_gcc", self.target)
                .built_by(self.build_compiler.build_compiler()),
        )
    }
}

macro_rules! tool_check_step {
    (
        $name:ident {
            // The part of this path after the final '/' is also used as a display name.
            path: $path:literal
            $(, alt_path: $alt_path:literal )*
            // Closure that returns `Mode` based on the passed `&Builder<'_>`
            , mode: $mode:expr
            // Subset of nightly features that are allowed to be used when checking
            $(, allow_features: $allow_features:expr )?
            // Features that should be enabled when checking
            $(, enable_features: [$($enable_features:expr),*] )?
            $(, default: $default:literal )?
            $( , )?
        }
    ) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            compiler: CompilerForCheck,
            target: TargetSelection,
        }

        impl Step for $name {
            type Output = ();
            const IS_HOST: bool = true;
            /// Most of the tool-checks using this macro are run by default.
            const DEFAULT: bool = true $( && $default )?;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.paths(&[ $path, $( $alt_path ),* ])
            }

            fn make_run(run: RunConfig<'_>) {
                let target = run.target;
                let builder = run.builder;
                let mode = $mode(builder);

                let compiler = prepare_compiler_for_check(run.builder, target, mode);

                // It doesn't make sense to cross-check bootstrap tools
                if mode == Mode::ToolBootstrap && target != run.builder.host_target {
                    println!("WARNING: not checking bootstrap tool {} for target {target} as it is a bootstrap (host-only) tool", stringify!($path));
                    return;
                };

                run.builder.ensure($name { target, compiler });
            }

            fn run(self, builder: &Builder<'_>) {
                let Self { target, compiler } = self;
                let allow_features = {
                    let mut _value = "";
                    $( _value = $allow_features; )?
                    _value
                };
                let extra_features: &[&str] = &[$($($enable_features),*)?];
                let mode = $mode(builder);
                run_tool_check_step(builder, compiler, target, $path, mode, allow_features, extra_features);
            }

            fn metadata(&self) -> Option<StepMetadata> {
                Some(StepMetadata::check(stringify!($name), self.target).built_by(self.compiler.build_compiler))
            }
        }
    }
}

/// Used by the implementation of `Step::run` in `tool_check_step!`.
fn run_tool_check_step(
    builder: &Builder<'_>,
    compiler: CompilerForCheck,
    target: TargetSelection,
    path: &str,
    mode: Mode,
    allow_features: &str,
    extra_features: &[&str],
) {
    let display_name = path.rsplit('/').next().unwrap();

    let build_compiler = compiler.build_compiler();

    let extra_features = extra_features.iter().map(|f| f.to_string()).collect::<Vec<String>>();
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
        &extra_features,
    );
    cargo.allow_features(allow_features);
    compiler.configure_cargo(&mut cargo);

    // FIXME: check bootstrap doesn't currently work when multiple targets are checked
    // FIXME: rust-analyzer does not work with --all-targets
    if display_name == "rust-analyzer" {
        cargo.arg("--bins");
        cargo.arg("--tests");
        cargo.arg("--benches");
    } else {
        cargo.arg("--all-targets");
    }

    let stamp = BuildStamp::new(&builder.cargo_out(build_compiler, mode, target))
        .with_prefix(&format!("{display_name}-check"));

    let _guard = builder.msg(builder.kind, display_name, mode, build_compiler, target);
    run_cargo(builder, cargo, builder.config.free_args.clone(), &stamp, vec![], true, false);
}

tool_check_step!(Rustdoc {
    path: "src/tools/rustdoc",
    alt_path: "src/librustdoc",
    mode: |_builder| Mode::ToolRustcPrivate
});
// Clippy, miri and Rustfmt are hybrids. They are external tools, but use a git subtree instead
// of a submodule. Since the SourceType only drives the deny-warnings
// behavior, treat it as in-tree so that any new warnings in clippy will be
// rejected.
tool_check_step!(Clippy { path: "src/tools/clippy", mode: |_builder| Mode::ToolRustcPrivate });
tool_check_step!(Miri { path: "src/tools/miri", mode: |_builder| Mode::ToolRustcPrivate });
tool_check_step!(CargoMiri {
    path: "src/tools/miri/cargo-miri",
    mode: |_builder| Mode::ToolRustcPrivate
});
tool_check_step!(Rustfmt { path: "src/tools/rustfmt", mode: |_builder| Mode::ToolRustcPrivate });
tool_check_step!(RustAnalyzer {
    path: "src/tools/rust-analyzer",
    mode: |_builder| Mode::ToolRustcPrivate,
    allow_features: tool::RustAnalyzer::ALLOW_FEATURES,
    enable_features: ["in-rust-tree"],
});
tool_check_step!(MiroptTestTools {
    path: "src/tools/miropt-test-tools",
    mode: |_builder| Mode::ToolBootstrap
});
// We want to test the local std
tool_check_step!(TestFloatParse {
    path: "src/tools/test-float-parse",
    mode: |_builder| Mode::ToolStd,
    allow_features: TEST_FLOAT_PARSE_ALLOW_FEATURES
});
tool_check_step!(FeaturesStatusDump {
    path: "src/tools/features-status-dump",
    mode: |_builder| Mode::ToolBootstrap
});

tool_check_step!(Bootstrap {
    path: "src/bootstrap",
    mode: |_builder| Mode::ToolBootstrap,
    default: false
});

// `run-make-support` will be built as part of suitable run-make compiletest test steps, but support
// check to make it easier to work on.
tool_check_step!(RunMakeSupport {
    path: "src/tools/run-make-support",
    mode: |_builder| Mode::ToolBootstrap,
    default: false
});

tool_check_step!(CoverageDump {
    path: "src/tools/coverage-dump",
    mode: |_builder| Mode::ToolBootstrap,
    default: false
});

// Compiletest is implicitly "checked" when it gets built in order to run tests,
// so this is mainly for people working on compiletest to run locally.
tool_check_step!(Compiletest {
    path: "src/tools/compiletest",
    mode: |builder: &Builder<'_>| if builder.config.compiletest_use_stage0_libtest {
        Mode::ToolBootstrap
    } else {
        Mode::ToolStd
    },
    allow_features: COMPILETEST_ALLOW_FEATURES,
    default: false,
});

tool_check_step!(Linkchecker {
    path: "src/tools/linkchecker",
    mode: |_builder| Mode::ToolBootstrap,
    default: false
});

tool_check_step!(BumpStage0 {
    path: "src/tools/bump-stage0",
    mode: |_builder| Mode::ToolBootstrap,
    default: false
});
