//! Compiler crate tests and codegen backend tests.

#![warn(unused_imports)]

use super::test_helpers::Crate;
use crate::core::build_steps::compile;
use crate::core::build_steps::tool::SourceType;
use crate::core::builder::{self, Alias, Builder, Compiler, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::helpers::{self, target_supports_cranelift_backend};
use crate::{DocTests, Mode};

/// Runs `cargo test` for the compiler crates in `compiler/`.
///
/// (This step does not test `rustc_codegen_cranelift` or `rustc_codegen_gcc`, which have their own
/// separate test steps.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateLibrustc {
    compiler: Compiler,
    target: TargetSelection,
    crates: Vec<String>,
}

impl Step for CrateLibrustc {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("rustc-main").path("compiler")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = builder.compiler_for(builder.top_stage, host, host);
        let crates = run.make_run_crates(Alias::Compiler);

        builder.ensure(CrateLibrustc { compiler, target: run.target, crates });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(compile::Std::new(self.compiler, self.target));

        // To actually run the tests, delegate to a copy of the `Crate` step.
        builder.ensure(Crate {
            compiler: self.compiler,
            target: self.target,
            mode: Mode::Rustc,
            crates: self.crates,
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodegenCranelift {
    compiler: Compiler,
    target: TargetSelection,
}

impl Step for CodegenCranelift {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["compiler/rustc_codegen_cranelift"])
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = run.builder.compiler_for(run.builder.top_stage, host, host);

        if builder.doc_tests == DocTests::Only {
            return;
        }

        if builder.download_rustc() {
            builder.info("CI rustc uses the default codegen backend. skipping");
            return;
        }

        if !target_supports_cranelift_backend(run.target) {
            builder.info("target not supported by rustc_codegen_cranelift. skipping");
            return;
        }

        if builder.remote_tested(run.target) {
            builder.info("remote testing is not supported by rustc_codegen_cranelift. skipping");
            return;
        }

        if !builder.config.codegen_backends(run.target).contains(&"cranelift".to_owned()) {
            builder.info("cranelift not in rust.codegen-backends. skipping");
            return;
        }

        builder.ensure(CodegenCranelift { compiler, target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(compile::Std::new(compiler, target));

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let build_cargo = || {
            let mut cargo = builder::Cargo::new(
                builder,
                compiler,
                Mode::Codegen, // Must be codegen to ensure dlopen on compiled dylibs works
                SourceType::InTree,
                target,
                Kind::Run,
            );

            cargo.current_dir(&builder.src.join("compiler/rustc_codegen_cranelift"));
            cargo
                .arg("--manifest-path")
                .arg(builder.src.join("compiler/rustc_codegen_cranelift/build_system/Cargo.toml"));
            compile::rustc_cargo_env(builder, &mut cargo, target, compiler.stage);

            // Avoid incremental cache issues when changing rustc
            cargo.env("CARGO_BUILD_INCREMENTAL", "false");

            cargo
        };

        builder.info(&format!(
            "{} cranelift stage{} ({} -> {})",
            Kind::Test.description(),
            compiler.stage,
            &compiler.host,
            target
        ));
        let _time = helpers::timeit(builder);

        // FIXME handle vendoring for source tarballs before removing the --skip-test below
        let download_dir = builder.out.join("cg_clif_download");

        // FIXME: Uncomment the `prepare` command below once vendoring is implemented.
        /*
        let mut prepare_cargo = build_cargo();
        prepare_cargo.arg("--").arg("prepare").arg("--download-dir").arg(&download_dir);
        #[allow(deprecated)]
        builder.config.try_run(&mut prepare_cargo.into()).unwrap();
        */

        let mut cargo = build_cargo();
        cargo
            .arg("--")
            .arg("test")
            .arg("--download-dir")
            .arg(&download_dir)
            .arg("--out-dir")
            .arg(builder.stage_out(compiler, Mode::ToolRustc).join("cg_clif"))
            .arg("--no-unstable-features")
            .arg("--use-backend")
            .arg("cranelift")
            // Avoid having to vendor the standard library dependencies
            .arg("--sysroot")
            .arg("llvm")
            // These tests depend on crates that are not yet vendored
            // FIXME remove once vendoring is handled
            .arg("--skip-test")
            .arg("testsuite.extended_sysroot");

        cargo.into_cmd().run(builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodegenGCC {
    compiler: Compiler,
    target: TargetSelection,
}

impl Step for CodegenGCC {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["compiler/rustc_codegen_gcc"])
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = run.builder.compiler_for(run.builder.top_stage, host, host);

        if builder.doc_tests == DocTests::Only {
            return;
        }

        if builder.download_rustc() {
            builder.info("CI rustc uses the default codegen backend. skipping");
            return;
        }

        let triple = run.target.triple;
        let target_supported =
            if triple.contains("linux") { triple.contains("x86_64") } else { false };
        if !target_supported {
            builder.info("target not supported by rustc_codegen_gcc. skipping");
            return;
        }

        if builder.remote_tested(run.target) {
            builder.info("remote testing is not supported by rustc_codegen_gcc. skipping");
            return;
        }

        if !builder.config.codegen_backends(run.target).contains(&"gcc".to_owned()) {
            builder.info("gcc not in rust.codegen-backends. skipping");
            return;
        }

        builder.ensure(CodegenGCC { compiler, target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(
            compile::Std::new(compiler, target)
                .extra_rust_args(&["-Csymbol-mangling-version=v0", "-Cpanic=abort"]),
        );

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let build_cargo = || {
            let mut cargo = builder::Cargo::new(
                builder,
                compiler,
                Mode::Codegen, // Must be codegen to ensure dlopen on compiled dylibs works
                SourceType::InTree,
                target,
                Kind::Run,
            );

            cargo.current_dir(&builder.src.join("compiler/rustc_codegen_gcc"));
            cargo
                .arg("--manifest-path")
                .arg(builder.src.join("compiler/rustc_codegen_gcc/build_system/Cargo.toml"));
            compile::rustc_cargo_env(builder, &mut cargo, target, compiler.stage);

            // Avoid incremental cache issues when changing rustc
            cargo.env("CARGO_BUILD_INCREMENTAL", "false");
            cargo.rustflag("-Cpanic=abort");

            cargo
        };

        builder.info(&format!(
            "{} GCC stage{} ({} -> {})",
            Kind::Test.description(),
            compiler.stage,
            &compiler.host,
            target
        ));
        let _time = helpers::timeit(builder);

        // FIXME: Uncomment the `prepare` command below once vendoring is implemented.
        /*
        let mut prepare_cargo = build_cargo();
        prepare_cargo.arg("--").arg("prepare");
        #[allow(deprecated)]
        builder.config.try_run(&mut prepare_cargo.into()).unwrap();
        */

        let mut cargo = build_cargo();

        cargo
            // cg_gcc's build system ignores RUSTFLAGS. pass some flags through CG_RUSTFLAGS instead.
            .env("CG_RUSTFLAGS", "-Alinker-messages")
            .arg("--")
            .arg("test")
            .arg("--use-system-gcc")
            .arg("--use-backend")
            .arg("gcc")
            .arg("--out-dir")
            .arg(builder.stage_out(compiler, Mode::ToolRustc).join("cg_gcc"))
            .arg("--release")
            .arg("--mini-tests")
            .arg("--std-tests");
        cargo.args(builder.config.test_args());

        cargo.into_cmd().run(builder);
    }
}
