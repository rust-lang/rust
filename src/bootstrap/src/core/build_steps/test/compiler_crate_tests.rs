use super::shared::{RemoteCopyLibs, run_cargo_test};
use crate::core::build_steps::compile;
use crate::core::build_steps::tool::SourceType;
use crate::core::builder::{
    self, Alias, Builder, Compiler, RunConfig, ShouldRun, Step, crate_description,
};
use crate::core::config::TargetSelection;
use crate::utils::helpers::{self, target_supports_cranelift_backend};
use crate::{DocTests, Kind, Mode};

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

        builder.ensure(Crate {
            compiler: self.compiler,
            target: self.target,
            mode: Mode::Rustc,
            crates: self.crates,
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Crate {
    pub compiler: Compiler,
    pub target: TargetSelection,
    pub mode: Mode,
    pub crates: Vec<String>,
}

impl Step for Crate {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("sysroot")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;
        let host = run.build_triple();
        let compiler = builder.compiler_for(builder.top_stage, host, host);
        let crates = run
            .paths
            .iter()
            .map(|p| builder.crate_paths[&p.assert_single_path().path].clone())
            .collect();

        builder.ensure(Crate { compiler, target: run.target, mode: Mode::Std, crates });
    }

    /// Runs all unit tests plus documentation tests for a given crate defined
    /// by a `Cargo.toml` (single manifest)
    ///
    /// This is what runs tests for crates like the standard library, compiler, etc.
    /// It essentially is the driver for running `cargo test`.
    ///
    /// Currently this runs all tests for a DAG by passing a bunch of `-p foo`
    /// arguments, and those arguments are discovered from `cargo metadata`.
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;
        let mode = self.mode;

        // Prepare sysroot
        // See [field@compile::Std::force_recompile].
        builder.ensure(compile::Std::new(compiler, compiler.host).force_recompile(true));

        // If we're not doing a full bootstrap but we're testing a stage2
        // version of libstd, then what we're actually testing is the libstd
        // produced in stage1. Reflect that here by updating the compiler that
        // we're working with automatically.
        let compiler = builder.compiler_for(compiler.stage, compiler.host, target);

        let mut cargo = if builder.kind == Kind::Miri {
            if builder.top_stage == 0 {
                eprintln!("ERROR: `x.py miri` requires stage 1 or higher");
                std::process::exit(1);
            }

            // Build `cargo miri test` command
            // (Implicitly prepares target sysroot)
            let mut cargo = builder::Cargo::new(
                builder,
                compiler,
                mode,
                SourceType::InTree,
                target,
                Kind::MiriTest,
            );
            // This hack helps bootstrap run standard library tests in Miri. The issue is as
            // follows: when running `cargo miri test` on libcore, cargo builds a local copy of core
            // and makes it a dependency of the integration test crate. This copy duplicates all the
            // lang items, so the build fails. (Regular testing avoids this because the sysroot is a
            // literal copy of what `cargo build` produces, but since Miri builds its own sysroot
            // this does not work for us.) So we need to make it so that the locally built libcore
            // contains all the items from `core`, but does not re-define them -- we want to replace
            // the entire crate but a re-export of the sysroot crate. We do this by swapping out the
            // source file: if `MIRI_REPLACE_LIBRS_IF_NOT_TEST` is set and we are building a
            // `lib.rs` file, and a `lib.miri.rs` file exists in the same folder, we build that
            // instead. But crucially we only do that for the library, not the test builds.
            cargo.env("MIRI_REPLACE_LIBRS_IF_NOT_TEST", "1");
            // std needs to be built with `-Zforce-unstable-if-unmarked`. For some reason the builder
            // does not set this directly, but relies on the rustc wrapper to set it, and we are not using
            // the wrapper -- hence we have to set it ourselves.
            cargo.rustflag("-Zforce-unstable-if-unmarked");
            cargo
        } else {
            // Also prepare a sysroot for the target.
            if builder.config.build != target {
                builder.ensure(compile::Std::new(compiler, target).force_recompile(true));
                builder.ensure(RemoteCopyLibs { compiler, target });
            }

            // Build `cargo test` command
            builder::Cargo::new(builder, compiler, mode, SourceType::InTree, target, builder.kind)
        };

        match mode {
            Mode::Std => {
                if builder.kind == Kind::Miri {
                    // We can't use `std_cargo` as that uses `optimized-compiler-builtins` which
                    // needs host tools for the given target. This is similar to what `compile::Std`
                    // does when `is_for_mir_opt_tests` is true. There's probably a chance for
                    // de-duplication here... `std_cargo` should support a mode that avoids needing
                    // host tools.
                    cargo
                        .arg("--manifest-path")
                        .arg(builder.src.join("library/sysroot/Cargo.toml"));
                } else {
                    compile::std_cargo(builder, target, compiler.stage, &mut cargo);
                    // `std_cargo` actually does the wrong thing: it passes `--sysroot build/host/stage2`,
                    // but we want to use the force-recompile std we just built in `build/host/stage2-test-sysroot`.
                    // Override it.
                    if builder.download_rustc() && compiler.stage > 0 {
                        let sysroot = builder
                            .out
                            .join(compiler.host)
                            .join(format!("stage{}-test-sysroot", compiler.stage));
                        cargo.env("RUSTC_SYSROOT", sysroot);
                    }
                }
            }
            Mode::Rustc => {
                compile::rustc_cargo(builder, &mut cargo, target, &compiler, &self.crates);
            }
            _ => panic!("can only test libraries"),
        };

        run_cargo_test(
            cargo,
            &[],
            &self.crates,
            &self.crates[0],
            &*crate_description(&self.crates),
            compiler,
            target,
            builder,
        );
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
