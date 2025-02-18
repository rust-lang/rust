//! Rustdoc's self-tests and its adjacent crate tests have special logic that isn't observed by
//! other devtool tests, so they are put into their own module.

use std::env;
use std::path::PathBuf;

use super::test_helpers::run_cargo_test;
use crate::Mode;
use crate::core::build_steps::compile;
use crate::core::build_steps::tool::{self, SourceType};
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::helpers::{dylib_path, dylib_path_var};

/// Rustdoc is special in various ways, which is why this step is different from `Crate`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateRustdoc {
    host: TargetSelection,
}

impl Step for CrateRustdoc {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["src/librustdoc", "src/tools/rustdoc"])
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;

        builder.ensure(CrateRustdoc { host: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.host;

        let compiler = if builder.download_rustc() {
            builder.compiler(builder.top_stage, target)
        } else {
            // Use the previous stage compiler to reuse the artifacts that are created when running
            // compiletest for tests/rustdoc. If this used `compiler`, then it would cause rustdoc
            // to be built *again*, which isn't really necessary.
            builder.compiler_for(builder.top_stage, target, target)
        };
        // NOTE: normally `ensure(Rustc)` automatically runs `ensure(Std)` for us. However, when
        // using `download-rustc`, the rustc_private artifacts may be in a *different sysroot* from
        // the target rustdoc (`ci-rustc-sysroot` vs `stage2`). In that case, we need to ensure this
        // explicitly to make sure it ends up in the stage2 sysroot.
        builder.ensure(compile::Std::new(compiler, target));
        builder.ensure(compile::Rustc::new(compiler, target));

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            target,
            builder.kind,
            "src/tools/rustdoc",
            SourceType::InTree,
            &[],
        );
        if self.host.contains("musl") {
            cargo.arg("'-Ctarget-feature=-crt-static'");
        }

        // This is needed for running doctests on librustdoc. This is a bit of an unfortunate
        // interaction with how bootstrap works and how cargo sets up the dylib path, and the fact
        // that the doctest (in html/markdown.rs) links to rustc-private libs. For stage1, the
        // compiler host dylibs (in stage1/lib) are not the same as the target dylibs (in
        // stage1/lib/rustlib/...). This is different from a normal rust distribution where they are
        // the same.
        //
        // On the cargo side, normal tests use `target_process` which handles setting up the dylib
        // for a *target* (stage1/lib/rustlib/... in this case). However, for doctests it uses
        // `rustdoc_process` which only sets up the dylib path for the *host* (stage1/lib), which is
        // the wrong directory.
        //
        // Recall that we special-cased `compiler_for(top_stage)` above, so we always use stage1.
        //
        // It should be considered to just stop running doctests on librustdoc. There is only one
        // test, and it doesn't look too important. There might be other ways to avoid this, but it
        // seems pretty convoluted.
        //
        // See also https://github.com/rust-lang/rust/issues/13983 where the host vs target dylibs
        // for rustdoc are consistently tricky to deal with.
        //
        // Note that this set the host libdir for `download_rustc`, which uses a normal rust
        // distribution.
        let libdir = if builder.download_rustc() {
            builder.rustc_libdir(compiler)
        } else {
            builder.sysroot_target_libdir(compiler, target).to_path_buf()
        };
        let mut dylib_path = dylib_path();
        dylib_path.insert(0, PathBuf::from(&*libdir));
        cargo.env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

        run_cargo_test(
            cargo,
            &[],
            &["rustdoc:0.0.0".to_string()],
            "rustdoc",
            "rustdoc",
            target,
            builder,
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateRustdocJsonTypes {
    host: TargetSelection,
}

impl Step for CrateRustdocJsonTypes {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/rustdoc-json-types")
    }

    fn make_run(run: RunConfig<'_>) {
        let builder = run.builder;

        builder.ensure(CrateRustdocJsonTypes { host: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.host;

        // Use the previous stage compiler to reuse the artifacts that are created when running
        // compiletest for tests/rustdoc. If this used `compiler`, then it would cause rustdoc to be
        // built *again*, which isn't really necessary.
        let compiler = builder.compiler_for(builder.top_stage, target, target);
        builder.ensure(compile::Rustc::new(compiler, target));

        let cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            target,
            builder.kind,
            "src/rustdoc-json-types",
            SourceType::InTree,
            &[],
        );

        // FIXME: this looks very wrong, libtest doesn't accept `-C` arguments and the quotes are
        // fishy.
        let libtest_args = if self.host.contains("musl") {
            ["'-Ctarget-feature=-crt-static'"].as_slice()
        } else {
            &[]
        };

        run_cargo_test(
            cargo,
            libtest_args,
            &["rustdoc-json-types".to_string()],
            "rustdoc-json-types",
            "rustdoc-json-types",
            target,
            builder,
        );
    }
}
