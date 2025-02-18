//! Build-and-run steps for `./x.py test` test fixtures
//!
//! `./x.py test` (aka [`Kind::Test`]) is currently allowed to reach build steps in other modules.
//! However, this contains ~all test parts we expect people to be able to build and run locally.

// FIXME(jieyouxu): keeping unused imports here before all test steps are properly split out.
#![expect(unused_imports)]

use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::{env, fs, iter};

use clap_complete::shells;

use self::test_helpers::{RemoteCopyLibs, prepare_cargo_test, run_cargo_test};
use crate::core::build_steps::compile::run_cargo;
use crate::core::build_steps::doc::DocumentationFormat;
use crate::core::build_steps::llvm::get_llvm_version;
use crate::core::build_steps::synthetic_targets::MirOptPanicAbortSyntheticTarget;
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::build_steps::toolstate::ToolState;
use crate::core::build_steps::{compile, dist, llvm};
use crate::core::builder::{
    self, Alias, Builder, Compiler, Kind, RunConfig, ShouldRun, Step, crate_description,
};
use crate::core::config::TargetSelection;
use crate::core::config::flags::{Subcommand, get_completion};
use crate::utils::build_stamp::{self, BuildStamp};
use crate::utils::exec::{BootstrapCommand, command};
use crate::utils::helpers::{
    self, LldThreads, add_rustdoc_cargo_linker_args, dylib_path, dylib_path_var, linker_args,
    linker_flags, t, target_supports_cranelift_backend, up_to_date,
};
use crate::utils::render_tests::{add_flags_and_try_run_tests, try_run_tests};
use crate::{CLang, DocTests, GitRepo, Mode, PathSet, envify};

mod book_tests;
mod bootstrap_self_tests;
mod cargotest;
mod compiler_crate_tests;
mod compiletest_self_tests;
mod compiletest_suites;
mod devtool_tests;
mod dist_tool_tests;
mod doc_tool_tests;
mod miri_tests;
mod rustdoc_tests;
mod std_tests;
mod test_helpers;
mod tidy;

pub(crate) use book_tests::{
    EditionGuide, EmbeddedBook, ErrorIndex, Nomicon, Reference, RustByExample, RustcBook,
    RustdocBook, TheBook, UnstableBook,
};
pub(crate) use bootstrap_self_tests::{Bootstrap, CrateBootstrap, CrateBuildHelper};
pub(crate) use cargotest::Cargotest;
pub(crate) use compiler_crate_tests::{CodegenCranelift, CodegenGCC, CrateLibrustc};
pub(crate) use compiletest_self_tests::CompiletestTest;
pub(crate) use compiletest_suites::{
    Assembly, Codegen, CodegenUnits, Coverage, CoverageRunRustdoc, Crashes, CrateRunMakeSupport,
    Debuginfo, Incremental, MirOpt, Pretty, RunMake, Rustdoc, RustdocGUI, RustdocJSNotStd,
    RustdocJSStd, RustdocJson, RustdocUi, Ui, UiFullDeps,
};
pub(crate) use devtool_tests::{Cargo, Clippy, RustAnalyzer, Rustfmt};
pub(crate) use dist_tool_tests::{CollectLicenseMetadata, Distcheck, RustInstaller};
pub(crate) use doc_tool_tests::{HtmlCheck, Linkcheck};
pub(crate) use miri_tests::{CargoMiri, Miri};
pub(crate) use rustdoc_tests::{CrateRustdoc, CrateRustdocJsonTypes, RustdocTheme};
pub(crate) use std_tests::TestFloatParse;
pub(crate) use test_helpers::Crate;
pub(crate) use tidy::Tidy;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TierCheck {
    pub compiler: Compiler,
}

impl Step for TierCheck {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/tier-check")
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler =
            run.builder.compiler_for(run.builder.top_stage, run.builder.build.build, run.target);
        run.builder.ensure(TierCheck { compiler });
    }

    /// Tests the Platform Support page in the rustc book.
    fn run(self, builder: &Builder<'_>) {
        builder.ensure(compile::Std::new(self.compiler, self.compiler.host));
        let mut cargo = tool::prepare_tool_cargo(
            builder,
            self.compiler,
            Mode::ToolStd,
            self.compiler.host,
            Kind::Run,
            "src/tools/tier-check",
            SourceType::InTree,
            &[],
        );
        cargo.arg(builder.src.join("src/doc/rustc/src/platform-support.md"));
        cargo.arg(builder.rustc(self.compiler));
        if builder.is_verbose() {
            cargo.arg("--verbose");
        }

        let _guard = builder.msg(
            Kind::Test,
            self.compiler.stage,
            "platform support check",
            self.compiler.host,
            self.compiler.host,
        );
        BootstrapCommand::from(cargo).delay_failure().run(builder);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LintDocs {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for LintDocs {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/lint-docs")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(LintDocs {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    /// Tests that the lint examples in the rustc book generate the correct lints and have the
    /// expected format.
    fn run(self, builder: &Builder<'_>) {
        builder.ensure(crate::core::build_steps::doc::RustcBook {
            compiler: self.compiler,
            target: self.target,
            validate: true,
        });
    }
}
