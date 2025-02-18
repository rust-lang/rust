//! Build-and-run steps for `./x.py test` test fixtures
//!
//! `./x.py test` (aka [`Kind::Test`]) is currently allowed to reach build steps in other modules.
//! However, this contains ~all test parts we expect people to be able to build and run locally.

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
pub(crate) use doc_tool_tests::{HtmlCheck, Linkcheck, LintDocs, TierCheck};
pub(crate) use miri_tests::{CargoMiri, Miri};
pub(crate) use rustdoc_tests::{CrateRustdoc, CrateRustdocJsonTypes, RustdocTheme};
pub(crate) use std_tests::TestFloatParse;
pub(crate) use test_helpers::Crate;
pub(crate) use tidy::Tidy;
