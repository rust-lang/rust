//! This module isolates the compiletest APIs used by the rustdoc-gui-test tool.
//!
//! Thanks to this isolation layer, changes to compiletest directive parsing
//! might require changes to the items in this module, but shouldn't require
//! changes to rustdoc-gui-test itself.
//!
//! The current relationship between compiletest and rustdoc-gui-test is
//! awkward. Ideally, rustdoc-gui-test should either split off its own
//! directive parser and become fully independent, or be incorporated into
//! compiletest as another test mode.
//!
//! See <https://github.com/rust-lang/rust/issues/143827> for more context.

use std::path::Path;

use camino::{Utf8Path, Utf8PathBuf};

use crate::common::{CodegenBackend, Config, TestMode, TestSuite};
use crate::directives::TestProps;

/// Subset of compiletest directive values that are actually used by
/// rustdoc-gui-test.
#[derive(Debug)]
pub struct RustdocGuiTestProps {
    pub compile_flags: Vec<String>,
    pub run_flags: Vec<String>,
}

impl RustdocGuiTestProps {
    pub fn from_file(test_file_path: &Path) -> Self {
        let test_file_path = Utf8Path::from_path(test_file_path).unwrap();
        let config = incomplete_config_for_rustdoc_gui_test();

        let props = TestProps::from_file(test_file_path, None, &config);

        let TestProps { compile_flags, run_flags, .. } = props;
        Self { compile_flags, run_flags }
    }
}

/// Incomplete config intended for `src/tools/rustdoc-gui-test` **only** as
/// `src/tools/rustdoc-gui-test` wants to reuse `compiletest`'s directive -> test property
/// handling for `//@ {compile,run}-flags`, do not use for any other purpose.
///
/// FIXME(#143827): this setup feels very hacky. It so happens that `tests/rustdoc-gui/`
/// **only** uses `//@ {compile,run}-flags` for now and not any directives that actually rely on
/// info that is assumed available in a fully populated [`Config`].
fn incomplete_config_for_rustdoc_gui_test() -> Config {
    // FIXME(#143827): spelling this out intentionally, because this is questionable.
    //
    // For instance, `//@ ignore-stage1` will not work at all.
    Config {
        mode: TestMode::Rustdoc,
        // E.g. this has no sensible default tbh.
        suite: TestSuite::Ui,

        // Dummy values.
        edition: Default::default(),
        bless: Default::default(),
        fail_fast: Default::default(),
        compile_lib_path: Utf8PathBuf::default(),
        run_lib_path: Utf8PathBuf::default(),
        rustc_path: Utf8PathBuf::default(),
        cargo_path: Default::default(),
        stage0_rustc_path: Default::default(),
        query_rustc_path: Default::default(),
        rustdoc_path: Default::default(),
        coverage_dump_path: Default::default(),
        python: Default::default(),
        jsondocck_path: Default::default(),
        jsondoclint_path: Default::default(),
        llvm_filecheck: Default::default(),
        llvm_bin_dir: Default::default(),
        run_clang_based_tests_with: Default::default(),
        src_root: Utf8PathBuf::default(),
        src_test_suite_root: Utf8PathBuf::default(),
        build_root: Utf8PathBuf::default(),
        build_test_suite_root: Utf8PathBuf::default(),
        sysroot_base: Utf8PathBuf::default(),
        stage: Default::default(),
        stage_id: String::default(),
        debugger: Default::default(),
        run_ignored: Default::default(),
        with_rustc_debug_assertions: Default::default(),
        with_std_debug_assertions: Default::default(),
        filters: Default::default(),
        skip: Default::default(),
        filter_exact: Default::default(),
        force_pass_mode: Default::default(),
        run: Default::default(),
        runner: Default::default(),
        host_rustcflags: Default::default(),
        target_rustcflags: Default::default(),
        rust_randomized_layout: Default::default(),
        optimize_tests: Default::default(),
        target: Default::default(),
        host: Default::default(),
        cdb: Default::default(),
        cdb_version: Default::default(),
        gdb: Default::default(),
        gdb_version: Default::default(),
        lldb_version: Default::default(),
        llvm_version: Default::default(),
        system_llvm: Default::default(),
        android_cross_path: Default::default(),
        adb_path: Default::default(),
        adb_test_dir: Default::default(),
        adb_device_status: Default::default(),
        lldb_python_dir: Default::default(),
        verbose: Default::default(),
        color: Default::default(),
        remote_test_client: Default::default(),
        compare_mode: Default::default(),
        rustfix_coverage: Default::default(),
        has_html_tidy: Default::default(),
        has_enzyme: Default::default(),
        channel: Default::default(),
        git_hash: Default::default(),
        cc: Default::default(),
        cxx: Default::default(),
        cflags: Default::default(),
        cxxflags: Default::default(),
        ar: Default::default(),
        target_linker: Default::default(),
        host_linker: Default::default(),
        llvm_components: Default::default(),
        nodejs: Default::default(),
        force_rerun: Default::default(),
        only_modified: Default::default(),
        target_cfgs: Default::default(),
        builtin_cfg_names: Default::default(),
        supported_crate_types: Default::default(),
        nocapture: Default::default(),
        nightly_branch: Default::default(),
        git_merge_commit_email: Default::default(),
        profiler_runtime: Default::default(),
        diff_command: Default::default(),
        minicore_path: Default::default(),
        default_codegen_backend: CodegenBackend::Llvm,
        override_codegen_backend: None,
        bypass_ignore_backends: Default::default(),
    }
}
