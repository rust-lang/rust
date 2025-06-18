//! This module defines the `Build` struct, which represents the `[build]` table
//! in the `bootstrap.toml` configuration file.
//!
//! The `[build]` table contains global options that influence the overall build process,
//! such as default host and target triples, paths to tools, build directories, and
//! various feature flags. These options apply across different stages and components
//! unless specifically overridden by other configuration sections or command-line flags.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer};

use crate::core::config::toml::ReplaceOpt;
use crate::core::config::{Merge, StringOrBool};
use crate::{HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of various global build decisions.
    #[derive(Default)]
    struct Build {
        build: Option<String> = "build",
        description: Option<String> = "description",
        host: Option<Vec<String>> = "host",
        target: Option<Vec<String>> = "target",
        build_dir: Option<String> = "build-dir",
        cargo: Option<PathBuf> = "cargo",
        rustc: Option<PathBuf> = "rustc",
        rustfmt: Option<PathBuf> = "rustfmt",
        cargo_clippy: Option<PathBuf> = "cargo-clippy",
        docs: Option<bool> = "docs",
        compiler_docs: Option<bool> = "compiler-docs",
        library_docs_private_items: Option<bool> = "library-docs-private-items",
        docs_minification: Option<bool> = "docs-minification",
        submodules: Option<bool> = "submodules",
        gdb: Option<String> = "gdb",
        lldb: Option<String> = "lldb",
        nodejs: Option<String> = "nodejs",
        npm: Option<String> = "npm",
        python: Option<String> = "python",
        reuse: Option<String> = "reuse",
        locked_deps: Option<bool> = "locked-deps",
        vendor: Option<bool> = "vendor",
        full_bootstrap: Option<bool> = "full-bootstrap",
        bootstrap_cache_path: Option<PathBuf> = "bootstrap-cache-path",
        extended: Option<bool> = "extended",
        tools: Option<HashSet<String>> = "tools",
        tool: Option<HashMap<String, Tool>> = "tool",
        verbose: Option<usize> = "verbose",
        sanitizers: Option<bool> = "sanitizers",
        profiler: Option<bool> = "profiler",
        cargo_native_static: Option<bool> = "cargo-native-static",
        low_priority: Option<bool> = "low-priority",
        configure_args: Option<Vec<String>> = "configure-args",
        local_rebuild: Option<bool> = "local-rebuild",
        print_step_timings: Option<bool> = "print-step-timings",
        print_step_rusage: Option<bool> = "print-step-rusage",
        check_stage: Option<u32> = "check-stage",
        doc_stage: Option<u32> = "doc-stage",
        build_stage: Option<u32> = "build-stage",
        test_stage: Option<u32> = "test-stage",
        install_stage: Option<u32> = "install-stage",
        dist_stage: Option<u32> = "dist-stage",
        bench_stage: Option<u32> = "bench-stage",
        patch_binaries_for_nix: Option<bool> = "patch-binaries-for-nix",
        // NOTE: only parsed by bootstrap.py, `--feature build-metrics` enables metrics unconditionally
        metrics: Option<bool> = "metrics",
        android_ndk: Option<PathBuf> = "android-ndk",
        optimized_compiler_builtins: Option<bool> = "optimized-compiler-builtins",
        jobs: Option<u32> = "jobs",
        compiletest_diff_tool: Option<String> = "compiletest-diff-tool",
        compiletest_use_stage0_libtest: Option<bool> = "compiletest-use-stage0-libtest",
        ccache: Option<StringOrBool> = "ccache",
        exclude: Option<Vec<PathBuf>> = "exclude",
    }
}

define_config! {
    /// Configuration specific for some tool, e.g. which features to enable during build.
    #[derive(Default, Clone)]
    struct Tool {
        features: Option<Vec<String>> = "features",
    }
}
