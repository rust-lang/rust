//! Handles the vendoring process for the bootstrap system.
//!
//! This module ensures that all required Cargo dependencies are gathered
//! and stored in the `<src>/<VENDOR_DIR>` directory.
use std::path::PathBuf;

use crate::core::build_steps::tool::SUBMODULES_FOR_RUSTBOOK;
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::utils::exec::command;

/// The name of the directory where vendored dependencies are stored.
pub const VENDOR_DIR: &str = "vendor";

/// Returns the cargo workspaces to vendor for `x vendor` and dist tarballs.
///
/// Returns a `Vec` of `(path_to_manifest, submodules_required)` where
/// `path_to_manifest` is the cargo workspace, and `submodules_required` is
/// the set of submodules that must be available.
pub fn default_paths_to_vendor(builder: &Builder<'_>) -> Vec<(PathBuf, Vec<&'static str>)> {
    [
        ("src/tools/cargo/Cargo.toml", vec!["src/tools/cargo"]),
        ("src/tools/clippy/clippy_test_deps/Cargo.toml", vec![]),
        ("src/tools/rust-analyzer/Cargo.toml", vec![]),
        ("compiler/rustc_codegen_cranelift/Cargo.toml", vec![]),
        ("compiler/rustc_codegen_gcc/Cargo.toml", vec![]),
        ("library/Cargo.toml", vec![]),
        ("src/bootstrap/Cargo.toml", vec![]),
        ("src/tools/rustbook/Cargo.toml", SUBMODULES_FOR_RUSTBOOK.into()),
        ("src/tools/rustc-perf/Cargo.toml", vec!["src/tools/rustc-perf"]),
        ("src/tools/opt-dist/Cargo.toml", vec![]),
        ("src/doc/book/packages/trpl/Cargo.toml", vec![]),
    ]
    .into_iter()
    .map(|(path, submodules)| (builder.src.join(path), submodules))
    .collect()
}

/// Defines the vendoring step in the bootstrap process.
///
/// This step executes `cargo vendor` to collect all dependencies
/// and store them in the `<src>/<VENDOR_DIR>` directory.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct Vendor {
    /// Additional paths to synchronize during vendoring.
    pub(crate) sync_args: Vec<PathBuf>,
    /// Determines whether vendored dependencies use versioned directories.
    pub(crate) versioned_dirs: bool,
    /// The root directory of the source code.
    pub(crate) root_dir: PathBuf,
    /// The target directory for storing vendored dependencies.
    pub(crate) output_dir: PathBuf,
}

impl Step for Vendor {
    type Output = VendorOutput;
    const DEFAULT: bool = true;
    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("placeholder").default_condition(true)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Vendor {
            sync_args: run.builder.config.cmd.vendor_sync_args(),
            versioned_dirs: run.builder.config.cmd.vendor_versioned_dirs(),
            root_dir: run.builder.src.clone(),
            output_dir: run.builder.src.join(VENDOR_DIR),
        });
    }

    /// Executes the vendoring process.
    ///
    /// This function runs `cargo vendor` and ensures all required submodules
    /// are initialized before vendoring begins.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        builder.info(&format!("Vendoring sources to {:?}", self.root_dir));

        let mut cmd = command(&builder.initial_cargo);
        cmd.arg("vendor");

        if self.versioned_dirs {
            cmd.arg("--versioned-dirs");
        }

        let to_vendor = default_paths_to_vendor(builder);
        // These submodules must be present for `x vendor` to work.
        for (_, submodules) in &to_vendor {
            for submodule in submodules {
                builder.build.require_submodule(submodule, None);
            }
        }

        // Sync these paths by default.
        for (p, _) in &to_vendor {
            cmd.arg("--sync").arg(p);
        }

        // Also sync explicitly requested paths.
        for sync_arg in self.sync_args {
            cmd.arg("--sync").arg(sync_arg);
        }

        // Will read the libstd Cargo.toml
        // which uses the unstable `public-dependency` feature.
        cmd.env("RUSTC_BOOTSTRAP", "1");
        cmd.env("RUSTC", &builder.initial_rustc);

        cmd.current_dir(self.root_dir).arg(&self.output_dir);

        let config = cmd.run_capture_stdout(builder);
        VendorOutput { config: config.stdout() }
    }
}

/// Stores the result of the vendoring step.
#[derive(Debug, Clone)]
pub(crate) struct VendorOutput {
    pub(crate) config: String,
}
