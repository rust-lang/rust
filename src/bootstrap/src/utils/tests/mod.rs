//! This module contains shared utilities for bootstrap tests.

use std::path::{Path, PathBuf};
use std::thread;

use tempfile::TempDir;

use crate::core::builder::Builder;
use crate::core::config::DryRun;
use crate::utils::helpers::get_host_target;
use crate::{Build, Config, Flags, t};

pub mod git;

// Note: tests for `shared_helpers` is separate here, as otherwise shim binaries that include the
// `shared_helpers` via `#[path]` would fail to find it, breaking `./x check bootstrap`.
mod shared_helpers_tests;

/// Holds temporary state of a bootstrap test.
/// Right now it is only used to redirect the build directory of the bootstrap
/// invocation, in the future it would be great if we could actually execute
/// the whole test with this directory set as the workdir.
pub struct TestCtx {
    directory: TempDir,
}

impl TestCtx {
    pub fn new() -> Self {
        let directory = TempDir::new().expect("cannot create temporary directory");
        eprintln!("Running test in {}", directory.path().display());
        Self { directory }
    }

    pub fn dir(&self) -> &Path {
        self.directory.path()
    }

    /// Starts a new invocation of bootstrap that executes `kind` as its top level command
    /// (i.e. `x <kind>`). Returns a builder that configures the created config through CLI flags.
    pub fn config(&self, kind: &str) -> ConfigBuilder {
        ConfigBuilder::from_args(&[kind], self.directory.path().to_owned())
    }
}

/// Used to configure an invocation of bootstrap.
/// Currently runs in the rustc checkout, long-term it should be switched
/// to run in a (cache-primed) temporary directory instead.
pub struct ConfigBuilder {
    args: Vec<String>,
    directory: PathBuf,
    override_download_ci_llvm: bool,
    dry_run: bool,
    explicit_config: bool,
}

impl ConfigBuilder {
    fn from_args(args: &[&str], directory: PathBuf) -> Self {
        Self {
            args: args.iter().copied().map(String::from).collect(),
            directory,
            override_download_ci_llvm: true,
            dry_run: true,
            explicit_config: true,
        }
    }

    pub fn path(mut self, path: &str) -> Self {
        self.arg(path)
    }

    pub fn paths(mut self, paths: &[&str]) -> Self {
        self.args(paths)
    }

    pub fn arg(mut self, arg: &str) -> Self {
        self.args.push(arg.to_string());
        self
    }

    pub fn args(mut self, args: &[&str]) -> Self {
        for arg in args {
            self = self.arg(arg);
        }
        self
    }

    /// Set the specified target to be treated as a no_std target.
    pub fn override_target_no_std(mut self, target: &str) -> Self {
        self.args(&["--set", &format!("target.{target}.no-std=true")])
    }

    pub fn hosts(mut self, targets: &[&str]) -> Self {
        self.args.push("--host".to_string());
        self.args.push(targets.join(","));
        self
    }

    pub fn targets(mut self, targets: &[&str]) -> Self {
        self.args.push("--target".to_string());
        self.args.push(targets.join(","));
        self
    }

    pub fn stage(mut self, stage: u32) -> Self {
        self.args.push("--stage".to_string());
        self.args.push(stage.to_string());
        self
    }

    pub fn with_default_toml_config(mut self, config_toml: &str) -> Self {
        let toml_path = self.directory.join("bootstrap.toml");
        std::fs::write(&toml_path, config_toml).unwrap();
        self.explicit_config = false;
        self.args.push("--config".to_string());
        self.args.push(toml_path.display().to_string());
        self
    }

    pub fn no_override_download_ci_llvm(mut self) -> Self {
        self.override_download_ci_llvm = false;
        self
    }

    pub fn no_dry_run(mut self) -> Self {
        self.dry_run = false;
        self
    }

    pub fn create_config(mut self) -> Config {
        if self.dry_run {
            // Run in dry-check, otherwise the test would be too slow
            self.args.push("--dry-run".to_string());
        }
        // Ignore submodules
        self.args.push("--set".to_string());
        self.args.push("build.submodules=false".to_string());

        if self.override_download_ci_llvm {
            // Override any external LLVM set and inhibit CI LLVM; pretend that we're always building
            // in-tree LLVM from sources.
            self.args.push("--set".to_string());
            self.args.push("llvm.download-ci-llvm=false".to_string());
        }

        // always use the bootstrap toml created in the
        // temporary directory and not from the <src>
        if self.explicit_config {
            self = self.with_default_toml_config("");
        }

        // Do not mess with the local rustc checkout build directory
        self.args.push("--build-dir".to_string());
        self.args.push(self.directory.join("build").display().to_string());

        Config::parse(Flags::parse(&self.args))
    }
}
