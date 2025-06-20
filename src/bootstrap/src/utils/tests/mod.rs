//! This module contains shared utilities for bootstrap tests.

use std::path::{Path, PathBuf};
use std::thread;

use tempfile::TempDir;

use crate::core::builder::Builder;
use crate::core::config::DryRun;
use crate::{Build, Config, Flags, t};

pub mod git;

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
}

impl ConfigBuilder {
    fn from_args(args: &[&str], directory: PathBuf) -> Self {
        Self { args: args.iter().copied().map(String::from).collect(), directory }
    }

    pub fn path(mut self, path: &str) -> Self {
        self.args.push(path.to_string());
        self
    }

    pub fn stage(mut self, stage: u32) -> Self {
        self.args.push("--stage".to_string());
        self.args.push(stage.to_string());
        self
    }

    pub fn create_config(mut self) -> Config {
        // Run in dry-check, otherwise the test would be too slow
        self.args.push("--dry-run".to_string());

        // Ignore submodules
        self.args.push("--set".to_string());
        self.args.push("build.submodules=false".to_string());

        // Do not mess with the local rustc checkout build directory
        self.args.push("--build-dir".to_string());
        self.args.push(self.directory.join("build").display().to_string());

        Config::parse(Flags::parse(&self.args))
    }
}
