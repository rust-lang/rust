//! This module contains shared utilities for bootstrap tests.

use crate::core::builder::Builder;
use crate::core::config::DryRun;
use crate::{Build, Config, Flags};

pub mod git;

/// Used to configure an invocation of bootstrap.
/// Currently runs in the rustc checkout, long-term it should be switched
/// to run in a (cache-primed) temporary directory instead.
pub struct ConfigBuilder {
    args: Vec<String>,
}

impl ConfigBuilder {
    pub fn from_args(args: &[&str]) -> Self {
        Self::new(args)
    }

    pub fn build() -> Self {
        Self::new(&["build"])
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

    fn new(args: &[&str]) -> Self {
        Self { args: args.iter().copied().map(String::from).collect() }
    }

    pub fn create_config(mut self) -> Config {
        let mut config = Config::parse(Flags::parse(&self.args));
        // Run in dry-check, otherwise the test would be too slow
        config.set_dry_run(DryRun::SelfCheck);
        config
    }
}
