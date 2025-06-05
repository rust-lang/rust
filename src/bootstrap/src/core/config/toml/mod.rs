//! This module defines the structures that directly mirror the `bootstrap.toml`
//! file's format. These types are used for `serde` deserialization.
//!
//! Crucially, this module also houses the core logic for loading, parsing, and merging
//! these raw TOML configurations from various sources (the main `bootstrap.toml`,
//! included files, profile defaults, and command-line overrides). This processed
//! TOML data then serves as an intermediate representation, which is further
//! transformed and applied to the final [`Config`] struct.

use serde::Deserialize;
use serde_derive::Deserialize;
pub mod build;
pub mod change_id;
pub mod dist;
pub mod gcc;
pub mod install;
pub mod llvm;
pub mod rust;
pub mod target;

use build::Build;
use change_id::{ChangeId, ChangeIdWrapper};
use dist::Dist;
use gcc::Gcc;
use install::Install;
use llvm::Llvm;
use rust::Rust;
use target::TomlTarget;

use crate::core::config::{Merge, ReplaceOpt};
use crate::{Config, HashMap, HashSet, Path, PathBuf, exit, fs, t};

/// Structure of the `bootstrap.toml` file that configuration is read from.
///
/// This structure uses `Decodable` to automatically decode a TOML configuration
/// file into this format, and then this is traversed and written into the above
/// `Config` structure.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub(crate) struct TomlConfig {
    #[serde(flatten)]
    pub(crate) change_id: ChangeIdWrapper,
    pub(super) build: Option<Build>,
    pub(super) install: Option<Install>,
    pub(super) llvm: Option<Llvm>,
    pub(super) gcc: Option<Gcc>,
    pub(super) rust: Option<Rust>,
    pub(super) target: Option<HashMap<String, TomlTarget>>,
    pub(super) dist: Option<Dist>,
    pub(super) profile: Option<String>,
    pub(super) include: Option<Vec<PathBuf>>,
}

impl Merge for TomlConfig {
    fn merge(
        &mut self,
        parent_config_path: Option<PathBuf>,
        included_extensions: &mut HashSet<PathBuf>,
        TomlConfig { build, install, llvm, gcc, rust, dist, target, profile, change_id, include }: Self,
        replace: ReplaceOpt,
    ) {
        fn do_merge<T: Merge>(x: &mut Option<T>, y: Option<T>, replace: ReplaceOpt) {
            if let Some(new) = y {
                if let Some(original) = x {
                    original.merge(None, &mut Default::default(), new, replace);
                } else {
                    *x = Some(new);
                }
            }
        }

        self.change_id.inner.merge(None, &mut Default::default(), change_id.inner, replace);
        self.profile.merge(None, &mut Default::default(), profile, replace);

        do_merge(&mut self.build, build, replace);
        do_merge(&mut self.install, install, replace);
        do_merge(&mut self.llvm, llvm, replace);
        do_merge(&mut self.gcc, gcc, replace);
        do_merge(&mut self.rust, rust, replace);
        do_merge(&mut self.dist, dist, replace);

        match (self.target.as_mut(), target) {
            (_, None) => {}
            (None, Some(target)) => self.target = Some(target),
            (Some(original_target), Some(new_target)) => {
                for (triple, new) in new_target {
                    if let Some(original) = original_target.get_mut(&triple) {
                        original.merge(None, &mut Default::default(), new, replace);
                    } else {
                        original_target.insert(triple, new);
                    }
                }
            }
        }

        let parent_dir = parent_config_path
            .as_ref()
            .and_then(|p| p.parent().map(ToOwned::to_owned))
            .unwrap_or_default();

        // `include` handled later since we ignore duplicates using `ReplaceOpt::IgnoreDuplicate` to
        // keep the upper-level configuration to take precedence.
        for include_path in include.clone().unwrap_or_default().iter().rev() {
            let include_path = parent_dir.join(include_path);
            let include_path = include_path.canonicalize().unwrap_or_else(|e| {
                eprintln!("ERROR: Failed to canonicalize '{}' path: {e}", include_path.display());
                exit!(2);
            });

            let included_toml = Config::get_toml_inner(&include_path).unwrap_or_else(|e| {
                eprintln!("ERROR: Failed to parse '{}': {e}", include_path.display());
                exit!(2);
            });

            assert!(
                included_extensions.insert(include_path.clone()),
                "Cyclic inclusion detected: '{}' is being included again before its previous inclusion was fully processed.",
                include_path.display()
            );

            self.merge(
                Some(include_path.clone()),
                included_extensions,
                included_toml,
                // Ensures that parent configuration always takes precedence
                // over child configurations.
                ReplaceOpt::IgnoreDuplicate,
            );

            included_extensions.remove(&include_path);
        }
    }
}

/// This file is embedded in the overlay directory of the tarball sources. It is
/// useful in scenarios where developers want to see how the tarball sources were
/// generated.
///
/// We also use this file to compare the host's bootstrap.toml against the CI rustc builder
/// configuration to detect any incompatible options.
pub const BUILDER_CONFIG_FILENAME: &str = "builder-config";

impl Config {
    pub(crate) fn get_builder_toml(&self, build_name: &str) -> Result<TomlConfig, toml::de::Error> {
        if self.dry_run() {
            return Ok(TomlConfig::default());
        }

        let builder_config_path =
            self.out.join(self.build.triple).join(build_name).join(BUILDER_CONFIG_FILENAME);
        Self::get_toml(&builder_config_path)
    }

    pub(crate) fn get_toml(file: &Path) -> Result<TomlConfig, toml::de::Error> {
        #[cfg(test)]
        return Ok(TomlConfig::default());

        #[cfg(not(test))]
        Self::get_toml_inner(file)
    }

    pub(crate) fn get_toml_inner(file: &Path) -> Result<TomlConfig, toml::de::Error> {
        let contents =
            t!(fs::read_to_string(file), format!("config file {} not found", file.display()));
        // Deserialize to Value and then TomlConfig to prevent the Deserialize impl of
        // TomlConfig and sub types to be monomorphized 5x by toml.
        toml::from_str(&contents)
            .and_then(|table: toml::Value| TomlConfig::deserialize(table))
            .inspect_err(|_| {
                if let Ok(ChangeIdWrapper { inner: Some(ChangeId::Id(id)) }) =
                    toml::from_str::<toml::Value>(&contents)
                        .and_then(|table: toml::Value| ChangeIdWrapper::deserialize(table))
                {
                    let changes = crate::find_recent_config_change_ids(id);
                    if !changes.is_empty() {
                        println!(
                            "WARNING: There have been changes to x.py since you last updated:\n{}",
                            crate::human_readable_changes(changes)
                        );
                    }
                }
            })
    }
}
