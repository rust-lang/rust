//! Entry point for the `config` module.
//!
//! This module defines two macros:
//!
//! - `define_config!`: A declarative macro used instead of `#[derive(Deserialize)]` to reduce
//!   compile time and binary size, especially for the bootstrap binary.
//!
//! - `check_ci_llvm!`: A compile-time assertion macro that ensures certain settings are
//!   not enabled when `download-ci-llvm` is active.
//!
//! A declarative macro is used here in place of a procedural derive macro to minimize
//! the compile time of the bootstrap process.
//!
//! Additionally, this module defines common types, enums, and helper functions used across
//! various TOML configuration sections in `bootstrap.toml`.
//!
//! It provides shared definitions for:
//! - Data types deserialized from TOML.
//! - Utility enums for specific configuration options.
//! - Helper functions for managing configuration values.

#[expect(clippy::module_inception)]
mod config;
pub mod flags;
pub mod target_selection;
#[cfg(test)]
mod tests;
pub mod toml;

use std::collections::HashSet;
use std::path::PathBuf;

use build_helper::exit;
pub use config::*;
use serde::{Deserialize, Deserializer};
use serde_derive::Deserialize;
pub use target_selection::TargetSelection;
pub use toml::BUILDER_CONFIG_FILENAME;
pub use toml::change_id::ChangeId;
pub use toml::rust::LldMode;
pub use toml::target::Target;
#[cfg(feature = "tracing")]
use tracing::{instrument, span};

use crate::Display;
use crate::str::FromStr;

// We are using a decl macro instead of a derive proc macro here to reduce the compile time of bootstrap.
#[macro_export]
macro_rules! define_config {
    ($(#[$attr:meta])* struct $name:ident {
        $($field:ident: Option<$field_ty:ty> = $field_key:literal,)*
    }) => {
        $(#[$attr])*
        pub struct $name {
            $(pub $field: Option<$field_ty>,)*
        }

        impl Merge for $name {
            fn merge(
                &mut self,
                _parent_config_path: Option<PathBuf>,
                _included_extensions: &mut HashSet<PathBuf>,
                other: Self,
                replace: ReplaceOpt
            ) {
                $(
                    match replace {
                        ReplaceOpt::IgnoreDuplicate => {
                            if self.$field.is_none() {
                                self.$field = other.$field;
                            }
                        },
                        ReplaceOpt::Override => {
                            if other.$field.is_some() {
                                self.$field = other.$field;
                            }
                        }
                        ReplaceOpt::ErrorOnDuplicate => {
                            if other.$field.is_some() {
                                if self.$field.is_some() {
                                    if cfg!(test) {
                                        panic!("overriding existing option")
                                    } else {
                                        eprintln!("overriding existing option: `{}`", stringify!($field));
                                        exit!(2);
                                    }
                                } else {
                                    self.$field = other.$field;
                                }
                            }
                        }
                    }
                )*
            }
        }

        // The following is a trimmed version of what serde_derive generates. All parts not relevant
        // for toml deserialization have been removed. This reduces the binary size and improves
        // compile time of bootstrap.
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct Field;
                impl<'de> serde::de::Visitor<'de> for Field {
                    type Value = $name;
                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str(concat!("struct ", stringify!($name)))
                    }

                    #[inline]
                    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::MapAccess<'de>,
                    {
                        $(let mut $field: Option<$field_ty> = None;)*
                        while let Some(key) =
                            match serde::de::MapAccess::next_key::<String>(&mut map) {
                                Ok(val) => val,
                                Err(err) => {
                                    return Err(err);
                                }
                            }
                        {
                            match &*key {
                                $($field_key => {
                                    if $field.is_some() {
                                        return Err(<A::Error as serde::de::Error>::duplicate_field(
                                            $field_key,
                                        ));
                                    }
                                    $field = match serde::de::MapAccess::next_value::<$field_ty>(
                                        &mut map,
                                    ) {
                                        Ok(val) => Some(val),
                                        Err(err) => {
                                            return Err(err);
                                        }
                                    };
                                })*
                                key => {
                                    return Err(serde::de::Error::unknown_field(key, FIELDS));
                                }
                            }
                        }
                        Ok($name { $($field),* })
                    }
                }
                const FIELDS: &'static [&'static str] = &[
                    $($field_key,)*
                ];
                Deserializer::deserialize_struct(
                    deserializer,
                    stringify!($name),
                    FIELDS,
                    Field,
                )
            }
        }
    }
}

#[macro_export]
macro_rules! check_ci_llvm {
    ($name:expr) => {
        assert!(
            $name.is_none(),
            "setting {} is incompatible with download-ci-llvm.",
            stringify!($name).replace("_", "-")
        );
    };
}

pub(crate) trait Merge {
    fn merge(
        &mut self,
        parent_config_path: Option<PathBuf>,
        included_extensions: &mut HashSet<PathBuf>,
        other: Self,
        replace: ReplaceOpt,
    );
}

impl<T> Merge for Option<T> {
    fn merge(
        &mut self,
        _parent_config_path: Option<PathBuf>,
        _included_extensions: &mut HashSet<PathBuf>,
        other: Self,
        replace: ReplaceOpt,
    ) {
        match replace {
            ReplaceOpt::IgnoreDuplicate => {
                if self.is_none() {
                    *self = other;
                }
            }
            ReplaceOpt::Override => {
                if other.is_some() {
                    *self = other;
                }
            }
            ReplaceOpt::ErrorOnDuplicate => {
                if other.is_some() {
                    if self.is_some() {
                        if cfg!(test) {
                            panic!("overriding existing option")
                        } else {
                            eprintln!("overriding existing option");
                            exit!(2);
                        }
                    } else {
                        *self = other;
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum DebuginfoLevel {
    #[default]
    None,
    LineDirectivesOnly,
    LineTablesOnly,
    Limited,
    Full,
}

// NOTE: can't derive(Deserialize) because the intermediate trip through toml::Value only
// deserializes i64, and derive() only generates visit_u64
impl<'de> Deserialize<'de> for DebuginfoLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        Ok(match Deserialize::deserialize(deserializer)? {
            StringOrInt::String(s) if s == "none" => DebuginfoLevel::None,
            StringOrInt::Int(0) => DebuginfoLevel::None,
            StringOrInt::String(s) if s == "line-directives-only" => {
                DebuginfoLevel::LineDirectivesOnly
            }
            StringOrInt::String(s) if s == "line-tables-only" => DebuginfoLevel::LineTablesOnly,
            StringOrInt::String(s) if s == "limited" => DebuginfoLevel::Limited,
            StringOrInt::Int(1) => DebuginfoLevel::Limited,
            StringOrInt::String(s) if s == "full" => DebuginfoLevel::Full,
            StringOrInt::Int(2) => DebuginfoLevel::Full,
            StringOrInt::Int(n) => {
                let other = serde::de::Unexpected::Signed(n);
                return Err(D::Error::invalid_value(other, &"expected 0, 1, or 2"));
            }
            StringOrInt::String(s) => {
                let other = serde::de::Unexpected::Str(&s);
                return Err(D::Error::invalid_value(
                    other,
                    &"expected none, line-tables-only, limited, or full",
                ));
            }
        })
    }
}

/// Suitable for passing to `-C debuginfo`
impl Display for DebuginfoLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use DebuginfoLevel::*;
        f.write_str(match self {
            None => "0",
            LineDirectivesOnly => "line-directives-only",
            LineTablesOnly => "line-tables-only",
            Limited => "1",
            Full => "2",
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum StringOrBool {
    String(String),
    Bool(bool),
}

impl Default for StringOrBool {
    fn default() -> StringOrBool {
        StringOrBool::Bool(false)
    }
}

impl StringOrBool {
    pub fn is_string_or_true(&self) -> bool {
        matches!(self, Self::String(_) | Self::Bool(true))
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StringOrInt {
    String(String),
    Int(i64),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LlvmLibunwind {
    #[default]
    No,
    InTree,
    System,
}

impl FromStr for LlvmLibunwind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "no" => Ok(Self::No),
            "in-tree" => Ok(Self::InTree),
            "system" => Ok(Self::System),
            invalid => Err(format!("Invalid value '{invalid}' for rust.llvm-libunwind config.")),
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SplitDebuginfo {
    Packed,
    Unpacked,
    #[default]
    Off,
}

impl std::str::FromStr for SplitDebuginfo {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "packed" => Ok(SplitDebuginfo::Packed),
            "unpacked" => Ok(SplitDebuginfo::Unpacked),
            "off" => Ok(SplitDebuginfo::Off),
            _ => Err(()),
        }
    }
}

/// Describes how to handle conflicts in merging two `TomlConfig`
#[derive(Copy, Clone, Debug)]
pub enum ReplaceOpt {
    /// Silently ignore a duplicated value
    IgnoreDuplicate,
    /// Override the current value, even if it's `Some`
    Override,
    /// Exit with an error on duplicate values
    ErrorOnDuplicate,
}

#[derive(Clone, Default)]
pub enum DryRun {
    /// This isn't a dry run.
    #[default]
    Disabled,
    /// This is a dry run enabled by bootstrap itself, so it can verify that no work is done.
    SelfCheck,
    /// This is a dry run enabled by the `--dry-run` flag.
    UserSelected,
}

/// LTO mode used for compiling rustc itself.
#[derive(Default, Clone, PartialEq, Debug)]
pub enum RustcLto {
    Off,
    #[default]
    ThinLocal,
    Thin,
    Fat,
}

impl std::str::FromStr for RustcLto {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "thin-local" => Ok(RustcLto::ThinLocal),
            "thin" => Ok(RustcLto::Thin),
            "fat" => Ok(RustcLto::Fat),
            "off" => Ok(RustcLto::Off),
            _ => Err(format!("Invalid value for rustc LTO: {s}")),
        }
    }
}

/// Determines how will GCC be provided.
#[derive(Default, Clone)]
pub enum GccCiMode {
    /// Build GCC from the local `src/gcc` submodule.
    #[default]
    BuildLocally,
    /// Try to download GCC from CI.
    /// If it is not available on CI, it will be built locally instead.
    DownloadFromCi,
}

pub fn set<T>(field: &mut T, val: Option<T>) {
    if let Some(v) = val {
        *field = v;
    }
}

pub fn threads_from_config(v: u32) -> u32 {
    match v {
        0 => std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get) as u32,
        n => n,
    }
}
