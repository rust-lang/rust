//! This module defines common types, enums, and helper functions used across
//! various TOML configuration sections within `bootstrap.toml`.
//!
//! It provides shared definitions for:
//! - Data types that are serialized from TOML.
//! - Utility enums for specific configuration options.
//! - Helper functions for managing configuration values.

use std::fmt::Display;

use serde::{Deserialize, Deserializer};
use serde_derive::Deserialize;

use crate::str::FromStr;

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

/// Describes how to handle conflicts in merging two [`TomlConfig`]
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
