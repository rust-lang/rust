//! This module defines the core, processed, and operational structures that represent
//! the final bootstrap configuration.

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
