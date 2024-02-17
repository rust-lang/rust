//! This module facilitates the tracking system for major changes made to the bootstrap,
//! with the goal of keeping developers synchronized with important modifications in
//! the bootstrap.

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct ChangeInfo {
    /// Represents the ID of PR caused major change on bootstrap.
    pub change_id: usize,
    pub severity: ChangeSeverity,
    /// Provides a short summary of the change that will guide developers
    /// on "how to handle/behave" in response to the changes.
    pub summary: &'static str,
}

#[derive(Clone, Debug)]
pub enum ChangeSeverity {
    /// Used when build configurations continue working as before.
    Info,
    /// Used when the default value of an option changes, or support for an option is removed entirely,
    /// potentially requiring developers to update their build configurations.
    Warning,
}

impl ToString for ChangeSeverity {
    fn to_string(&self) -> String {
        match self {
            ChangeSeverity::Info => "INFO".to_string(),
            ChangeSeverity::Warning => "WARNING".to_string(),
        }
    }
}

pub fn find_recent_config_change_ids(current_id: usize) -> Vec<ChangeInfo> {
    if !CONFIG_CHANGE_HISTORY.iter().any(|config| config.change_id == current_id) {
        // If the current change-id is greater than the most recent one, return
        // an empty list (it may be due to switching from a recent branch to an
        // older one); otherwise, return the full list (assuming the user provided
        // the incorrect change-id by accident).
        if let Some(config) = CONFIG_CHANGE_HISTORY.iter().max_by_key(|config| config.change_id) {
            if &current_id > &config.change_id {
                return Vec::new();
            }
        }

        return CONFIG_CHANGE_HISTORY.to_vec();
    }

    let index =
        CONFIG_CHANGE_HISTORY.iter().position(|config| config.change_id == current_id).unwrap();

    CONFIG_CHANGE_HISTORY
        .iter()
        .skip(index + 1) // Skip the current_id and IDs before it
        .cloned()
        .collect()
}

/// Keeps track of major changes made to the bootstrap configuration.
///
/// If you make any major changes (such as adding new values or changing default values),
/// please ensure adding `ChangeInfo` to the end(because the list must be sorted by the merge date)
/// of this list.
pub const CONFIG_CHANGE_HISTORY: &[ChangeInfo] = &[
    ChangeInfo {
        change_id: 115898,
        severity: ChangeSeverity::Info,
        summary: "Implementation of this change-tracking system. Ignore this.",
    },
    ChangeInfo {
        change_id: 116998,
        severity: ChangeSeverity::Info,
        summary: "Removed android-ndk r15 support in favor of android-ndk r25b.",
    },
    ChangeInfo {
        change_id: 117435,
        severity: ChangeSeverity::Info,
        summary: "New option `rust.parallel-compiler` added to config.toml.",
    },
    ChangeInfo {
        change_id: 116881,
        severity: ChangeSeverity::Warning,
        summary: "Default value of `download-ci-llvm` was changed for `codegen` profile.",
    },
    ChangeInfo {
        change_id: 117813,
        severity: ChangeSeverity::Info,
        summary: "Use of the `if-available` value for `download-ci-llvm` is deprecated; prefer using the new `if-unchanged` value.",
    },
    ChangeInfo {
        change_id: 116278,
        severity: ChangeSeverity::Info,
        summary: "The `rust.use-lld` configuration now has different options ('external'/true or 'self-contained'), and its behaviour has changed.",
    },
    ChangeInfo {
        change_id: 118703,
        severity: ChangeSeverity::Info,
        summary: "Removed rust.run_dsymutil and dist.gpg_password_file config options, as they were unused.",
    },
    ChangeInfo {
        change_id: 119124,
        severity: ChangeSeverity::Warning,
        summary: "rust-analyzer-proc-macro-srv is no longer enabled by default. To build it, you must either enable it in the configuration or explicitly invoke it with x.py.",
    },
    ChangeInfo {
        change_id: 119373,
        severity: ChangeSeverity::Info,
        summary: "The dist.missing-tools config option was deprecated, as it was unused. If you are using it, remove it from your config, it will be removed soon.",
    },
    ChangeInfo {
        change_id: 102579,
        severity: ChangeSeverity::Warning,
        summary: "A new `optimized-compiler-builtins` option has been introduced. Whether to build llvm's `compiler-rt` from source is no longer implicitly controlled by git state. See the PR for more details.",
    },
    ChangeInfo {
        change_id: 120348,
        severity: ChangeSeverity::Info,
        summary: "New option `target.<triple>.codegen-backends` added to config.toml.",
    },
];
