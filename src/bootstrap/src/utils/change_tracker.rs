//! This module facilitates the tracking system for major changes made to the bootstrap,
//! with the goal of keeping developers synchronized with important modifications in
//! the bootstrap.

use std::fmt::Display;

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

impl Display for ChangeSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeSeverity::Info => write!(f, "INFO"),
            ChangeSeverity::Warning => write!(f, "WARNING"),
        }
    }
}

pub fn find_recent_config_change_ids(current_id: usize) -> &'static [ChangeInfo] {
    if let Some(index) =
        CONFIG_CHANGE_HISTORY.iter().position(|config| config.change_id == current_id)
    {
        // Skip the current_id and IDs before it
        &CONFIG_CHANGE_HISTORY[index + 1..]
    } else {
        // If the current change-id is greater than the most recent one, return
        // an empty list (it may be due to switching from a recent branch to an
        // older one); otherwise, return the full list (assuming the user provided
        // the incorrect change-id by accident).
        if let Some(config) = CONFIG_CHANGE_HISTORY.iter().max_by_key(|config| config.change_id) {
            if current_id > config.change_id {
                return &[];
            }
        }

        CONFIG_CHANGE_HISTORY
    }
}

pub fn human_readable_changes(changes: &[ChangeInfo]) -> String {
    let mut message = String::new();

    for change in changes {
        message.push_str(&format!("  [{}] {}\n", change.severity, change.summary));
        message.push_str(&format!(
            "    - PR Link https://github.com/rust-lang/rust/pull/{}\n",
            change.change_id
        ));
    }

    message
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
    ChangeInfo {
        change_id: 121203,
        severity: ChangeSeverity::Info,
        summary: "A new `rust.frame-pointers` option has been introduced and made the default in the compiler and codegen profiles.",
    },
    ChangeInfo {
        change_id: 121278,
        severity: ChangeSeverity::Warning,
        summary: "The \"codegen\"/\"llvm\" profile has been removed and replaced with \"compiler\", use it instead for the same behavior.",
    },
    ChangeInfo {
        change_id: 118724,
        severity: ChangeSeverity::Info,
        summary: "`x install` now skips providing tarball sources (under 'build/dist' path) to speed up the installation process.",
    },
    ChangeInfo {
        change_id: 121976,
        severity: ChangeSeverity::Info,
        summary: "A new `boostrap-cache-path` option has been introduced which can be utilized to modify the cache path for bootstrap.",
    },
    ChangeInfo {
        change_id: 122108,
        severity: ChangeSeverity::Info,
        summary: "a new `target.*.runner` option is available to specify a wrapper executable required to run tests for a target",
    },
    ChangeInfo {
        change_id: 117458,
        severity: ChangeSeverity::Info,
        summary: "New option `rust.llvm-bitcode-linker` that will build the llvm-bitcode-linker.",
    },
    ChangeInfo {
        change_id: 121754,
        severity: ChangeSeverity::Warning,
        summary: "`rust.split-debuginfo` has been moved to `target.<triple>.split-debuginfo` and its default value is determined for each target individually.",
    },
    ChangeInfo {
        change_id: 123711,
        severity: ChangeSeverity::Warning,
        summary: "The deprecated field `changelog-seen` has been removed. Using that field in `config.toml` from now on will result in breakage.",
    },
    ChangeInfo {
        change_id: 124501,
        severity: ChangeSeverity::Info,
        summary: "New option `build.lldb` that will override the default lldb binary path used in debuginfo tests",
    },
    ChangeInfo {
        change_id: 123337,
        severity: ChangeSeverity::Info,
        summary: r#"The compiler profile now defaults to rust.debuginfo-level = "line-tables-only""#,
    },
    ChangeInfo {
        change_id: 124129,
        severity: ChangeSeverity::Warning,
        summary: "`rust.lld` has a new default value of `true` on `x86_64-unknown-linux-gnu`. Starting at stage1, `rust-lld` will thus be this target's default linker. No config changes should be necessary.",
    },
    ChangeInfo {
        change_id: 125535,
        severity: ChangeSeverity::Warning,
        summary: "Removed `dist.missing-tools` configuration as it was deprecated long time ago.",
    },
    ChangeInfo {
        change_id: 126701,
        severity: ChangeSeverity::Warning,
        summary: "`llvm.lld` is enabled by default for the dist profile. If set to false, `lld` will not be included in the dist build.",
    },
    ChangeInfo {
        change_id: 127913,
        severity: ChangeSeverity::Warning,
        summary: "`debug-logging` option has been removed from the default `tools` profile.",
    },
    ChangeInfo {
        change_id: 127866,
        severity: ChangeSeverity::Info,
        summary: "the `wasm-component-ld` tool is now built as part of `build.extended` and can be a member of `build.tools`",
    },
    ChangeInfo {
        change_id: 120593,
        severity: ChangeSeverity::Info,
        summary: "Removed android-ndk r25b support in favor of android-ndk r26d.",
    },
    ChangeInfo {
        change_id: 125181,
        severity: ChangeSeverity::Warning,
        summary: "For tarball sources, default value for `rust.channel` will be taken from `src/ci/channel` file.",
    },
    ChangeInfo {
        change_id: 125642,
        severity: ChangeSeverity::Info,
        summary: "New option `llvm.libzstd` to control whether llvm is built with zstd support.",
    },
    ChangeInfo {
        change_id: 128841,
        severity: ChangeSeverity::Warning,
        summary: "./x test --rustc-args was renamed to --compiletest-rustc-args as it only applies there. ./x miri --rustc-args was also removed.",
    },
    ChangeInfo {
        change_id: 129295,
        severity: ChangeSeverity::Info,
        summary: "The `build.profiler` option now tries to use source code from `download-ci-llvm` if possible, instead of checking out the `src/llvm-project` submodule.",
    },
    ChangeInfo {
        change_id: 129152,
        severity: ChangeSeverity::Info,
        summary: "New option `build.cargo-clippy` added for supporting the use of custom/external clippy.",
    },
    ChangeInfo {
        change_id: 129925,
        severity: ChangeSeverity::Warning,
        summary: "Removed `rust.split-debuginfo` as it was deprecated long time ago.",
    },
    ChangeInfo {
        change_id: 129176,
        severity: ChangeSeverity::Info,
        summary: "New option `llvm.enzyme` to control whether the llvm based autodiff tool (Enzyme) is built.",
    },
    ChangeInfo {
        change_id: 129473,
        severity: ChangeSeverity::Warning,
        summary: "`download-ci-llvm = true` now checks if CI llvm is available and has become the default for the compiler profile",
    },
    ChangeInfo {
        change_id: 130202,
        severity: ChangeSeverity::Info,
        summary: "'tools' and 'library' profiles switched `download-ci-llvm` option from `if-unchanged` to `true`.",
    },
    ChangeInfo {
        change_id: 130110,
        severity: ChangeSeverity::Info,
        summary: "New option `dist.vendor` added to control whether bootstrap should vendor dependencies for dist tarball.",
    },
    ChangeInfo {
        change_id: 130529,
        severity: ChangeSeverity::Info,
        summary: "If `llvm.download-ci-llvm` is not defined, it defaults to `true`.",
    },
    ChangeInfo {
        change_id: 131075,
        severity: ChangeSeverity::Info,
        summary: "New option `./x setup editor` added, replacing `./x setup vscode` and adding support for vim, emacs and helix.",
    },
    ChangeInfo {
        change_id: 131838,
        severity: ChangeSeverity::Info,
        summary: "Allow setting `--jobs` in config.toml with `build.jobs`.",
    },
    ChangeInfo {
        change_id: 131181,
        severity: ChangeSeverity::Info,
        summary: "New option `build.compiletest-diff-tool` that adds support for a custom differ for compiletest",
    },
    ChangeInfo {
        change_id: 131513,
        severity: ChangeSeverity::Info,
        summary: "New option `llvm.offload` to control whether the llvm offload runtime for GPU support is built. Implicitly enables the openmp runtime as dependency.",
    },
    ChangeInfo {
        change_id: 132282,
        severity: ChangeSeverity::Warning,
        summary: "Deprecated `rust.parallel_compiler` as the compiler now always defaults to being parallel (with 1 thread)",
    },
    ChangeInfo {
        change_id: 132494,
        severity: ChangeSeverity::Info,
        summary: "`download-rustc='if-unchanged'` is now a default option for library profile.",
    },
    ChangeInfo {
        change_id: 133207,
        severity: ChangeSeverity::Info,
        summary: "`rust.llvm-tools` is now enabled by default when no `config.toml` is provided.",
    },
    ChangeInfo {
        change_id: 133068,
        severity: ChangeSeverity::Warning,
        summary: "Revert `rust.download-rustc` global default to `false` and only use `rust.download-rustc = \"if-unchanged\"` default for library and tools profile. As alt CI rustc is built without debug assertions, `rust.debug-assertions = true` will now inhibit downloading CI rustc.",
    },
    ChangeInfo {
        change_id: 133853,
        severity: ChangeSeverity::Info,
        summary: "`build.vendor` is now enabled by default for dist/tarball sources when 'vendor' directory and '.cargo/config.toml' file are present.",
    },
    ChangeInfo {
        change_id: 134809,
        severity: ChangeSeverity::Warning,
        summary: "compiletest now takes `--no-capture` instead of `--nocapture`; bootstrap now accepts `--no-capture` as an argument to test commands directly",
    },
    ChangeInfo {
        change_id: 134650,
        severity: ChangeSeverity::Warning,
        summary: "Removed `rust.parallel-compiler` as it was deprecated in #132282 long time ago.",
    },
    ChangeInfo {
        change_id: 135326,
        severity: ChangeSeverity::Warning,
        summary: "It is now possible to configure `optimized-compiler-builtins` for per target.",
    },
    ChangeInfo {
        change_id: 135281,
        severity: ChangeSeverity::Warning,
        summary: "Some stamp names in the build artifacts may have changed slightly (e.g., from `llvm-finished-building` to `.llvm-stamp`).",
    },
    ChangeInfo {
        change_id: 135729,
        severity: ChangeSeverity::Info,
        summary: "Change the compiler profile to default to rust.debug-assertions = true",
    },
    ChangeInfo {
        change_id: 135832,
        severity: ChangeSeverity::Info,
        summary: "Rustdoc now respects the value of rust.lto.",
    },
    ChangeInfo {
        change_id: 136941,
        severity: ChangeSeverity::Info,
        summary: "The llvm.ccache option has moved to build.ccache. llvm.ccache is now deprecated.",
    },
    ChangeInfo {
        change_id: 137170,
        severity: ChangeSeverity::Info,
        summary: "It is now possible to configure `jemalloc` for each target",
    },
    ChangeInfo {
        change_id: 137215,
        severity: ChangeSeverity::Info,
        summary: "Added `build.test-stage = 2` to 'tools' profile defaults",
    },
    ChangeInfo {
        change_id: 137220,
        severity: ChangeSeverity::Info,
        summary: "`rust.channel` now supports \"auto-detect\" to load the channel from `src/ci/channel`",
    },
    ChangeInfo {
        change_id: 137723,
        severity: ChangeSeverity::Info,
        summary: "The rust.description option has moved to build.description and rust.description is now deprecated.",
    },
    ChangeInfo {
        change_id: 138051,
        severity: ChangeSeverity::Info,
        summary: "There is now a new `gcc` config section that can be used to download GCC from CI using `gcc.download-ci-gcc = true`",
    },
    ChangeInfo {
        change_id: 126856,
        severity: ChangeSeverity::Warning,
        summary: "Removed `src/tools/rls` tool as it was deprecated long time ago.",
    },
    ChangeInfo {
        change_id: 137147,
        severity: ChangeSeverity::Info,
        summary: "Added new option `build.exclude` which works the same way as `--exclude` flag on `x`.",
    },
    ChangeInfo {
        change_id: 137081,
        severity: ChangeSeverity::Warning,
        summary: "The default configuration filename has changed from `config.toml` to `bootstrap.toml`. `config.toml` is deprecated but remains supported for backward compatibility.",
    },
    ChangeInfo {
        change_id: 138986,
        severity: ChangeSeverity::Info,
        summary: "You can now use `change-id = \"ignore\"` to suppress `change-id ` warnings in the console.",
    },
];
