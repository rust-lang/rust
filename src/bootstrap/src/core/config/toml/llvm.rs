//! This module defines the `Llvm` struct, which represents the `[llvm]` table
//! in the `bootstrap.toml` configuration file.

use serde::{Deserialize, Deserializer};

use crate::core::config::StringOrBool;
use crate::core::config::toml::{Merge, ReplaceOpt, TomlConfig};
use crate::{HashMap, HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of how the LLVM build is configured.
    #[derive(Default)]
    struct Llvm {
        optimize: Option<bool> = "optimize",
        thin_lto: Option<bool> = "thin-lto",
        release_debuginfo: Option<bool> = "release-debuginfo",
        assertions: Option<bool> = "assertions",
        tests: Option<bool> = "tests",
        enzyme: Option<bool> = "enzyme",
        plugins: Option<bool> = "plugins",
        static_libstdcpp: Option<bool> = "static-libstdcpp",
        libzstd: Option<bool> = "libzstd",
        ninja: Option<bool> = "ninja",
        targets: Option<String> = "targets",
        experimental_targets: Option<String> = "experimental-targets",
        link_jobs: Option<u32> = "link-jobs",
        link_shared: Option<bool> = "link-shared",
        version_suffix: Option<String> = "version-suffix",
        clang_cl: Option<String> = "clang-cl",
        cflags: Option<String> = "cflags",
        cxxflags: Option<String> = "cxxflags",
        ldflags: Option<String> = "ldflags",
        use_libcxx: Option<bool> = "use-libcxx",
        use_linker: Option<String> = "use-linker",
        allow_old_toolchain: Option<bool> = "allow-old-toolchain",
        offload: Option<bool> = "offload",
        polly: Option<bool> = "polly",
        clang: Option<bool> = "clang",
        enable_warnings: Option<bool> = "enable-warnings",
        download_ci_llvm: Option<StringOrBool> = "download-ci-llvm",
        build_config: Option<HashMap<String, String>> = "build-config",
    }
}

/// Compares the current `Llvm` options against those in the CI LLVM builder and detects any incompatible options.
/// It does this by destructuring the `Llvm` instance to make sure every `Llvm` field is covered and not missing.
#[cfg(not(test))]
pub fn check_incompatible_options_for_ci_llvm(
    current_config_toml: TomlConfig,
    ci_config_toml: TomlConfig,
) -> Result<(), String> {
    macro_rules! err {
        ($current:expr, $expected:expr) => {
            if let Some(current) = &$current {
                if Some(current) != $expected.as_ref() {
                    return Err(format!(
                        "ERROR: Setting `llvm.{}` is incompatible with `llvm.download-ci-llvm`. \
                        Current value: {:?}, Expected value(s): {}{:?}",
                        stringify!($expected).replace("_", "-"),
                        $current,
                        if $expected.is_some() { "None/" } else { "" },
                        $expected,
                    ));
                };
            };
        };
    }

    macro_rules! warn {
        ($current:expr, $expected:expr) => {
            if let Some(current) = &$current {
                if Some(current) != $expected.as_ref() {
                    println!(
                        "WARNING: `llvm.{}` has no effect with `llvm.download-ci-llvm`. \
                        Current value: {:?}, Expected value(s): {}{:?}",
                        stringify!($expected).replace("_", "-"),
                        $current,
                        if $expected.is_some() { "None/" } else { "" },
                        $expected,
                    );
                };
            };
        };
    }

    let (Some(current_llvm_config), Some(ci_llvm_config)) =
        (current_config_toml.llvm, ci_config_toml.llvm)
    else {
        return Ok(());
    };

    let Llvm {
        optimize,
        thin_lto,
        release_debuginfo,
        assertions: _,
        tests: _,
        plugins,
        static_libstdcpp: _,
        libzstd,
        ninja: _,
        targets,
        experimental_targets,
        link_jobs: _,
        link_shared: _,
        version_suffix,
        clang_cl,
        cflags,
        cxxflags,
        ldflags,
        use_libcxx,
        use_linker,
        allow_old_toolchain,
        offload,
        polly,
        clang,
        enable_warnings,
        download_ci_llvm: _,
        build_config,
        enzyme,
    } = ci_llvm_config;

    err!(current_llvm_config.optimize, optimize);
    err!(current_llvm_config.thin_lto, thin_lto);
    err!(current_llvm_config.release_debuginfo, release_debuginfo);
    err!(current_llvm_config.libzstd, libzstd);
    err!(current_llvm_config.targets, targets);
    err!(current_llvm_config.experimental_targets, experimental_targets);
    err!(current_llvm_config.clang_cl, clang_cl);
    err!(current_llvm_config.version_suffix, version_suffix);
    err!(current_llvm_config.cflags, cflags);
    err!(current_llvm_config.cxxflags, cxxflags);
    err!(current_llvm_config.ldflags, ldflags);
    err!(current_llvm_config.use_libcxx, use_libcxx);
    err!(current_llvm_config.use_linker, use_linker);
    err!(current_llvm_config.allow_old_toolchain, allow_old_toolchain);
    err!(current_llvm_config.offload, offload);
    err!(current_llvm_config.polly, polly);
    err!(current_llvm_config.clang, clang);
    err!(current_llvm_config.build_config, build_config);
    err!(current_llvm_config.plugins, plugins);
    err!(current_llvm_config.enzyme, enzyme);

    warn!(current_llvm_config.enable_warnings, enable_warnings);

    Ok(())
}
