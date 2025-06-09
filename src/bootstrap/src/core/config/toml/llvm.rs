//! This module defines the `Llvm` struct, which represents the `[llvm]` table
//! in the `bootstrap.toml` configuration file.

use serde::{Deserialize, Deserializer};

use crate::core::config::toml::{Merge, ReplaceOpt, TomlConfig};
use crate::core::config::{StringOrBool, set};
use crate::{Config, HashMap, HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of how the LLVM build is configured.
    struct Llvm {
        optimize: Option<bool> = "optimize",
        thin_lto: Option<bool> = "thin-lto",
        release_debuginfo: Option<bool> = "release-debuginfo",
        assertions: Option<bool> = "assertions",
        tests: Option<bool> = "tests",
        enzyme: Option<bool> = "enzyme",
        plugins: Option<bool> = "plugins",
        // FIXME: Remove this field at Q2 2025, it has been replaced by build.ccache
        ccache: Option<StringOrBool> = "ccache",
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
        ccache: _,
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

impl Config {
    pub fn apply_llvm_config(
        &mut self,
        toml_llvm: Option<Llvm>,
        ccache: &mut Option<StringOrBool>,
    ) {
        let mut llvm_tests = None;
        let mut llvm_enzyme = None;
        let mut llvm_offload = None;
        let mut llvm_plugins = None;

        if let Some(llvm) = toml_llvm {
            let Llvm {
                optimize: optimize_toml,
                thin_lto,
                release_debuginfo,
                assertions: _,
                tests,
                enzyme,
                plugins,
                ccache: llvm_ccache,
                static_libstdcpp,
                libzstd,
                ninja,
                targets,
                experimental_targets,
                link_jobs,
                link_shared,
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
                download_ci_llvm,
                build_config,
            } = llvm;
            if llvm_ccache.is_some() {
                eprintln!("Warning: llvm.ccache is deprecated. Use build.ccache instead.");
            }

            if ccache.is_none() {
                *ccache = llvm_ccache;
            }
            set(&mut self.ninja_in_file, ninja);
            llvm_tests = tests;
            llvm_enzyme = enzyme;
            llvm_offload = offload;
            llvm_plugins = plugins;
            set(&mut self.llvm_optimize, optimize_toml);
            set(&mut self.llvm_thin_lto, thin_lto);
            set(&mut self.llvm_release_debuginfo, release_debuginfo);
            set(&mut self.llvm_static_stdcpp, static_libstdcpp);
            set(&mut self.llvm_libzstd, libzstd);
            if let Some(v) = link_shared {
                self.llvm_link_shared.set(Some(v));
            }
            self.llvm_targets.clone_from(&targets);
            self.llvm_experimental_targets.clone_from(&experimental_targets);
            self.llvm_link_jobs = link_jobs;
            self.llvm_version_suffix.clone_from(&version_suffix);
            self.llvm_clang_cl.clone_from(&clang_cl);

            self.llvm_cflags.clone_from(&cflags);
            self.llvm_cxxflags.clone_from(&cxxflags);
            self.llvm_ldflags.clone_from(&ldflags);
            set(&mut self.llvm_use_libcxx, use_libcxx);
            self.llvm_use_linker.clone_from(&use_linker);
            self.llvm_allow_old_toolchain = allow_old_toolchain.unwrap_or(false);
            self.llvm_offload = offload.unwrap_or(false);
            self.llvm_polly = polly.unwrap_or(false);
            self.llvm_clang = clang.unwrap_or(false);
            self.llvm_enable_warnings = enable_warnings.unwrap_or(false);
            self.llvm_build_config = build_config.clone().unwrap_or(Default::default());

            self.llvm_from_ci = self.parse_download_ci_llvm(download_ci_llvm, self.llvm_assertions);

            if self.llvm_from_ci {
                let warn = |option: &str| {
                    println!(
                        "WARNING: `{option}` will only be used on `compiler/rustc_llvm` build, not for the LLVM build."
                    );
                    println!(
                        "HELP: To use `{option}` for LLVM builds, set `download-ci-llvm` option to false."
                    );
                };

                if static_libstdcpp.is_some() {
                    warn("static-libstdcpp");
                }

                if link_shared.is_some() {
                    warn("link-shared");
                }

                // FIXME(#129153): instead of all the ad-hoc `download-ci-llvm` checks that follow,
                // use the `builder-config` present in tarballs since #128822 to compare the local
                // config to the ones used to build the LLVM artifacts on CI, and only notify users
                // if they've chosen a different value.

                if libzstd.is_some() {
                    println!(
                        "WARNING: when using `download-ci-llvm`, the local `llvm.libzstd` option, \
                        like almost all `llvm.*` options, will be ignored and set by the LLVM CI \
                        artifacts builder config."
                    );
                    println!(
                        "HELP: To use `llvm.libzstd` for LLVM/LLD builds, set `download-ci-llvm` option to false."
                    );
                }
            }

            if !self.llvm_from_ci && self.llvm_thin_lto && link_shared.is_none() {
                // If we're building with ThinLTO on, by default we want to link
                // to LLVM shared, to avoid re-doing ThinLTO (which happens in
                // the link step) with each stage.
                self.llvm_link_shared.set(Some(true));
            }
        } else {
            self.llvm_from_ci = self.parse_download_ci_llvm(None, false);
        }

        self.llvm_tests = llvm_tests.unwrap_or(false);
        self.llvm_enzyme = llvm_enzyme.unwrap_or(false);
        self.llvm_offload = llvm_offload.unwrap_or(false);
        self.llvm_plugins = llvm_plugins.unwrap_or(false);
    }
}
