//! This module defines the structures that directly mirror the `bootstrap.toml`
//! file's format. These types are used for `serde` deserialization, serving as an
//! intermediate representation that gets processed and merged into the final `Config`
//! types from the `types` module.
//!
//! It also houses the `Merge` trait and `define_config!` macro, which are essential
//! for handling these raw TOML structures.

use serde::{Deserialize, Deserializer};
use serde_derive::Deserialize;

use crate::core::config::types::{DebuginfoLevel, LldMode, RustOptimize, StringOrBool};
use crate::{BTreeSet, Config, HashMap, HashSet, PathBuf, define_config, exit};

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

/// Describes how to handle conflicts in merging two [`TomlConfig`]
#[derive(Copy, Clone, Debug)]
pub(super) enum ReplaceOpt {
    /// Silently ignore a duplicated value
    IgnoreDuplicate,
    /// Override the current value, even if it's `Some`
    Override,
    /// Exit with an error on duplicate values
    ErrorOnDuplicate,
}

pub(super) trait Merge {
    fn merge(
        &mut self,
        parent_config_path: Option<PathBuf>,
        included_extensions: &mut HashSet<PathBuf>,
        other: Self,
        replace: ReplaceOpt,
    );
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

/// This enum is used for deserializing change IDs from TOML, allowing both numeric values and the string `"ignore"`.
#[derive(Clone, Debug, PartialEq)]
pub enum ChangeId {
    Ignore,
    Id(usize),
}

/// Since we use `#[serde(deny_unknown_fields)]` on `TomlConfig`, we need a wrapper type
/// for the "change-id" field to parse it even if other fields are invalid. This ensures
/// that if deserialization fails due to other fields, we can still provide the changelogs
/// to allow developers to potentially find the reason for the failure in the logs..
#[derive(Deserialize, Default)]
pub(crate) struct ChangeIdWrapper {
    #[serde(alias = "change-id", default, deserialize_with = "deserialize_change_id")]
    pub(crate) inner: Option<ChangeId>,
}

fn deserialize_change_id<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ChangeId>, D::Error> {
    let value = toml::Value::deserialize(deserializer)?;
    Ok(match value {
        toml::Value::String(s) if s == "ignore" => Some(ChangeId::Ignore),
        toml::Value::Integer(i) => Some(ChangeId::Id(i as usize)),
        _ => {
            return Err(serde::de::Error::custom(
                "expected \"ignore\" or an integer for change-id",
            ));
        }
    })
}

define_config! {
    /// TOML representation of various global build decisions.
    #[derive(Default)]
    struct Build {
        build: Option<String> = "build",
        description: Option<String> = "description",
        host: Option<Vec<String>> = "host",
        target: Option<Vec<String>> = "target",
        build_dir: Option<String> = "build-dir",
        cargo: Option<PathBuf> = "cargo",
        rustc: Option<PathBuf> = "rustc",
        rustfmt: Option<PathBuf> = "rustfmt",
        cargo_clippy: Option<PathBuf> = "cargo-clippy",
        docs: Option<bool> = "docs",
        compiler_docs: Option<bool> = "compiler-docs",
        library_docs_private_items: Option<bool> = "library-docs-private-items",
        docs_minification: Option<bool> = "docs-minification",
        submodules: Option<bool> = "submodules",
        gdb: Option<String> = "gdb",
        lldb: Option<String> = "lldb",
        nodejs: Option<String> = "nodejs",
        npm: Option<String> = "npm",
        python: Option<String> = "python",
        reuse: Option<String> = "reuse",
        locked_deps: Option<bool> = "locked-deps",
        vendor: Option<bool> = "vendor",
        full_bootstrap: Option<bool> = "full-bootstrap",
        bootstrap_cache_path: Option<PathBuf> = "bootstrap-cache-path",
        extended: Option<bool> = "extended",
        tools: Option<HashSet<String>> = "tools",
        verbose: Option<usize> = "verbose",
        sanitizers: Option<bool> = "sanitizers",
        profiler: Option<bool> = "profiler",
        cargo_native_static: Option<bool> = "cargo-native-static",
        low_priority: Option<bool> = "low-priority",
        configure_args: Option<Vec<String>> = "configure-args",
        local_rebuild: Option<bool> = "local-rebuild",
        print_step_timings: Option<bool> = "print-step-timings",
        print_step_rusage: Option<bool> = "print-step-rusage",
        check_stage: Option<u32> = "check-stage",
        doc_stage: Option<u32> = "doc-stage",
        build_stage: Option<u32> = "build-stage",
        test_stage: Option<u32> = "test-stage",
        install_stage: Option<u32> = "install-stage",
        dist_stage: Option<u32> = "dist-stage",
        bench_stage: Option<u32> = "bench-stage",
        patch_binaries_for_nix: Option<bool> = "patch-binaries-for-nix",
        // NOTE: only parsed by bootstrap.py, `--feature build-metrics` enables metrics unconditionally
        metrics: Option<bool> = "metrics",
        android_ndk: Option<PathBuf> = "android-ndk",
        optimized_compiler_builtins: Option<bool> = "optimized-compiler-builtins",
        jobs: Option<u32> = "jobs",
        compiletest_diff_tool: Option<String> = "compiletest-diff-tool",
        compiletest_use_stage0_libtest: Option<bool> = "compiletest-use-stage0-libtest",
        ccache: Option<StringOrBool> = "ccache",
        exclude: Option<Vec<PathBuf>> = "exclude",
    }
}

define_config! {
    /// TOML representation of various global install decisions.
    struct Install {
        prefix: Option<String> = "prefix",
        sysconfdir: Option<String> = "sysconfdir",
        docdir: Option<String> = "docdir",
        bindir: Option<String> = "bindir",
        libdir: Option<String> = "libdir",
        mandir: Option<String> = "mandir",
        datadir: Option<String> = "datadir",
    }
}

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

define_config! {
    /// TOML representation of how the GCC build is configured.
    struct Gcc {
        download_ci_gcc: Option<bool> = "download-ci-gcc",
    }
}

define_config! {
    /// TOML representation of how the Rust build is configured.
    struct Rust {
        optimize: Option<RustOptimize> = "optimize",
        debug: Option<bool> = "debug",
        codegen_units: Option<u32> = "codegen-units",
        codegen_units_std: Option<u32> = "codegen-units-std",
        rustc_debug_assertions: Option<bool> = "debug-assertions",
        randomize_layout: Option<bool> = "randomize-layout",
        std_debug_assertions: Option<bool> = "debug-assertions-std",
        tools_debug_assertions: Option<bool> = "debug-assertions-tools",
        overflow_checks: Option<bool> = "overflow-checks",
        overflow_checks_std: Option<bool> = "overflow-checks-std",
        debug_logging: Option<bool> = "debug-logging",
        debuginfo_level: Option<DebuginfoLevel> = "debuginfo-level",
        debuginfo_level_rustc: Option<DebuginfoLevel> = "debuginfo-level-rustc",
        debuginfo_level_std: Option<DebuginfoLevel> = "debuginfo-level-std",
        debuginfo_level_tools: Option<DebuginfoLevel> = "debuginfo-level-tools",
        debuginfo_level_tests: Option<DebuginfoLevel> = "debuginfo-level-tests",
        backtrace: Option<bool> = "backtrace",
        incremental: Option<bool> = "incremental",
        default_linker: Option<String> = "default-linker",
        channel: Option<String> = "channel",
        // FIXME: Remove this field at Q2 2025, it has been replaced by build.description
        description: Option<String> = "description",
        musl_root: Option<String> = "musl-root",
        rpath: Option<bool> = "rpath",
        strip: Option<bool> = "strip",
        frame_pointers: Option<bool> = "frame-pointers",
        stack_protector: Option<String> = "stack-protector",
        verbose_tests: Option<bool> = "verbose-tests",
        optimize_tests: Option<bool> = "optimize-tests",
        codegen_tests: Option<bool> = "codegen-tests",
        omit_git_hash: Option<bool> = "omit-git-hash",
        dist_src: Option<bool> = "dist-src",
        save_toolstates: Option<String> = "save-toolstates",
        codegen_backends: Option<Vec<String>> = "codegen-backends",
        llvm_bitcode_linker: Option<bool> = "llvm-bitcode-linker",
        lld: Option<bool> = "lld",
        lld_mode: Option<LldMode> = "use-lld",
        llvm_tools: Option<bool> = "llvm-tools",
        deny_warnings: Option<bool> = "deny-warnings",
        backtrace_on_ice: Option<bool> = "backtrace-on-ice",
        verify_llvm_ir: Option<bool> = "verify-llvm-ir",
        thin_lto_import_instr_limit: Option<u32> = "thin-lto-import-instr-limit",
        remap_debuginfo: Option<bool> = "remap-debuginfo",
        jemalloc: Option<bool> = "jemalloc",
        test_compare_mode: Option<bool> = "test-compare-mode",
        llvm_libunwind: Option<String> = "llvm-libunwind",
        control_flow_guard: Option<bool> = "control-flow-guard",
        ehcont_guard: Option<bool> = "ehcont-guard",
        new_symbol_mangling: Option<bool> = "new-symbol-mangling",
        profile_generate: Option<String> = "profile-generate",
        profile_use: Option<String> = "profile-use",
        // ignored; this is set from an env var set by bootstrap.py
        download_rustc: Option<StringOrBool> = "download-rustc",
        lto: Option<String> = "lto",
        validate_mir_opts: Option<u32> = "validate-mir-opts",
        std_features: Option<BTreeSet<String>> = "std-features",
    }
}

define_config! {
    struct Dist {
        sign_folder: Option<String> = "sign-folder",
        upload_addr: Option<String> = "upload-addr",
        src_tarball: Option<bool> = "src-tarball",
        compression_formats: Option<Vec<String>> = "compression-formats",
        compression_profile: Option<String> = "compression-profile",
        include_mingw_linker: Option<bool> = "include-mingw-linker",
        vendor: Option<bool> = "vendor",
    }
}

define_config! {
    /// TOML representation of how each build target is configured.
    struct TomlTarget {
        cc: Option<String> = "cc",
        cxx: Option<String> = "cxx",
        ar: Option<String> = "ar",
        ranlib: Option<String> = "ranlib",
        default_linker: Option<PathBuf> = "default-linker",
        linker: Option<String> = "linker",
        split_debuginfo: Option<String> = "split-debuginfo",
        llvm_config: Option<String> = "llvm-config",
        llvm_has_rust_patches: Option<bool> = "llvm-has-rust-patches",
        llvm_filecheck: Option<String> = "llvm-filecheck",
        llvm_libunwind: Option<String> = "llvm-libunwind",
        sanitizers: Option<bool> = "sanitizers",
        profiler: Option<StringOrBool> = "profiler",
        rpath: Option<bool> = "rpath",
        crt_static: Option<bool> = "crt-static",
        musl_root: Option<String> = "musl-root",
        musl_libdir: Option<String> = "musl-libdir",
        wasi_root: Option<String> = "wasi-root",
        qemu_rootfs: Option<String> = "qemu-rootfs",
        no_std: Option<bool> = "no-std",
        codegen_backends: Option<Vec<String>> = "codegen-backends",
        runner: Option<String> = "runner",
        optimized_compiler_builtins: Option<bool> = "optimized-compiler-builtins",
        jemalloc: Option<bool> = "jemalloc",
    }
}
