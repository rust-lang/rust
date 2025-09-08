//! This module defines the central `Config` struct, which aggregates all components
//! of the bootstrap configuration into a single unit.
//!
//! It serves as the primary public interface for accessing the bootstrap configuration.
//! The module coordinates the overall configuration parsing process using logic from `parsing.rs`
//! and provides top-level methods such as `Config::parse()` for initialization, as well as
//! utility methods for querying and manipulating the complete configuration state.
//!
//! Additionally, this module contains the core logic for parsing, validating, and inferring
//! the final `Config` from various raw inputs.
//!
//! It manages the process of reading command-line arguments, environment variables,
//! and the `bootstrap.toml` file—merging them, applying defaults, and performing
//! cross-component validation. The main `parse_inner` function and its supporting
//! helpers reside here, transforming raw `Toml` data into the structured `Config` type.
use std::cell::Cell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::IsTerminal;
use std::path::{Path, PathBuf, absolute};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::{cmp, env, fs};

use build_helper::ci::CiEnv;
use build_helper::exit;
use build_helper::git::{GitConfig, PathFreshness, check_path_modifications};
use serde::Deserialize;
#[cfg(feature = "tracing")]
use tracing::{instrument, span};

use crate::core::build_steps::llvm;
use crate::core::build_steps::llvm::LLVM_INVALIDATION_PATHS;
pub use crate::core::config::flags::Subcommand;
use crate::core::config::flags::{Color, Flags, Warnings};
use crate::core::config::target_selection::TargetSelectionList;
use crate::core::config::toml::TomlConfig;
use crate::core::config::toml::build::{Build, Tool};
use crate::core::config::toml::change_id::ChangeId;
use crate::core::config::toml::dist::Dist;
use crate::core::config::toml::gcc::Gcc;
use crate::core::config::toml::install::Install;
use crate::core::config::toml::llvm::Llvm;
use crate::core::config::toml::rust::{
    LldMode, Rust, RustOptimize, check_incompatible_options_for_ci_rustc,
    default_lld_opt_in_targets, parse_codegen_backends,
};
use crate::core::config::toml::target::Target;
use crate::core::config::{
    CompilerBuiltins, DebuginfoLevel, DryRun, GccCiMode, LlvmLibunwind, Merge, ReplaceOpt,
    RustcLto, SplitDebuginfo, StringOrBool, threads_from_config,
};
use crate::core::download::{
    DownloadContext, download_beta_toolchain, is_download_ci_available, maybe_download_rustfmt,
};
use crate::utils::channel;
use crate::utils::exec::{ExecutionContext, command};
use crate::utils::helpers::{exe, get_host_target};
use crate::{CodegenBackendKind, GitInfo, OnceLock, TargetSelection, check_ci_llvm, helpers, t};

/// Each path in this list is considered "allowed" in the `download-rustc="if-unchanged"` logic.
/// This means they can be modified and changes to these paths should never trigger a compiler build
/// when "if-unchanged" is set.
///
/// NOTE: Paths must have the ":!" prefix to tell git to ignore changes in those paths during
/// the diff check.
///
/// WARNING: Be cautious when adding paths to this list. If a path that influences the compiler build
/// is added here, it will cause bootstrap to skip necessary rebuilds, which may lead to risky results.
/// For example, "src/bootstrap" should never be included in this list as it plays a crucial role in the
/// final output/compiler, which can be significantly affected by changes made to the bootstrap sources.
#[rustfmt::skip] // We don't want rustfmt to oneline this list
pub const RUSTC_IF_UNCHANGED_ALLOWED_PATHS: &[&str] = &[
    ":!library",
    ":!src/tools",
    ":!src/librustdoc",
    ":!src/rustdoc-json-types",
    ":!tests",
    ":!triagebot.toml",
];

/// Global configuration for the entire build and/or bootstrap.
///
/// This structure is parsed from `bootstrap.toml`, and some of the fields are inferred from `git` or build-time parameters.
///
/// Note that this structure is not decoded directly into, but rather it is
/// filled out from the decoded forms of the structs below. For documentation
/// on each field, see the corresponding fields in
/// `bootstrap.example.toml`.
#[derive(Default, Clone)]
pub struct Config {
    pub change_id: Option<ChangeId>,
    pub bypass_bootstrap_lock: bool,
    pub ccache: Option<String>,
    /// Call Build::ninja() instead of this.
    pub ninja_in_file: bool,
    pub submodules: Option<bool>,
    pub compiler_docs: bool,
    pub library_docs_private_items: bool,
    pub docs_minification: bool,
    pub docs: bool,
    pub locked_deps: bool,
    pub vendor: bool,
    pub target_config: HashMap<TargetSelection, Target>,
    pub full_bootstrap: bool,
    pub bootstrap_cache_path: Option<PathBuf>,
    pub extended: bool,
    pub tools: Option<HashSet<String>>,
    /// Specify build configuration specific for some tool, such as enabled features, see [Tool].
    /// The key in the map is the name of the tool, and the value is tool-specific configuration.
    pub tool: HashMap<String, Tool>,
    pub sanitizers: bool,
    pub profiler: bool,
    pub omit_git_hash: bool,
    pub skip: Vec<PathBuf>,
    pub include_default_paths: bool,
    pub rustc_error_format: Option<String>,
    pub json_output: bool,
    pub compile_time_deps: bool,
    pub test_compare_mode: bool,
    pub color: Color,
    pub patch_binaries_for_nix: Option<bool>,
    pub stage0_metadata: build_helper::stage0_parser::Stage0,
    pub android_ndk: Option<PathBuf>,
    pub optimized_compiler_builtins: CompilerBuiltins,

    pub stdout_is_tty: bool,
    pub stderr_is_tty: bool,

    pub on_fail: Option<String>,
    pub explicit_stage_from_cli: bool,
    pub explicit_stage_from_config: bool,
    pub stage: u32,
    pub keep_stage: Vec<u32>,
    pub keep_stage_std: Vec<u32>,
    pub src: PathBuf,
    /// defaults to `bootstrap.toml`
    pub config: Option<PathBuf>,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,
    pub dump_bootstrap_shims: bool,
    /// Arguments appearing after `--` to be forwarded to tools,
    /// e.g. `--fix-broken` or test arguments.
    pub free_args: Vec<String>,

    /// `None` if we shouldn't download CI compiler artifacts, or the commit to download if we should.
    pub download_rustc_commit: Option<String>,

    pub deny_warnings: bool,
    pub backtrace_on_ice: bool,

    // llvm codegen options
    pub llvm_assertions: bool,
    pub llvm_tests: bool,
    pub llvm_enzyme: bool,
    pub llvm_offload: bool,
    pub llvm_plugins: bool,
    pub llvm_optimize: bool,
    pub llvm_thin_lto: bool,
    pub llvm_release_debuginfo: bool,
    pub llvm_static_stdcpp: bool,
    pub llvm_libzstd: bool,
    pub llvm_link_shared: Cell<Option<bool>>,
    pub llvm_clang_cl: Option<String>,
    pub llvm_targets: Option<String>,
    pub llvm_experimental_targets: Option<String>,
    pub llvm_link_jobs: Option<u32>,
    pub llvm_version_suffix: Option<String>,
    pub llvm_use_linker: Option<String>,
    pub llvm_allow_old_toolchain: bool,
    pub llvm_polly: bool,
    pub llvm_clang: bool,
    pub llvm_enable_warnings: bool,
    pub llvm_from_ci: bool,
    pub llvm_build_config: HashMap<String, String>,

    pub lld_mode: LldMode,
    pub lld_enabled: bool,
    pub llvm_tools_enabled: bool,
    pub llvm_bitcode_linker_enabled: bool,

    pub llvm_cflags: Option<String>,
    pub llvm_cxxflags: Option<String>,
    pub llvm_ldflags: Option<String>,
    pub llvm_use_libcxx: bool,

    // gcc codegen options
    pub gcc_ci_mode: GccCiMode,

    // rust codegen options
    pub rust_optimize: RustOptimize,
    pub rust_codegen_units: Option<u32>,
    pub rust_codegen_units_std: Option<u32>,

    pub rustc_debug_assertions: bool,
    pub std_debug_assertions: bool,
    pub tools_debug_assertions: bool,

    pub rust_overflow_checks: bool,
    pub rust_overflow_checks_std: bool,
    pub rust_debug_logging: bool,
    pub rust_debuginfo_level_rustc: DebuginfoLevel,
    pub rust_debuginfo_level_std: DebuginfoLevel,
    pub rust_debuginfo_level_tools: DebuginfoLevel,
    pub rust_debuginfo_level_tests: DebuginfoLevel,
    pub rust_rpath: bool,
    pub rust_strip: bool,
    pub rust_frame_pointers: bool,
    pub rust_stack_protector: Option<String>,
    pub rustc_default_linker: Option<String>,
    pub rust_optimize_tests: bool,
    pub rust_dist_src: bool,
    pub rust_codegen_backends: Vec<CodegenBackendKind>,
    pub rust_verify_llvm_ir: bool,
    pub rust_thin_lto_import_instr_limit: Option<u32>,
    pub rust_randomize_layout: bool,
    pub rust_remap_debuginfo: bool,
    pub rust_new_symbol_mangling: Option<bool>,
    pub rust_profile_use: Option<String>,
    pub rust_profile_generate: Option<String>,
    pub rust_lto: RustcLto,
    pub rust_validate_mir_opts: Option<u32>,
    pub rust_std_features: BTreeSet<String>,
    pub rust_break_on_ice: bool,
    pub llvm_profile_use: Option<String>,
    pub llvm_profile_generate: bool,
    pub llvm_libunwind_default: Option<LlvmLibunwind>,
    pub enable_bolt_settings: bool,

    pub reproducible_artifacts: Vec<String>,

    pub host_target: TargetSelection,
    pub hosts: Vec<TargetSelection>,
    pub targets: Vec<TargetSelection>,
    pub local_rebuild: bool,
    pub jemalloc: bool,
    pub control_flow_guard: bool,
    pub ehcont_guard: bool,

    // dist misc
    pub dist_sign_folder: Option<PathBuf>,
    pub dist_upload_addr: Option<String>,
    pub dist_compression_formats: Option<Vec<String>>,
    pub dist_compression_profile: String,
    pub dist_include_mingw_linker: bool,
    pub dist_vendor: bool,

    // libstd features
    pub backtrace: bool, // support for RUST_BACKTRACE

    // misc
    pub low_priority: bool,
    pub channel: String,
    pub description: Option<String>,
    pub verbose_tests: bool,
    pub save_toolstates: Option<PathBuf>,
    pub print_step_timings: bool,
    pub print_step_rusage: bool,

    // Fallback musl-root for all targets
    pub musl_root: Option<PathBuf>,
    pub prefix: Option<PathBuf>,
    pub sysconfdir: Option<PathBuf>,
    pub datadir: Option<PathBuf>,
    pub docdir: Option<PathBuf>,
    pub bindir: PathBuf,
    pub libdir: Option<PathBuf>,
    pub mandir: Option<PathBuf>,
    pub codegen_tests: bool,
    pub nodejs: Option<PathBuf>,
    pub npm: Option<PathBuf>,
    pub gdb: Option<PathBuf>,
    pub lldb: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub reuse: Option<PathBuf>,
    pub cargo_native_static: bool,
    pub configure_args: Vec<String>,
    pub out: PathBuf,
    pub rust_info: channel::GitInfo,

    pub cargo_info: channel::GitInfo,
    pub rust_analyzer_info: channel::GitInfo,
    pub clippy_info: channel::GitInfo,
    pub miri_info: channel::GitInfo,
    pub rustfmt_info: channel::GitInfo,
    pub enzyme_info: channel::GitInfo,
    pub in_tree_llvm_info: channel::GitInfo,
    pub in_tree_gcc_info: channel::GitInfo,

    // These are either the stage0 downloaded binaries or the locally installed ones.
    pub initial_cargo: PathBuf,
    pub initial_rustc: PathBuf,
    pub initial_cargo_clippy: Option<PathBuf>,
    pub initial_sysroot: PathBuf,
    pub initial_rustfmt: Option<PathBuf>,

    /// The paths to work with. For example: with `./x check foo bar` we get
    /// `paths=["foo", "bar"]`.
    pub paths: Vec<PathBuf>,

    /// Command for visual diff display, e.g. `diff-tool --color=always`.
    pub compiletest_diff_tool: Option<String>,

    /// Whether to allow running both `compiletest` self-tests and `compiletest`-managed test suites
    /// against the stage 0 (rustc, std).
    ///
    /// This is only intended to be used when the stage 0 compiler is actually built from in-tree
    /// sources.
    pub compiletest_allow_stage0: bool,

    /// Whether to use the precompiled stage0 libtest with compiletest.
    pub compiletest_use_stage0_libtest: bool,

    /// Default value for `--extra-checks`
    pub tidy_extra_checks: Option<String>,
    pub is_running_on_ci: bool,

    /// Cache for determining path modifications
    pub path_modification_cache: Arc<Mutex<HashMap<Vec<&'static str>, PathFreshness>>>,

    /// Skip checking the standard library if `rust.download-rustc` isn't available.
    /// This is mostly for RA as building the stage1 compiler to check the library tree
    /// on each code change might be too much for some computers.
    pub skip_std_check_if_no_download_rustc: bool,

    pub exec_ctx: ExecutionContext,
}

impl Config {
    pub fn set_dry_run(&mut self, dry_run: DryRun) {
        self.exec_ctx.set_dry_run(dry_run);
    }

    pub fn get_dry_run(&self) -> &DryRun {
        self.exec_ctx.get_dry_run()
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(target = "CONFIG_HANDLING", level = "trace", name = "Config::parse", skip_all)
    )]
    pub fn parse(flags: Flags) -> Config {
        Self::parse_inner(flags, Self::get_toml)
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(
            target = "CONFIG_HANDLING",
            level = "trace",
            name = "Config::parse_inner",
            skip_all
        )
    )]
    pub(crate) fn parse_inner(
        flags: Flags,
        get_toml: impl Fn(&Path) -> Result<TomlConfig, toml::de::Error>,
    ) -> Config {
        // Destructure flags to ensure that we use all its fields
        // The field variables are prefixed with `flags_` to avoid clashes
        // with values from TOML config files with same names.
        let Flags {
            cmd: flags_cmd,
            verbose: flags_verbose,
            incremental: flags_incremental,
            config: flags_config,
            build_dir: flags_build_dir,
            build: flags_build,
            host: flags_host,
            target: flags_target,
            exclude: flags_exclude,
            skip: flags_skip,
            include_default_paths: flags_include_default_paths,
            rustc_error_format: flags_rustc_error_format,
            on_fail: flags_on_fail,
            dry_run: flags_dry_run,
            dump_bootstrap_shims: flags_dump_bootstrap_shims,
            stage: flags_stage,
            keep_stage: flags_keep_stage,
            keep_stage_std: flags_keep_stage_std,
            src: flags_src,
            jobs: flags_jobs,
            warnings: flags_warnings,
            json_output: flags_json_output,
            compile_time_deps: flags_compile_time_deps,
            color: flags_color,
            bypass_bootstrap_lock: flags_bypass_bootstrap_lock,
            rust_profile_generate: flags_rust_profile_generate,
            rust_profile_use: flags_rust_profile_use,
            llvm_profile_use: flags_llvm_profile_use,
            llvm_profile_generate: flags_llvm_profile_generate,
            enable_bolt_settings: flags_enable_bolt_settings,
            skip_stage0_validation: flags_skip_stage0_validation,
            reproducible_artifact: flags_reproducible_artifact,
            paths: flags_paths,
            set: flags_set,
            free_args: flags_free_args,
            ci: flags_ci,
            skip_std_check_if_no_download_rustc: flags_skip_std_check_if_no_download_rustc,
        } = flags;

        #[cfg(feature = "tracing")]
        span!(
            target: "CONFIG_HANDLING",
            tracing::Level::TRACE,
            "collecting paths and path exclusions",
            "flags.paths" = ?flags_paths,
            "flags.skip" = ?flags_skip,
            "flags.exclude" = ?flags_exclude
        );

        // Set config values based on flags.
        let mut exec_ctx = ExecutionContext::new(flags_verbose, flags_cmd.fail_fast());
        exec_ctx.set_dry_run(if flags_dry_run { DryRun::UserSelected } else { DryRun::Disabled });
        let mut src = {
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            // Undo `src/bootstrap`
            manifest_dir.parent().unwrap().parent().unwrap().to_owned()
        };

        if let Some(src_) = compute_src_directory(flags_src, &exec_ctx) {
            src = src_;
        }

        // Now load the TOML config, as soon as possible
        let (mut toml, toml_path) = load_toml_config(&src, flags_config, &get_toml);

        postprocess_toml(&mut toml, &src, toml_path.clone(), &exec_ctx, &flags_set, &get_toml);

        // Now override TOML values with flags, to make sure that we won't later override flags with
        // TOML values by accident instead, because flags have higher priority.
        let Build {
            description: build_description,
            build: build_build,
            host: build_host,
            target: build_target,
            build_dir: build_build_dir,
            cargo: mut build_cargo,
            rustc: mut build_rustc,
            rustfmt: build_rustfmt,
            cargo_clippy: build_cargo_clippy,
            docs: build_docs,
            compiler_docs: build_compiler_docs,
            library_docs_private_items: build_library_docs_private_items,
            docs_minification: build_docs_minification,
            submodules: build_submodules,
            gdb: build_gdb,
            lldb: build_lldb,
            nodejs: build_nodejs,
            npm: build_npm,
            python: build_python,
            reuse: build_reuse,
            locked_deps: build_locked_deps,
            vendor: build_vendor,
            full_bootstrap: build_full_bootstrap,
            bootstrap_cache_path: build_bootstrap_cache_path,
            extended: build_extended,
            tools: build_tools,
            tool: build_tool,
            verbose: build_verbose,
            sanitizers: build_sanitizers,
            profiler: build_profiler,
            cargo_native_static: build_cargo_native_static,
            low_priority: build_low_priority,
            configure_args: build_configure_args,
            local_rebuild: build_local_rebuild,
            print_step_timings: build_print_step_timings,
            print_step_rusage: build_print_step_rusage,
            check_stage: build_check_stage,
            doc_stage: build_doc_stage,
            build_stage: build_build_stage,
            test_stage: build_test_stage,
            install_stage: build_install_stage,
            dist_stage: build_dist_stage,
            bench_stage: build_bench_stage,
            patch_binaries_for_nix: build_patch_binaries_for_nix,
            // This field is only used by bootstrap.py
            metrics: _,
            android_ndk: build_android_ndk,
            optimized_compiler_builtins: build_optimized_compiler_builtins,
            jobs: build_jobs,
            compiletest_diff_tool: build_compiletest_diff_tool,
            compiletest_use_stage0_libtest: build_compiletest_use_stage0_libtest,
            tidy_extra_checks: build_tidy_extra_checks,
            ccache: build_ccache,
            exclude: build_exclude,
            compiletest_allow_stage0: build_compiletest_allow_stage0,
        } = toml.build.unwrap_or_default();

        let Install {
            prefix: install_prefix,
            sysconfdir: install_sysconfdir,
            docdir: install_docdir,
            bindir: install_bindir,
            libdir: install_libdir,
            mandir: install_mandir,
            datadir: install_datadir,
        } = toml.install.unwrap_or_default();

        let Rust {
            optimize: rust_optimize,
            debug: rust_debug,
            codegen_units: rust_codegen_units,
            codegen_units_std: rust_codegen_units_std,
            rustc_debug_assertions: rust_rustc_debug_assertions,
            std_debug_assertions: rust_std_debug_assertions,
            tools_debug_assertions: rust_tools_debug_assertions,
            overflow_checks: rust_overflow_checks,
            overflow_checks_std: rust_overflow_checks_std,
            debug_logging: rust_debug_logging,
            debuginfo_level: rust_debuginfo_level,
            debuginfo_level_rustc: rust_debuginfo_level_rustc,
            debuginfo_level_std: rust_debuginfo_level_std,
            debuginfo_level_tools: rust_debuginfo_level_tools,
            debuginfo_level_tests: rust_debuginfo_level_tests,
            backtrace: rust_backtrace,
            incremental: rust_incremental,
            randomize_layout: rust_randomize_layout,
            default_linker: rust_default_linker,
            channel: rust_channel,
            musl_root: rust_musl_root,
            rpath: rust_rpath,
            verbose_tests: rust_verbose_tests,
            optimize_tests: rust_optimize_tests,
            codegen_tests: rust_codegen_tests,
            omit_git_hash: rust_omit_git_hash,
            dist_src: rust_dist_src,
            save_toolstates: rust_save_toolstates,
            codegen_backends: rust_codegen_backends,
            lld: rust_lld_enabled,
            llvm_tools: rust_llvm_tools,
            llvm_bitcode_linker: rust_llvm_bitcode_linker,
            deny_warnings: rust_deny_warnings,
            backtrace_on_ice: rust_backtrace_on_ice,
            verify_llvm_ir: rust_verify_llvm_ir,
            thin_lto_import_instr_limit: rust_thin_lto_import_instr_limit,
            remap_debuginfo: rust_remap_debuginfo,
            jemalloc: rust_jemalloc,
            test_compare_mode: rust_test_compare_mode,
            llvm_libunwind: rust_llvm_libunwind,
            control_flow_guard: rust_control_flow_guard,
            ehcont_guard: rust_ehcont_guard,
            new_symbol_mangling: rust_new_symbol_mangling,
            profile_generate: rust_profile_generate,
            profile_use: rust_profile_use,
            download_rustc: rust_download_rustc,
            lto: rust_lto,
            validate_mir_opts: rust_validate_mir_opts,
            frame_pointers: rust_frame_pointers,
            stack_protector: rust_stack_protector,
            strip: rust_strip,
            lld_mode: rust_lld_mode,
            std_features: rust_std_features,
            break_on_ice: rust_break_on_ice,
        } = toml.rust.unwrap_or_default();

        let Llvm {
            optimize: llvm_optimize,
            thin_lto: llvm_thin_lto,
            release_debuginfo: llvm_release_debuginfo,
            assertions: llvm_assertions,
            tests: llvm_tests,
            enzyme: llvm_enzyme,
            plugins: llvm_plugin,
            static_libstdcpp: llvm_static_libstdcpp,
            libzstd: llvm_libzstd,
            ninja: llvm_ninja,
            targets: llvm_targets,
            experimental_targets: llvm_experimental_targets,
            link_jobs: llvm_link_jobs,
            link_shared: llvm_link_shared,
            version_suffix: llvm_version_suffix,
            clang_cl: llvm_clang_cl,
            cflags: llvm_cflags,
            cxxflags: llvm_cxxflags,
            ldflags: llvm_ldflags,
            use_libcxx: llvm_use_libcxx,
            use_linker: llvm_use_linker,
            allow_old_toolchain: llvm_allow_old_toolchain,
            offload: llvm_offload,
            polly: llvm_polly,
            clang: llvm_clang,
            enable_warnings: llvm_enable_warnings,
            download_ci_llvm: llvm_download_ci_llvm,
            build_config: llvm_build_config,
        } = toml.llvm.unwrap_or_default();

        let Dist {
            sign_folder: dist_sign_folder,
            upload_addr: dist_upload_addr,
            src_tarball: dist_src_tarball,
            compression_formats: dist_compression_formats,
            compression_profile: dist_compression_profile,
            include_mingw_linker: dist_include_mingw_linker,
            vendor: dist_vendor,
        } = toml.dist.unwrap_or_default();

        let Gcc { download_ci_gcc: gcc_download_ci_gcc } = toml.gcc.unwrap_or_default();

        if rust_optimize.as_ref().is_some_and(|v| matches!(v, RustOptimize::Bool(false))) {
            eprintln!(
                "WARNING: setting `optimize` to `false` is known to cause errors and \
                should be considered unsupported. Refer to `bootstrap.example.toml` \
                for more details."
            );
        }

        // Prefer CLI verbosity flags if set (`flags_verbose` > 0), otherwise take the value from
        // TOML.
        exec_ctx.set_verbosity(cmp::max(build_verbose.unwrap_or_default() as u8, flags_verbose));

        let stage0_metadata = build_helper::stage0_parser::parse_stage0_file();
        let path_modification_cache = Arc::new(Mutex::new(HashMap::new()));

        let host_target = flags_build
            .or(build_build)
            .map(|build| TargetSelection::from_user(&build))
            .unwrap_or_else(get_host_target);
        let hosts = flags_host
            .map(|TargetSelectionList(hosts)| hosts)
            .or_else(|| {
                build_host.map(|h| h.iter().map(|t| TargetSelection::from_user(t)).collect())
            })
            .unwrap_or_else(|| vec![host_target]);

        let llvm_assertions = llvm_assertions.unwrap_or(false);
        let mut target_config = HashMap::new();
        let mut channel = "dev".to_string();
        let out = flags_build_dir.or(build_build_dir.map(PathBuf::from)).unwrap_or_else(|| {
            if cfg!(test) {
                // Use the build directory of the original x.py invocation, so that we can set `initial_rustc` properly.
                Path::new(
                    &env::var_os("CARGO_TARGET_DIR").expect("cargo test directly is not supported"),
                )
                .parent()
                .unwrap()
                .to_path_buf()
            } else {
                PathBuf::from("build")
            }
        });

        // NOTE: Bootstrap spawns various commands with different working directories.
        // To avoid writing to random places on the file system, `config.out` needs to be an absolute path.
        let mut out = if !out.is_absolute() {
            // `canonicalize` requires the path to already exist. Use our vendored copy of `absolute` instead.
            absolute(&out).expect("can't make empty path absolute")
        } else {
            out
        };

        if cfg!(test) {
            // When configuring bootstrap for tests, make sure to set the rustc and Cargo to the
            // same ones used to call the tests (if custom ones are not defined in the toml). If we
            // don't do that, bootstrap will use its own detection logic to find a suitable rustc
            // and Cargo, which doesn't work when the caller is specìfying a custom local rustc or
            // Cargo in their bootstrap.toml.
            build_rustc = build_rustc.take().or(std::env::var_os("RUSTC").map(|p| p.into()));
            build_cargo = build_cargo.take().or(std::env::var_os("CARGO").map(|p| p.into()));
        }

        if !flags_skip_stage0_validation {
            if let Some(rustc) = &build_rustc {
                check_stage0_version(rustc, "rustc", &src, &exec_ctx);
            }
            if let Some(cargo) = &build_cargo {
                check_stage0_version(cargo, "cargo", &src, &exec_ctx);
            }
        }

        if build_cargo_clippy.is_some() && build_rustc.is_none() {
            println!(
                "WARNING: Using `build.cargo-clippy` without `build.rustc` usually fails due to toolchain conflict."
            );
        }

        let is_running_on_ci = flags_ci.unwrap_or(CiEnv::is_ci());
        let dwn_ctx = DownloadContext {
            path_modification_cache: path_modification_cache.clone(),
            src: &src,
            submodules: &build_submodules,
            host_target,
            patch_binaries_for_nix: build_patch_binaries_for_nix,
            exec_ctx: &exec_ctx,
            stage0_metadata: &stage0_metadata,
            llvm_assertions,
            bootstrap_cache_path: &build_bootstrap_cache_path,
            is_running_on_ci,
        };

        let initial_rustc = build_rustc.unwrap_or_else(|| {
            download_beta_toolchain(&dwn_ctx, &out);
            out.join(host_target).join("stage0").join("bin").join(exe("rustc", host_target))
        });

        let initial_sysroot = t!(PathBuf::from_str(
            command(&initial_rustc)
                .args(["--print", "sysroot"])
                .run_in_dry_run()
                .run_capture_stdout(&exec_ctx)
                .stdout()
                .trim()
        ));

        let initial_cargo = build_cargo.unwrap_or_else(|| {
            download_beta_toolchain(&dwn_ctx, &out);
            initial_sysroot.join("bin").join(exe("cargo", host_target))
        });

        // NOTE: it's important this comes *after* we set `initial_rustc` just above.
        if exec_ctx.dry_run() {
            out = out.join("tmp-dry-run");
            fs::create_dir_all(&out).expect("Failed to create dry-run directory");
        }

        let file_content = t!(fs::read_to_string(src.join("src/ci/channel")));
        let ci_channel = file_content.trim_end();

        let is_user_configured_rust_channel = match rust_channel {
            Some(channel_) if channel_ == "auto-detect" => {
                channel = ci_channel.into();
                true
            }
            Some(channel_) => {
                channel = channel_;
                true
            }
            None => false,
        };

        let omit_git_hash = rust_omit_git_hash.unwrap_or(channel == "dev");

        let rust_info = git_info(&exec_ctx, omit_git_hash, &src);

        if !is_user_configured_rust_channel && rust_info.is_from_tarball() {
            channel = ci_channel.into();
        }

        // FIXME(#133381): alt rustc builds currently do *not* have rustc debug assertions
        // enabled. We should not download a CI alt rustc if we need rustc to have debug
        // assertions (e.g. for crashes test suite). This can be changed once something like
        // [Enable debug assertions on alt
        // builds](https://github.com/rust-lang/rust/pull/131077) lands.
        //
        // Note that `rust.debug = true` currently implies `rust.debug-assertions = true`!
        //
        // This relies also on the fact that the global default for `download-rustc` will be
        // `false` if it's not explicitly set.
        let debug_assertions_requested = matches!(rust_rustc_debug_assertions, Some(true))
            || (matches!(rust_debug, Some(true))
                && !matches!(rust_rustc_debug_assertions, Some(false)));

        if debug_assertions_requested
            && let Some(ref opt) = rust_download_rustc
            && opt.is_string_or_true()
        {
            eprintln!(
                "WARN: currently no CI rustc builds have rustc debug assertions \
                        enabled. Please either set `rust.debug-assertions` to `false` if you \
                        want to use download CI rustc or set `rust.download-rustc` to `false`."
            );
        }

        let mut download_rustc_commit =
            download_ci_rustc_commit(&dwn_ctx, &rust_info, rust_download_rustc, llvm_assertions);

        if debug_assertions_requested && download_rustc_commit.is_some() {
            eprintln!(
                "WARN: `rust.debug-assertions = true` will prevent downloading CI rustc as alt CI \
                rustc is not currently built with debug assertions."
            );
            // We need to put this later down_ci_rustc_commit.
            download_rustc_commit = None;
        }

        // We need to override `rust.channel` if it's manually specified when using the CI rustc.
        // This is because if the compiler uses a different channel than the one specified in bootstrap.toml,
        // tests may fail due to using a different channel than the one used by the compiler during tests.
        if let Some(commit) = &download_rustc_commit
            && is_user_configured_rust_channel
        {
            println!(
                "WARNING: `rust.download-rustc` is enabled. The `rust.channel` option will be overridden by the CI rustc's channel."
            );

            channel =
                read_file_by_commit(&dwn_ctx, &rust_info, Path::new("src/ci/channel"), commit)
                    .trim()
                    .to_owned();
        }

        if let Some(t) = toml.target {
            for (triple, cfg) in t {
                let mut target = Target::from_triple(&triple);

                if let Some(ref s) = cfg.llvm_config {
                    if download_rustc_commit.is_some() && triple == *host_target.triple {
                        panic!(
                            "setting llvm_config for the host is incompatible with download-rustc"
                        );
                    }
                    target.llvm_config = Some(src.join(s));
                }
                if let Some(patches) = cfg.llvm_has_rust_patches {
                    assert!(
                        build_submodules == Some(false) || cfg.llvm_config.is_some(),
                        "use of `llvm-has-rust-patches` is restricted to cases where either submodules are disabled or llvm-config been provided"
                    );
                    target.llvm_has_rust_patches = Some(patches);
                }
                if let Some(ref s) = cfg.llvm_filecheck {
                    target.llvm_filecheck = Some(src.join(s));
                }
                target.llvm_libunwind = cfg.llvm_libunwind.as_ref().map(|v| {
                    v.parse().unwrap_or_else(|_| {
                        panic!("failed to parse target.{triple}.llvm-libunwind")
                    })
                });
                if let Some(s) = cfg.no_std {
                    target.no_std = s;
                }
                target.cc = cfg.cc.map(PathBuf::from);
                target.cxx = cfg.cxx.map(PathBuf::from);
                target.ar = cfg.ar.map(PathBuf::from);
                target.ranlib = cfg.ranlib.map(PathBuf::from);
                target.linker = cfg.linker.map(PathBuf::from);
                target.crt_static = cfg.crt_static;
                target.musl_root = cfg.musl_root.map(PathBuf::from);
                target.musl_libdir = cfg.musl_libdir.map(PathBuf::from);
                target.wasi_root = cfg.wasi_root.map(PathBuf::from);
                target.qemu_rootfs = cfg.qemu_rootfs.map(PathBuf::from);
                target.runner = cfg.runner;
                target.sanitizers = cfg.sanitizers;
                target.profiler = cfg.profiler;
                target.rpath = cfg.rpath;
                target.optimized_compiler_builtins = cfg.optimized_compiler_builtins;
                target.jemalloc = cfg.jemalloc;
                if let Some(backends) = cfg.codegen_backends {
                    target.codegen_backends =
                        Some(parse_codegen_backends(backends, &format!("target.{triple}")))
                }

                target.split_debuginfo = cfg.split_debuginfo.as_ref().map(|v| {
                    v.parse().unwrap_or_else(|_| {
                        panic!("invalid value for target.{triple}.split-debuginfo")
                    })
                });

                target_config.insert(TargetSelection::from_user(&triple), target);
            }
        }

        let llvm_from_ci = parse_download_ci_llvm(
            &dwn_ctx,
            &rust_info,
            &download_rustc_commit,
            llvm_download_ci_llvm,
            llvm_assertions,
        );

        // We make `x86_64-unknown-linux-gnu` use the self-contained linker by default, so we will
        // build our internal lld and use it as the default linker, by setting the `rust.lld` config
        // to true by default:
        // - on the `x86_64-unknown-linux-gnu` target
        // - when building our in-tree llvm (i.e. the target has not set an `llvm-config`), so that
        //   we're also able to build the corresponding lld
        // - or when using an external llvm that's downloaded from CI, which also contains our prebuilt
        //   lld
        // - otherwise, we'd be using an external llvm, and lld would not necessarily available and
        //   thus, disabled
        // - similarly, lld will not be built nor used by default when explicitly asked not to, e.g.
        //   when the config sets `rust.lld = false`
        let lld_enabled = if default_lld_opt_in_targets().contains(&host_target.triple.to_string())
            && hosts == [host_target]
        {
            let no_llvm_config =
                target_config.get(&host_target).is_none_or(|config| config.llvm_config.is_none());
            rust_lld_enabled.unwrap_or(llvm_from_ci || no_llvm_config)
        } else {
            rust_lld_enabled.unwrap_or(false)
        };

        if llvm_from_ci {
            let warn = |option: &str| {
                println!(
                    "WARNING: `{option}` will only be used on `compiler/rustc_llvm` build, not for the LLVM build."
                );
                println!(
                    "HELP: To use `{option}` for LLVM builds, set `download-ci-llvm` option to false."
                );
            };

            if llvm_static_libstdcpp.is_some() {
                warn("static-libstdcpp");
            }

            if llvm_link_shared.is_some() {
                warn("link-shared");
            }

            // FIXME(#129153): instead of all the ad-hoc `download-ci-llvm` checks that follow,
            // use the `builder-config` present in tarballs since #128822 to compare the local
            // config to the ones used to build the LLVM artifacts on CI, and only notify users
            // if they've chosen a different value.

            if llvm_libzstd.is_some() {
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

        if llvm_from_ci {
            let triple = &host_target.triple;
            let ci_llvm_bin = ci_llvm_root(&dwn_ctx, llvm_from_ci, &out).join("bin");
            let build_target =
                target_config.entry(host_target).or_insert_with(|| Target::from_triple(triple));
            check_ci_llvm!(build_target.llvm_config);
            check_ci_llvm!(build_target.llvm_filecheck);
            build_target.llvm_config = Some(ci_llvm_bin.join(exe("llvm-config", host_target)));
            build_target.llvm_filecheck = Some(ci_llvm_bin.join(exe("FileCheck", host_target)));
        }

        let initial_rustfmt = build_rustfmt.or_else(|| maybe_download_rustfmt(&dwn_ctx, &out));

        if matches!(rust_lld_mode.unwrap_or_default(), LldMode::SelfContained)
            && !lld_enabled
            && flags_stage.unwrap_or(0) > 0
        {
            panic!(
                "Trying to use self-contained lld as a linker, but LLD is not being added to the sysroot. Enable it with rust.lld = true."
            );
        }

        if lld_enabled && is_system_llvm(&dwn_ctx, &target_config, llvm_from_ci, host_target) {
            panic!("Cannot enable LLD with `rust.lld = true` when using external llvm-config.");
        }

        let download_rustc = download_rustc_commit.is_some();

        let stage = match flags_cmd {
            Subcommand::Check { .. } => flags_stage.or(build_check_stage).unwrap_or(1),
            Subcommand::Clippy { .. } | Subcommand::Fix => {
                flags_stage.or(build_check_stage).unwrap_or(1)
            }
            // `download-rustc` only has a speed-up for stage2 builds. Default to stage2 unless explicitly overridden.
            Subcommand::Doc { .. } => {
                flags_stage.or(build_doc_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Build { .. } => {
                flags_stage.or(build_build_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Test { .. } | Subcommand::Miri { .. } => {
                flags_stage.or(build_test_stage).unwrap_or(if download_rustc { 2 } else { 1 })
            }
            Subcommand::Bench { .. } => flags_stage.or(build_bench_stage).unwrap_or(2),
            Subcommand::Dist => flags_stage.or(build_dist_stage).unwrap_or(2),
            Subcommand::Install => flags_stage.or(build_install_stage).unwrap_or(2),
            Subcommand::Perf { .. } => flags_stage.unwrap_or(1),
            // Most of the run commands execute bootstrap tools, which don't depend on the compiler.
            // Other commands listed here should always use bootstrap tools.
            Subcommand::Clean { .. }
            | Subcommand::Run { .. }
            | Subcommand::Setup { .. }
            | Subcommand::Format { .. }
            | Subcommand::Vendor { .. } => flags_stage.unwrap_or(0),
        };

        let local_rebuild = build_local_rebuild.unwrap_or(false);

        let check_stage0 = |kind: &str| {
            if local_rebuild {
                eprintln!("WARNING: running {kind} in stage 0. This might not work as expected.");
            } else {
                eprintln!(
                    "ERROR: cannot {kind} anything on stage 0. Use at least stage 1 or set build.local-rebuild=true and use a stage0 compiler built from in-tree sources."
                );
                exit!(1);
            }
        };

        // Now check that the selected stage makes sense, and if not, print an error and end
        match (stage, &flags_cmd) {
            (0, Subcommand::Build { .. }) => {
                check_stage0("build");
            }
            (0, Subcommand::Check { .. }) => {
                check_stage0("check");
            }
            (0, Subcommand::Doc { .. }) => {
                check_stage0("doc");
            }
            (0, Subcommand::Clippy { .. }) => {
                check_stage0("clippy");
            }
            (0, Subcommand::Dist) => {
                check_stage0("dist");
            }
            (0, Subcommand::Install) => {
                check_stage0("install");
            }
            (0, Subcommand::Test { .. }) if build_compiletest_allow_stage0 != Some(true) => {
                eprintln!(
                    "ERROR: cannot test anything on stage 0. Use at least stage 1. If you want to run compiletest with an external stage0 toolchain, enable `build.compiletest-allow-stage0`."
                );
                exit!(1);
            }
            _ => {}
        }

        if flags_compile_time_deps && !matches!(flags_cmd, Subcommand::Check { .. }) {
            eprintln!("ERROR: Can't use --compile-time-deps with any subcommand other than check.");
            exit!(1);
        }

        // CI should always run stage 2 builds, unless it specifically states otherwise
        #[cfg(not(test))]
        if flags_stage.is_none() && is_running_on_ci {
            match flags_cmd {
                Subcommand::Test { .. }
                | Subcommand::Miri { .. }
                | Subcommand::Doc { .. }
                | Subcommand::Build { .. }
                | Subcommand::Bench { .. }
                | Subcommand::Dist
                | Subcommand::Install => {
                    assert_eq!(
                        stage, 2,
                        "x.py should be run with `--stage 2` on CI, but was run with `--stage {stage}`",
                    );
                }
                Subcommand::Clean { .. }
                | Subcommand::Check { .. }
                | Subcommand::Clippy { .. }
                | Subcommand::Fix
                | Subcommand::Run { .. }
                | Subcommand::Setup { .. }
                | Subcommand::Format { .. }
                | Subcommand::Vendor { .. }
                | Subcommand::Perf { .. } => {}
            }
        }

        let with_defaults = |debuginfo_level_specific: Option<_>| {
            debuginfo_level_specific.or(rust_debuginfo_level).unwrap_or(
                if rust_debug == Some(true) {
                    DebuginfoLevel::Limited
                } else {
                    DebuginfoLevel::None
                },
            )
        };

        let ccache = match build_ccache {
            Some(StringOrBool::String(s)) => Some(s),
            Some(StringOrBool::Bool(true)) => Some("ccache".to_string()),
            _ => None,
        };

        let explicit_stage_from_config = build_test_stage.is_some()
            || build_build_stage.is_some()
            || build_doc_stage.is_some()
            || build_dist_stage.is_some()
            || build_install_stage.is_some()
            || build_check_stage.is_some()
            || build_bench_stage.is_some();

        let deny_warnings = match flags_warnings {
            Warnings::Deny => true,
            Warnings::Warn => false,
            Warnings::Default => rust_deny_warnings.unwrap_or(true),
        };

        let gcc_ci_mode = match gcc_download_ci_gcc {
            Some(value) => match value {
                true => GccCiMode::DownloadFromCi,
                false => GccCiMode::BuildLocally,
            },
            None => GccCiMode::default(),
        };

        let targets = flags_target
            .map(|TargetSelectionList(targets)| targets)
            .or_else(|| {
                build_target.map(|t| t.iter().map(|t| TargetSelection::from_user(t)).collect())
            })
            .unwrap_or_else(|| hosts.clone());

        #[allow(clippy::map_identity)]
        let skip = flags_skip
            .into_iter()
            .chain(flags_exclude)
            .chain(build_exclude.unwrap_or_default())
            .map(|p| {
                // Never return top-level path here as it would break `--skip`
                // logic on rustc's internal test framework which is utilized by compiletest.
                #[cfg(windows)]
                {
                    PathBuf::from(p.to_string_lossy().replace('/', "\\"))
                }
                #[cfg(not(windows))]
                {
                    p
                }
            })
            .collect();

        let cargo_info = git_info(&exec_ctx, omit_git_hash, &src.join("src/tools/cargo"));
        let clippy_info = git_info(&exec_ctx, omit_git_hash, &src.join("src/tools/clippy"));
        let in_tree_gcc_info = git_info(&exec_ctx, false, &src.join("src/gcc"));
        let in_tree_llvm_info = git_info(&exec_ctx, false, &src.join("src/llvm-project"));
        let enzyme_info = git_info(&exec_ctx, omit_git_hash, &src.join("src/tools/enzyme"));
        let miri_info = git_info(&exec_ctx, omit_git_hash, &src.join("src/tools/miri"));
        let rust_analyzer_info =
            git_info(&exec_ctx, omit_git_hash, &src.join("src/tools/rust-analyzer"));
        let rustfmt_info = git_info(&exec_ctx, omit_git_hash, &src.join("src/tools/rustfmt"));

        let optimized_compiler_builtins =
            build_optimized_compiler_builtins.unwrap_or(if channel == "dev" {
                CompilerBuiltins::BuildRustOnly
            } else {
                CompilerBuiltins::BuildLLVMFuncs
            });
        let vendor = build_vendor.unwrap_or(
            rust_info.is_from_tarball()
                && src.join("vendor").exists()
                && src.join(".cargo/config.toml").exists(),
        );
        let verbose_tests = rust_verbose_tests.unwrap_or(exec_ctx.is_verbose());

        Config {
            // tidy-alphabetical-start
            android_ndk: build_android_ndk,
            backtrace: rust_backtrace.unwrap_or(true),
            backtrace_on_ice: rust_backtrace_on_ice.unwrap_or(false),
            bindir: install_bindir.map(PathBuf::from).unwrap_or("bin".into()),
            bootstrap_cache_path: build_bootstrap_cache_path,
            bypass_bootstrap_lock: flags_bypass_bootstrap_lock,
            cargo_info,
            cargo_native_static: build_cargo_native_static.unwrap_or(false),
            ccache,
            change_id: toml.change_id.inner,
            channel,
            clippy_info,
            cmd: flags_cmd,
            codegen_tests: rust_codegen_tests.unwrap_or(true),
            color: flags_color,
            compile_time_deps: flags_compile_time_deps,
            compiler_docs: build_compiler_docs.unwrap_or(false),
            compiletest_allow_stage0: build_compiletest_allow_stage0.unwrap_or(false),
            compiletest_diff_tool: build_compiletest_diff_tool,
            compiletest_use_stage0_libtest: build_compiletest_use_stage0_libtest.unwrap_or(true),
            config: toml_path,
            configure_args: build_configure_args.unwrap_or_default(),
            control_flow_guard: rust_control_flow_guard.unwrap_or(false),
            datadir: install_datadir.map(PathBuf::from),
            deny_warnings,
            description: build_description,
            dist_compression_formats,
            dist_compression_profile: dist_compression_profile.unwrap_or("fast".into()),
            dist_include_mingw_linker: dist_include_mingw_linker.unwrap_or(true),
            dist_sign_folder: dist_sign_folder.map(PathBuf::from),
            dist_upload_addr,
            dist_vendor: dist_vendor.unwrap_or_else(|| {
                // If we're building from git or tarball sources, enable it by default.
                rust_info.is_managed_git_subrepository() || rust_info.is_from_tarball()
            }),
            docdir: install_docdir.map(PathBuf::from),
            docs: build_docs.unwrap_or(true),
            docs_minification: build_docs_minification.unwrap_or(true),
            download_rustc_commit,
            dump_bootstrap_shims: flags_dump_bootstrap_shims,
            ehcont_guard: rust_ehcont_guard.unwrap_or(false),
            enable_bolt_settings: flags_enable_bolt_settings,
            enzyme_info,
            exec_ctx,
            explicit_stage_from_cli: flags_stage.is_some(),
            explicit_stage_from_config,
            extended: build_extended.unwrap_or(false),
            free_args: flags_free_args,
            full_bootstrap: build_full_bootstrap.unwrap_or(false),
            gcc_ci_mode,
            gdb: build_gdb.map(PathBuf::from),
            host_target,
            hosts,
            in_tree_gcc_info,
            in_tree_llvm_info,
            include_default_paths: flags_include_default_paths,
            incremental: flags_incremental || rust_incremental == Some(true),
            initial_cargo,
            initial_cargo_clippy: build_cargo_clippy,
            initial_rustc,
            initial_rustfmt,
            initial_sysroot,
            is_running_on_ci,
            jemalloc: rust_jemalloc.unwrap_or(false),
            jobs: Some(threads_from_config(flags_jobs.or(build_jobs).unwrap_or(0))),
            json_output: flags_json_output,
            keep_stage: flags_keep_stage,
            keep_stage_std: flags_keep_stage_std,
            libdir: install_libdir.map(PathBuf::from),
            library_docs_private_items: build_library_docs_private_items.unwrap_or(false),
            lld_enabled,
            lld_mode: rust_lld_mode.unwrap_or_default(),
            lldb: build_lldb.map(PathBuf::from),
            llvm_allow_old_toolchain: llvm_allow_old_toolchain.unwrap_or(false),
            llvm_assertions,
            llvm_bitcode_linker_enabled: rust_llvm_bitcode_linker.unwrap_or(false),
            llvm_build_config: llvm_build_config.clone().unwrap_or(Default::default()),
            llvm_cflags,
            llvm_clang: llvm_clang.unwrap_or(false),
            llvm_clang_cl,
            llvm_cxxflags,
            llvm_enable_warnings: llvm_enable_warnings.unwrap_or(false),
            llvm_enzyme: llvm_enzyme.unwrap_or(false),
            llvm_experimental_targets,
            llvm_from_ci,
            llvm_ldflags,
            llvm_libunwind_default: rust_llvm_libunwind
                .map(|v| v.parse().expect("failed to parse rust.llvm-libunwind")),
            llvm_libzstd: llvm_libzstd.unwrap_or(false),
            llvm_link_jobs,
            // If we're building with ThinLTO on, by default we want to link
            // to LLVM shared, to avoid re-doing ThinLTO (which happens in
            // the link step) with each stage.
            llvm_link_shared: Cell::new(
                llvm_link_shared
                    .or((!llvm_from_ci && llvm_thin_lto.unwrap_or(false)).then_some(true)),
            ),
            llvm_offload: llvm_offload.unwrap_or(false),
            llvm_optimize: llvm_optimize.unwrap_or(true),
            llvm_plugins: llvm_plugin.unwrap_or(false),
            llvm_polly: llvm_polly.unwrap_or(false),
            llvm_profile_generate: flags_llvm_profile_generate,
            llvm_profile_use: flags_llvm_profile_use,
            llvm_release_debuginfo: llvm_release_debuginfo.unwrap_or(false),
            llvm_static_stdcpp: llvm_static_libstdcpp.unwrap_or(false),
            llvm_targets,
            llvm_tests: llvm_tests.unwrap_or(false),
            llvm_thin_lto: llvm_thin_lto.unwrap_or(false),
            llvm_tools_enabled: rust_llvm_tools.unwrap_or(true),
            llvm_use_libcxx: llvm_use_libcxx.unwrap_or(false),
            llvm_use_linker,
            llvm_version_suffix,
            local_rebuild,
            locked_deps: build_locked_deps.unwrap_or(false),
            low_priority: build_low_priority.unwrap_or(false),
            mandir: install_mandir.map(PathBuf::from),
            miri_info,
            musl_root: rust_musl_root.map(PathBuf::from),
            ninja_in_file: llvm_ninja.unwrap_or(true),
            nodejs: build_nodejs.map(PathBuf::from),
            npm: build_npm.map(PathBuf::from),
            omit_git_hash,
            on_fail: flags_on_fail,
            optimized_compiler_builtins,
            out,
            patch_binaries_for_nix: build_patch_binaries_for_nix,
            path_modification_cache,
            paths: flags_paths,
            prefix: install_prefix.map(PathBuf::from),
            print_step_rusage: build_print_step_rusage.unwrap_or(false),
            print_step_timings: build_print_step_timings.unwrap_or(false),
            profiler: build_profiler.unwrap_or(false),
            python: build_python.map(PathBuf::from),
            reproducible_artifacts: flags_reproducible_artifact,
            reuse: build_reuse.map(PathBuf::from),
            rust_analyzer_info,
            rust_break_on_ice: rust_break_on_ice.unwrap_or(true),
            rust_codegen_backends: rust_codegen_backends
                .map(|backends| parse_codegen_backends(backends, "rust"))
                .unwrap_or(vec![CodegenBackendKind::Llvm]),
            rust_codegen_units: rust_codegen_units.map(threads_from_config),
            rust_codegen_units_std: rust_codegen_units_std.map(threads_from_config),
            rust_debug_logging: rust_debug_logging
                .or(rust_rustc_debug_assertions)
                .unwrap_or(rust_debug == Some(true)),
            rust_debuginfo_level_rustc: with_defaults(rust_debuginfo_level_rustc),
            rust_debuginfo_level_std: with_defaults(rust_debuginfo_level_std),
            rust_debuginfo_level_tests: rust_debuginfo_level_tests.unwrap_or(DebuginfoLevel::None),
            rust_debuginfo_level_tools: with_defaults(rust_debuginfo_level_tools),
            rust_dist_src: dist_src_tarball.unwrap_or_else(|| rust_dist_src.unwrap_or(true)),
            rust_frame_pointers: rust_frame_pointers.unwrap_or(false),
            rust_info,
            rust_lto: rust_lto
                .as_deref()
                .map(|value| RustcLto::from_str(value).unwrap())
                .unwrap_or_default(),
            rust_new_symbol_mangling,
            rust_optimize: rust_optimize.unwrap_or(RustOptimize::Bool(true)),
            rust_optimize_tests: rust_optimize_tests.unwrap_or(true),
            rust_overflow_checks: rust_overflow_checks.unwrap_or(rust_debug == Some(true)),
            rust_overflow_checks_std: rust_overflow_checks_std
                .or(rust_overflow_checks)
                .unwrap_or(rust_debug == Some(true)),
            rust_profile_generate: flags_rust_profile_generate.or(rust_profile_generate),
            rust_profile_use: flags_rust_profile_use.or(rust_profile_use),
            rust_randomize_layout: rust_randomize_layout.unwrap_or(false),
            rust_remap_debuginfo: rust_remap_debuginfo.unwrap_or(false),
            rust_rpath: rust_rpath.unwrap_or(true),
            rust_stack_protector,
            rust_std_features: rust_std_features
                .unwrap_or(BTreeSet::from([String::from("panic-unwind")])),
            rust_strip: rust_strip.unwrap_or(false),
            rust_thin_lto_import_instr_limit,
            rust_validate_mir_opts,
            rust_verify_llvm_ir: rust_verify_llvm_ir.unwrap_or(false),
            rustc_debug_assertions: rust_rustc_debug_assertions.unwrap_or(rust_debug == Some(true)),
            rustc_default_linker: rust_default_linker,
            rustc_error_format: flags_rustc_error_format,
            rustfmt_info,
            sanitizers: build_sanitizers.unwrap_or(false),
            save_toolstates: rust_save_toolstates.map(PathBuf::from),
            skip,
            skip_std_check_if_no_download_rustc: flags_skip_std_check_if_no_download_rustc,
            src,
            stage,
            stage0_metadata,
            std_debug_assertions: rust_std_debug_assertions
                .or(rust_rustc_debug_assertions)
                .unwrap_or(rust_debug == Some(true)),
            stderr_is_tty: std::io::stderr().is_terminal(),
            stdout_is_tty: std::io::stdout().is_terminal(),
            submodules: build_submodules,
            sysconfdir: install_sysconfdir.map(PathBuf::from),
            target_config,
            targets,
            test_compare_mode: rust_test_compare_mode.unwrap_or(false),
            tidy_extra_checks: build_tidy_extra_checks,
            tool: build_tool.unwrap_or_default(),
            tools: build_tools,
            tools_debug_assertions: rust_tools_debug_assertions
                .or(rust_rustc_debug_assertions)
                .unwrap_or(rust_debug == Some(true)),
            vendor,
            verbose_tests,
            // tidy-alphabetical-end
        }
    }

    pub fn dry_run(&self) -> bool {
        self.exec_ctx.dry_run()
    }

    pub fn is_explicit_stage(&self) -> bool {
        self.explicit_stage_from_cli || self.explicit_stage_from_config
    }

    pub(crate) fn test_args(&self) -> Vec<&str> {
        let mut test_args = match self.cmd {
            Subcommand::Test { ref test_args, .. }
            | Subcommand::Bench { ref test_args, .. }
            | Subcommand::Miri { ref test_args, .. } => {
                test_args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => vec![],
        };
        test_args.extend(self.free_args.iter().map(|s| s.as_str()));
        test_args
    }

    pub(crate) fn args(&self) -> Vec<&str> {
        let mut args = match self.cmd {
            Subcommand::Run { ref args, .. } => {
                args.iter().flat_map(|s| s.split_whitespace()).collect()
            }
            _ => vec![],
        };
        args.extend(self.free_args.iter().map(|s| s.as_str()));
        args
    }

    /// Returns the content of the given file at a specific commit.
    pub(crate) fn read_file_by_commit(&self, file: &Path, commit: &str) -> String {
        let dwn_ctx = DownloadContext::from(self);
        read_file_by_commit(dwn_ctx, &self.rust_info, file, commit)
    }

    /// Bootstrap embeds a version number into the name of shared libraries it uploads in CI.
    /// Return the version it would have used for the given commit.
    pub(crate) fn artifact_version_part(&self, commit: &str) -> String {
        let (channel, version) = if self.rust_info.is_managed_git_subrepository() {
            let channel =
                self.read_file_by_commit(Path::new("src/ci/channel"), commit).trim().to_owned();
            let version =
                self.read_file_by_commit(Path::new("src/version"), commit).trim().to_owned();
            (channel, version)
        } else {
            let channel = fs::read_to_string(self.src.join("src/ci/channel"));
            let version = fs::read_to_string(self.src.join("src/version"));
            match (channel, version) {
                (Ok(channel), Ok(version)) => {
                    (channel.trim().to_owned(), version.trim().to_owned())
                }
                (channel, version) => {
                    let src = self.src.display();
                    eprintln!("ERROR: failed to determine artifact channel and/or version");
                    eprintln!(
                        "HELP: consider using a git checkout or ensure these files are readable"
                    );
                    if let Err(channel) = channel {
                        eprintln!("reading {src}/src/ci/channel failed: {channel:?}");
                    }
                    if let Err(version) = version {
                        eprintln!("reading {src}/src/version failed: {version:?}");
                    }
                    panic!();
                }
            }
        };

        match channel.as_str() {
            "stable" => version,
            "beta" => channel,
            "nightly" => channel,
            other => unreachable!("{:?} is not recognized as a valid channel", other),
        }
    }

    /// Try to find the relative path of `bindir`, otherwise return it in full.
    pub fn bindir_relative(&self) -> &Path {
        let bindir = &self.bindir;
        if bindir.is_absolute() {
            // Try to make it relative to the prefix.
            if let Some(prefix) = &self.prefix
                && let Ok(stripped) = bindir.strip_prefix(prefix)
            {
                return stripped;
            }
        }
        bindir
    }

    /// Try to find the relative path of `libdir`.
    pub fn libdir_relative(&self) -> Option<&Path> {
        let libdir = self.libdir.as_ref()?;
        if libdir.is_relative() {
            Some(libdir)
        } else {
            // Try to make it relative to the prefix.
            libdir.strip_prefix(self.prefix.as_ref()?).ok()
        }
    }

    /// The absolute path to the downloaded LLVM artifacts.
    pub(crate) fn ci_llvm_root(&self) -> PathBuf {
        let dwn_ctx = DownloadContext::from(self);
        ci_llvm_root(dwn_ctx, self.llvm_from_ci, &self.out)
    }

    /// Directory where the extracted `rustc-dev` component is stored.
    pub(crate) fn ci_rustc_dir(&self) -> PathBuf {
        assert!(self.download_rustc());
        self.out.join(self.host_target).join("ci-rustc")
    }

    /// Determine whether llvm should be linked dynamically.
    ///
    /// If `false`, llvm should be linked statically.
    /// This is computed on demand since LLVM might have to first be downloaded from CI.
    pub(crate) fn llvm_link_shared(&self) -> bool {
        let mut opt = self.llvm_link_shared.get();
        if opt.is_none() && self.dry_run() {
            // just assume static for now - dynamic linking isn't supported on all platforms
            return false;
        }

        let llvm_link_shared = *opt.get_or_insert_with(|| {
            if self.llvm_from_ci {
                self.maybe_download_ci_llvm();
                let ci_llvm = self.ci_llvm_root();
                let link_type = t!(
                    std::fs::read_to_string(ci_llvm.join("link-type.txt")),
                    format!("CI llvm missing: {}", ci_llvm.display())
                );
                link_type == "dynamic"
            } else {
                // unclear how thought-through this default is, but it maintains compatibility with
                // previous behavior
                false
            }
        });
        self.llvm_link_shared.set(opt);
        llvm_link_shared
    }

    /// Return whether we will use a downloaded, pre-compiled version of rustc, or just build from source.
    pub(crate) fn download_rustc(&self) -> bool {
        self.download_rustc_commit().is_some()
    }

    pub(crate) fn download_rustc_commit(&self) -> Option<&str> {
        static DOWNLOAD_RUSTC: OnceLock<Option<String>> = OnceLock::new();
        if self.dry_run() && DOWNLOAD_RUSTC.get().is_none() {
            // avoid trying to actually download the commit
            return self.download_rustc_commit.as_deref();
        }

        DOWNLOAD_RUSTC
            .get_or_init(|| match &self.download_rustc_commit {
                None => None,
                Some(commit) => {
                    self.download_ci_rustc(commit);

                    // CI-rustc can't be used without CI-LLVM. If `self.llvm_from_ci` is false, it means the "if-unchanged"
                    // logic has detected some changes in the LLVM submodule (download-ci-llvm=false can't happen here as
                    // we don't allow it while parsing the configuration).
                    if !self.llvm_from_ci {
                        // This happens when LLVM submodule is updated in CI, we should disable ci-rustc without an error
                        // to not break CI. For non-CI environments, we should return an error.
                        if self.is_running_on_ci {
                            println!("WARNING: LLVM submodule has changes, `download-rustc` will be disabled.");
                            return None;
                        } else {
                            panic!("ERROR: LLVM submodule has changes, `download-rustc` can't be used.");
                        }
                    }

                    if let Some(config_path) = &self.config {
                        let ci_config_toml = match self.get_builder_toml("ci-rustc") {
                            Ok(ci_config_toml) => ci_config_toml,
                            Err(e) if e.to_string().contains("unknown field") => {
                                println!("WARNING: CI rustc has some fields that are no longer supported in bootstrap; download-rustc will be disabled.");
                                println!("HELP: Consider rebasing to a newer commit if available.");
                                return None;
                            },
                            Err(e) => {
                                eprintln!("ERROR: Failed to parse CI rustc bootstrap.toml: {e}");
                                exit!(2);
                            },
                        };

                        let current_config_toml = Self::get_toml(config_path).unwrap();

                        // Check the config compatibility
                        // FIXME: this doesn't cover `--set` flags yet.
                        let res = check_incompatible_options_for_ci_rustc(
                            self.host_target,
                            current_config_toml,
                            ci_config_toml,
                        );

                        // Primarily used by CI runners to avoid handling download-rustc incompatible
                        // options one by one on shell scripts.
                        let disable_ci_rustc_if_incompatible = env::var_os("DISABLE_CI_RUSTC_IF_INCOMPATIBLE")
                            .is_some_and(|s| s == "1" || s == "true");

                        if disable_ci_rustc_if_incompatible && res.is_err() {
                            println!("WARNING: download-rustc is disabled with `DISABLE_CI_RUSTC_IF_INCOMPATIBLE` env.");
                            return None;
                        }

                        res.unwrap();
                    }

                    Some(commit.clone())
                }
            })
            .as_deref()
    }

    /// Runs a function if verbosity is greater than 0
    pub fn verbose(&self, f: impl Fn()) {
        self.exec_ctx.verbose(f);
    }

    pub fn any_sanitizers_to_build(&self) -> bool {
        self.target_config
            .iter()
            .any(|(ts, t)| !ts.is_msvc() && t.sanitizers.unwrap_or(self.sanitizers))
    }

    pub fn any_profiler_enabled(&self) -> bool {
        self.target_config.values().any(|t| matches!(&t.profiler, Some(p) if p.is_string_or_true()))
            || self.profiler
    }

    /// Returns whether or not submodules should be managed by bootstrap.
    pub fn submodules(&self) -> bool {
        // If not specified in config, the default is to only manage
        // submodules if we're currently inside a git repository.
        self.submodules.unwrap_or(self.rust_info.is_managed_git_subrepository())
    }

    pub fn git_config(&self) -> GitConfig<'_> {
        GitConfig {
            nightly_branch: &self.stage0_metadata.config.nightly_branch,
            git_merge_commit_email: &self.stage0_metadata.config.git_merge_commit_email,
        }
    }

    /// Given a path to the directory of a submodule, update it.
    ///
    /// `relative_path` should be relative to the root of the git repository, not an absolute path.
    ///
    /// This *does not* update the submodule if `bootstrap.toml` explicitly says
    /// not to, or if we're not in a git repository (like a plain source
    /// tarball). Typically [`crate::Build::require_submodule`] should be
    /// used instead to provide a nice error to the user if the submodule is
    /// missing.
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "Config::update_submodule",
            skip_all,
            fields(relative_path = ?relative_path),
        ),
    )]
    pub(crate) fn update_submodule(&self, relative_path: &str) {
        let dwn_ctx = DownloadContext::from(self);
        update_submodule(dwn_ctx, &self.rust_info, relative_path);
    }

    /// Returns true if any of the `paths` have been modified locally.
    pub fn has_changes_from_upstream(&self, paths: &[&'static str]) -> bool {
        let dwn_ctx = DownloadContext::from(self);
        has_changes_from_upstream(dwn_ctx, paths)
    }

    /// Checks whether any of the given paths have been modified w.r.t. upstream.
    pub fn check_path_modifications(&self, paths: &[&'static str]) -> PathFreshness {
        // Checking path modifications through git can be relatively expensive (>100ms).
        // We do not assume that the sources would change during bootstrap's execution,
        // so we can cache the results here.
        // Note that we do not use a static variable for the cache, because it would cause problems
        // in tests that create separate `Config` instsances.
        self.path_modification_cache
            .lock()
            .unwrap()
            .entry(paths.to_vec())
            .or_insert_with(|| {
                check_path_modifications(&self.src, &self.git_config(), paths, CiEnv::current())
                    .unwrap()
            })
            .clone()
    }

    pub fn sanitizers_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).and_then(|t| t.sanitizers).unwrap_or(self.sanitizers)
    }

    pub fn needs_sanitizer_runtime_built(&self, target: TargetSelection) -> bool {
        // MSVC uses the Microsoft-provided sanitizer runtime, but all other runtimes we build.
        !target.is_msvc() && self.sanitizers_enabled(target)
    }

    pub fn profiler_path(&self, target: TargetSelection) -> Option<&str> {
        match self.target_config.get(&target)?.profiler.as_ref()? {
            StringOrBool::String(s) => Some(s),
            StringOrBool::Bool(_) => None,
        }
    }

    pub fn profiler_enabled(&self, target: TargetSelection) -> bool {
        self.target_config
            .get(&target)
            .and_then(|t| t.profiler.as_ref())
            .map(StringOrBool::is_string_or_true)
            .unwrap_or(self.profiler)
    }

    /// Returns codegen backends that should be:
    /// - Built and added to the sysroot when we build the compiler.
    /// - Distributed when `x dist` is executed (if the codegen backend has a dist step).
    pub fn enabled_codegen_backends(&self, target: TargetSelection) -> &[CodegenBackendKind] {
        self.target_config
            .get(&target)
            .and_then(|cfg| cfg.codegen_backends.as_deref())
            .unwrap_or(&self.rust_codegen_backends)
    }

    /// Returns the codegen backend that should be configured as the *default* codegen backend
    /// for a rustc compiled by bootstrap.
    pub fn default_codegen_backend(&self, target: TargetSelection) -> &CodegenBackendKind {
        // We're guaranteed to have always at least one codegen backend listed.
        self.enabled_codegen_backends(target).first().unwrap()
    }

    pub fn jemalloc(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).and_then(|cfg| cfg.jemalloc).unwrap_or(self.jemalloc)
    }

    pub fn rpath_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).and_then(|t| t.rpath).unwrap_or(self.rust_rpath)
    }

    pub fn optimized_compiler_builtins(&self, target: TargetSelection) -> &CompilerBuiltins {
        self.target_config
            .get(&target)
            .and_then(|t| t.optimized_compiler_builtins.as_ref())
            .unwrap_or(&self.optimized_compiler_builtins)
    }

    pub fn llvm_enabled(&self, target: TargetSelection) -> bool {
        self.enabled_codegen_backends(target).contains(&CodegenBackendKind::Llvm)
    }

    pub fn llvm_libunwind(&self, target: TargetSelection) -> LlvmLibunwind {
        self.target_config
            .get(&target)
            .and_then(|t| t.llvm_libunwind)
            .or(self.llvm_libunwind_default)
            .unwrap_or(if target.contains("fuchsia") {
                LlvmLibunwind::InTree
            } else {
                LlvmLibunwind::No
            })
    }

    pub fn split_debuginfo(&self, target: TargetSelection) -> SplitDebuginfo {
        self.target_config
            .get(&target)
            .and_then(|t| t.split_debuginfo)
            .unwrap_or_else(|| SplitDebuginfo::default_for_platform(target))
    }

    /// Checks if the given target is the same as the host target.
    pub fn is_host_target(&self, target: TargetSelection) -> bool {
        self.host_target == target
    }

    /// Returns `true` if this is an external version of LLVM not managed by bootstrap.
    /// In particular, we expect llvm sources to be available when this is false.
    ///
    /// NOTE: this is not the same as `!is_rust_llvm` when `llvm_has_patches` is set.
    pub fn is_system_llvm(&self, target: TargetSelection) -> bool {
        let dwn_ctx = DownloadContext::from(self);
        is_system_llvm(dwn_ctx, &self.target_config, self.llvm_from_ci, target)
    }

    /// Returns `true` if this is our custom, patched, version of LLVM.
    ///
    /// This does not necessarily imply that we're managing the `llvm-project` submodule.
    pub fn is_rust_llvm(&self, target: TargetSelection) -> bool {
        match self.target_config.get(&target) {
            // We're using a user-controlled version of LLVM. The user has explicitly told us whether the version has our patches.
            // (They might be wrong, but that's not a supported use-case.)
            // In particular, this tries to support `submodules = false` and `patches = false`, for using a newer version of LLVM that's not through `rust-lang/llvm-project`.
            Some(Target { llvm_has_rust_patches: Some(patched), .. }) => *patched,
            // The user hasn't promised the patches match.
            // This only has our patches if it's downloaded from CI or built from source.
            _ => !self.is_system_llvm(target),
        }
    }

    pub fn exec_ctx(&self) -> &ExecutionContext {
        &self.exec_ctx
    }

    pub fn git_info(&self, omit_git_hash: bool, dir: &Path) -> GitInfo {
        GitInfo::new(omit_git_hash, dir, self)
    }
}

impl AsRef<ExecutionContext> for Config {
    fn as_ref(&self) -> &ExecutionContext {
        &self.exec_ctx
    }
}

fn compute_src_directory(src_dir: Option<PathBuf>, exec_ctx: &ExecutionContext) -> Option<PathBuf> {
    if let Some(src) = src_dir {
        return Some(src);
    } else {
        // Infer the source directory. This is non-trivial because we want to support a downloaded bootstrap binary,
        // running on a completely different machine from where it was compiled.
        let mut cmd = helpers::git(None);
        // NOTE: we cannot support running from outside the repository because the only other path we have available
        // is set at compile time, which can be wrong if bootstrap was downloaded rather than compiled locally.
        // We still support running outside the repository if we find we aren't in a git directory.

        // NOTE: We get a relative path from git to work around an issue on MSYS/mingw. If we used an absolute path,
        // and end up using MSYS's git rather than git-for-windows, we would get a unix-y MSYS path. But as bootstrap
        // has already been (kinda-cross-)compiled to Windows land, we require a normal Windows path.
        cmd.arg("rev-parse").arg("--show-cdup");
        // Discard stderr because we expect this to fail when building from a tarball.
        let output = cmd.allow_failure().run_capture_stdout(exec_ctx);
        if output.is_success() {
            let git_root_relative = output.stdout();
            // We need to canonicalize this path to make sure it uses backslashes instead of forward slashes,
            // and to resolve any relative components.
            let git_root = env::current_dir()
                .unwrap()
                .join(PathBuf::from(git_root_relative.trim()))
                .canonicalize()
                .unwrap();
            let s = git_root.to_str().unwrap();

            // Bootstrap is quite bad at handling /? in front of paths
            let git_root = match s.strip_prefix("\\\\?\\") {
                Some(p) => PathBuf::from(p),
                None => git_root,
            };
            // If this doesn't have at least `stage0`, we guessed wrong. This can happen when,
            // for example, the build directory is inside of another unrelated git directory.
            // In that case keep the original `CARGO_MANIFEST_DIR` handling.
            //
            // NOTE: this implies that downloadable bootstrap isn't supported when the build directory is outside
            // the source directory. We could fix that by setting a variable from all three of python, ./x, and x.ps1.
            if git_root.join("src").join("stage0").exists() {
                return Some(git_root);
            }
        } else {
            // We're building from a tarball, not git sources.
            // We don't support pre-downloaded bootstrap in this case.
        }
    };
    None
}

/// Loads bootstrap TOML config and returns the config together with a path from where
/// it was loaded.
/// `src` is the source root directory, and `config_path` is an optionally provided path to the
/// config.
fn load_toml_config(
    src: &Path,
    config_path: Option<PathBuf>,
    get_toml: &impl Fn(&Path) -> Result<TomlConfig, toml::de::Error>,
) -> (TomlConfig, Option<PathBuf>) {
    // Locate the configuration file using the following priority (first match wins):
    // 1. `--config <path>` (explicit flag)
    // 2. `RUST_BOOTSTRAP_CONFIG` environment variable
    // 3. `./bootstrap.toml` (local file)
    // 4. `<root>/bootstrap.toml`
    // 5. `./config.toml` (fallback for backward compatibility)
    // 6. `<root>/config.toml`
    let toml_path = config_path.or_else(|| env::var_os("RUST_BOOTSTRAP_CONFIG").map(PathBuf::from));
    let using_default_path = toml_path.is_none();
    let mut toml_path = toml_path.unwrap_or_else(|| PathBuf::from("bootstrap.toml"));

    if using_default_path && !toml_path.exists() {
        toml_path = src.join(PathBuf::from("bootstrap.toml"));
        if !toml_path.exists() {
            toml_path = PathBuf::from("config.toml");
            if !toml_path.exists() {
                toml_path = src.join(PathBuf::from("config.toml"));
            }
        }
    }

    // Give a hard error if `--config` or `RUST_BOOTSTRAP_CONFIG` are set to a missing path,
    // but not if `bootstrap.toml` hasn't been created.
    if !using_default_path || toml_path.exists() {
        let path = Some(if cfg!(not(test)) {
            toml_path = toml_path.canonicalize().unwrap();
            toml_path.clone()
        } else {
            toml_path.clone()
        });
        (
            get_toml(&toml_path).unwrap_or_else(|e| {
                eprintln!("ERROR: Failed to parse '{}': {e}", toml_path.display());
                exit!(2);
            }),
            path,
        )
    } else {
        (TomlConfig::default(), None)
    }
}

fn postprocess_toml(
    toml: &mut TomlConfig,
    src_dir: &Path,
    toml_path: Option<PathBuf>,
    exec_ctx: &ExecutionContext,
    override_set: &[String],
    get_toml: &impl Fn(&Path) -> Result<TomlConfig, toml::de::Error>,
) {
    let git_info = GitInfo::new(false, src_dir, exec_ctx);

    if git_info.is_from_tarball() && toml.profile.is_none() {
        toml.profile = Some("dist".into());
    }

    // Reverse the list to ensure the last added config extension remains the most dominant.
    // For example, given ["a.toml", "b.toml"], "b.toml" should take precedence over "a.toml".
    //
    // This must be handled before applying the `profile` since `include`s should always take
    // precedence over `profile`s.
    for include_path in toml.include.clone().unwrap_or_default().iter().rev() {
        let include_path = toml_path
            .as_ref()
            .expect("include found in default TOML config")
            .parent()
            .unwrap()
            .join(include_path);

        let included_toml = get_toml(&include_path).unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to parse '{}': {e}", include_path.display());
            exit!(2);
        });
        toml.merge(
            Some(include_path),
            &mut Default::default(),
            included_toml,
            ReplaceOpt::IgnoreDuplicate,
        );
    }

    if let Some(include) = &toml.profile {
        // Allows creating alias for profile names, allowing
        // profiles to be renamed while maintaining back compatibility
        // Keep in sync with `profile_aliases` in bootstrap.py
        let profile_aliases = HashMap::from([("user", "dist")]);
        let include = match profile_aliases.get(include.as_str()) {
            Some(alias) => alias,
            None => include.as_str(),
        };
        let mut include_path = PathBuf::from(src_dir);
        include_path.push("src");
        include_path.push("bootstrap");
        include_path.push("defaults");
        include_path.push(format!("bootstrap.{include}.toml"));
        let included_toml = get_toml(&include_path).unwrap_or_else(|e| {
            eprintln!(
                "ERROR: Failed to parse default config profile at '{}': {e}",
                include_path.display()
            );
            exit!(2);
        });
        toml.merge(
            Some(include_path),
            &mut Default::default(),
            included_toml,
            ReplaceOpt::IgnoreDuplicate,
        );
    }

    let mut override_toml = TomlConfig::default();
    for option in override_set.iter() {
        fn get_table(option: &str) -> Result<TomlConfig, toml::de::Error> {
            toml::from_str(option).and_then(|table: toml::Value| TomlConfig::deserialize(table))
        }

        let mut err = match get_table(option) {
            Ok(v) => {
                override_toml.merge(None, &mut Default::default(), v, ReplaceOpt::ErrorOnDuplicate);
                continue;
            }
            Err(e) => e,
        };
        // We want to be able to set string values without quotes,
        // like in `configure.py`. Try adding quotes around the right hand side
        if let Some((key, value)) = option.split_once('=')
            && !value.contains('"')
        {
            match get_table(&format!(r#"{key}="{value}""#)) {
                Ok(v) => {
                    override_toml.merge(
                        None,
                        &mut Default::default(),
                        v,
                        ReplaceOpt::ErrorOnDuplicate,
                    );
                    continue;
                }
                Err(e) => err = e,
            }
        }
        eprintln!("failed to parse override `{option}`: `{err}");
        exit!(2)
    }
    toml.merge(None, &mut Default::default(), override_toml, ReplaceOpt::Override);
}

#[cfg(test)]
pub fn check_stage0_version(
    _program_path: &Path,
    _component_name: &'static str,
    _src_dir: &Path,
    _exec_ctx: &ExecutionContext,
) {
}

/// check rustc/cargo version is same or lower with 1 apart from the building one
#[cfg(not(test))]
pub fn check_stage0_version(
    program_path: &Path,
    component_name: &'static str,
    src_dir: &Path,
    exec_ctx: &ExecutionContext,
) {
    use build_helper::util::fail;

    if exec_ctx.dry_run() {
        return;
    }

    let stage0_output =
        command(program_path).arg("--version").run_capture_stdout(exec_ctx).stdout();
    let mut stage0_output = stage0_output.lines().next().unwrap().split(' ');

    let stage0_name = stage0_output.next().unwrap();
    if stage0_name != component_name {
        fail(&format!(
            "Expected to find {component_name} at {} but it claims to be {stage0_name}",
            program_path.display()
        ));
    }

    let stage0_version =
        semver::Version::parse(stage0_output.next().unwrap().split('-').next().unwrap().trim())
            .unwrap();
    let source_version =
        semver::Version::parse(fs::read_to_string(src_dir.join("src/version")).unwrap().trim())
            .unwrap();
    if !(source_version == stage0_version
        || (source_version.major == stage0_version.major
            && (source_version.minor == stage0_version.minor
                || source_version.minor == stage0_version.minor + 1)))
    {
        let prev_version = format!("{}.{}.x", source_version.major, source_version.minor - 1);
        fail(&format!(
            "Unexpected {component_name} version: {stage0_version}, we should use {prev_version}/{source_version} to build source with {source_version}"
        ));
    }
}

pub fn download_ci_rustc_commit<'a>(
    dwn_ctx: impl AsRef<DownloadContext<'a>>,
    rust_info: &channel::GitInfo,
    download_rustc: Option<StringOrBool>,
    llvm_assertions: bool,
) -> Option<String> {
    let dwn_ctx = dwn_ctx.as_ref();

    if !is_download_ci_available(&dwn_ctx.host_target.triple, llvm_assertions) {
        return None;
    }

    // If `download-rustc` is not set, default to rebuilding.
    let if_unchanged = match download_rustc {
        // Globally default `download-rustc` to `false`, because some contributors don't use
        // profiles for reasons such as:
        // - They need to seamlessly switch between compiler/library work.
        // - They don't want to use compiler profile because they need to override too many
        //   things and it's easier to not use a profile.
        None | Some(StringOrBool::Bool(false)) => return None,
        Some(StringOrBool::Bool(true)) => false,
        Some(StringOrBool::String(s)) if s == "if-unchanged" => {
            if !rust_info.is_managed_git_subrepository() {
                println!(
                    "ERROR: `download-rustc=if-unchanged` is only compatible with Git managed sources."
                );
                crate::exit!(1);
            }

            true
        }
        Some(StringOrBool::String(other)) => {
            panic!("unrecognized option for download-rustc: {other}")
        }
    };

    let commit = if rust_info.is_managed_git_subrepository() {
        // Look for a version to compare to based on the current commit.
        // Only commits merged by bors will have CI artifacts.
        let freshness = check_path_modifications_(dwn_ctx, RUSTC_IF_UNCHANGED_ALLOWED_PATHS);
        dwn_ctx.exec_ctx.verbose(|| {
            eprintln!("rustc freshness: {freshness:?}");
        });
        match freshness {
            PathFreshness::LastModifiedUpstream { upstream } => upstream,
            PathFreshness::HasLocalModifications { upstream } => {
                if if_unchanged {
                    return None;
                }

                if dwn_ctx.is_running_on_ci {
                    eprintln!("CI rustc commit matches with HEAD and we are in CI.");
                    eprintln!(
                        "`rustc.download-ci` functionality will be skipped as artifacts are not available."
                    );
                    return None;
                }

                upstream
            }
            PathFreshness::MissingUpstream => {
                eprintln!("No upstream commit found");
                return None;
            }
        }
    } else {
        channel::read_commit_info_file(dwn_ctx.src)
            .map(|info| info.sha.trim().to_owned())
            .expect("git-commit-info is missing in the project root")
    };

    Some(commit)
}

pub fn check_path_modifications_<'a>(
    dwn_ctx: impl AsRef<DownloadContext<'a>>,
    paths: &[&'static str],
) -> PathFreshness {
    let dwn_ctx = dwn_ctx.as_ref();
    // Checking path modifications through git can be relatively expensive (>100ms).
    // We do not assume that the sources would change during bootstrap's execution,
    // so we can cache the results here.
    // Note that we do not use a static variable for the cache, because it would cause problems
    // in tests that create separate `Config` instsances.
    dwn_ctx
        .path_modification_cache
        .lock()
        .unwrap()
        .entry(paths.to_vec())
        .or_insert_with(|| {
            check_path_modifications(
                dwn_ctx.src,
                &git_config(dwn_ctx.stage0_metadata),
                paths,
                CiEnv::current(),
            )
            .unwrap()
        })
        .clone()
}

pub fn git_config(stage0_metadata: &build_helper::stage0_parser::Stage0) -> GitConfig<'_> {
    GitConfig {
        nightly_branch: &stage0_metadata.config.nightly_branch,
        git_merge_commit_email: &stage0_metadata.config.git_merge_commit_email,
    }
}

pub fn parse_download_ci_llvm<'a>(
    dwn_ctx: impl AsRef<DownloadContext<'a>>,
    rust_info: &channel::GitInfo,
    download_rustc_commit: &Option<String>,
    download_ci_llvm: Option<StringOrBool>,
    asserts: bool,
) -> bool {
    let dwn_ctx = dwn_ctx.as_ref();

    // We don't ever want to use `true` on CI, as we should not
    // download upstream artifacts if there are any local modifications.
    let default = if dwn_ctx.is_running_on_ci {
        StringOrBool::String("if-unchanged".to_string())
    } else {
        StringOrBool::Bool(true)
    };
    let download_ci_llvm = download_ci_llvm.unwrap_or(default);

    let if_unchanged = || {
        if rust_info.is_from_tarball() {
            // Git is needed for running "if-unchanged" logic.
            println!("ERROR: 'if-unchanged' is only compatible with Git managed sources.");
            crate::exit!(1);
        }

        // Fetching the LLVM submodule is unnecessary for self-tests.
        #[cfg(not(test))]
        update_submodule(dwn_ctx, rust_info, "src/llvm-project");

        // Check for untracked changes in `src/llvm-project` and other important places.
        let has_changes = has_changes_from_upstream(dwn_ctx, LLVM_INVALIDATION_PATHS);

        // Return false if there are untracked changes, otherwise check if CI LLVM is available.
        if has_changes {
            false
        } else {
            llvm::is_ci_llvm_available_for_target(&dwn_ctx.host_target, asserts)
        }
    };

    match download_ci_llvm {
        StringOrBool::Bool(b) => {
            if !b && download_rustc_commit.is_some() {
                panic!(
                    "`llvm.download-ci-llvm` cannot be set to `false` if `rust.download-rustc` is set to `true` or `if-unchanged`."
                );
            }

            if b && dwn_ctx.is_running_on_ci {
                // On CI, we must always rebuild LLVM if there were any modifications to it
                panic!(
                    "`llvm.download-ci-llvm` cannot be set to `true` on CI. Use `if-unchanged` instead."
                );
            }

            // If download-ci-llvm=true we also want to check that CI llvm is available
            b && llvm::is_ci_llvm_available_for_target(&dwn_ctx.host_target, asserts)
        }
        StringOrBool::String(s) if s == "if-unchanged" => if_unchanged(),
        StringOrBool::String(other) => {
            panic!("unrecognized option for download-ci-llvm: {other:?}")
        }
    }
}

pub fn has_changes_from_upstream<'a>(
    dwn_ctx: impl AsRef<DownloadContext<'a>>,
    paths: &[&'static str],
) -> bool {
    let dwn_ctx = dwn_ctx.as_ref();
    match check_path_modifications_(dwn_ctx, paths) {
        PathFreshness::LastModifiedUpstream { .. } => false,
        PathFreshness::HasLocalModifications { .. } | PathFreshness::MissingUpstream => true,
    }
}

#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "trace",
        name = "Config::update_submodule",
        skip_all,
        fields(relative_path = ?relative_path),
    ),
)]
pub(crate) fn update_submodule<'a>(
    dwn_ctx: impl AsRef<DownloadContext<'a>>,
    rust_info: &channel::GitInfo,
    relative_path: &str,
) {
    let dwn_ctx = dwn_ctx.as_ref();
    if rust_info.is_from_tarball() || !submodules_(dwn_ctx.submodules, rust_info) {
        return;
    }

    let absolute_path = dwn_ctx.src.join(relative_path);

    // NOTE: This check is required because `jj git clone` doesn't create directories for
    // submodules, they are completely ignored. The code below assumes this directory exists,
    // so create it here.
    if !absolute_path.exists() {
        t!(fs::create_dir_all(&absolute_path));
    }

    // NOTE: The check for the empty directory is here because when running x.py the first time,
    // the submodule won't be checked out. Check it out now so we can build it.
    if !git_info(dwn_ctx.exec_ctx, false, &absolute_path).is_managed_git_subrepository()
        && !helpers::dir_is_empty(&absolute_path)
    {
        return;
    }

    // Submodule updating actually happens during in the dry run mode. We need to make sure that
    // all the git commands below are actually executed, because some follow-up code
    // in bootstrap might depend on the submodules being checked out. Furthermore, not all
    // the command executions below work with an empty output (produced during dry run).
    // Therefore, all commands below are marked with `run_in_dry_run()`, so that they also run in
    // dry run mode.
    let submodule_git = || {
        let mut cmd = helpers::git(Some(&absolute_path));
        cmd.run_in_dry_run();
        cmd
    };

    // Determine commit checked out in submodule.
    let checked_out_hash =
        submodule_git().args(["rev-parse", "HEAD"]).run_capture_stdout(dwn_ctx.exec_ctx).stdout();
    let checked_out_hash = checked_out_hash.trim_end();
    // Determine commit that the submodule *should* have.
    let recorded = helpers::git(Some(dwn_ctx.src))
        .run_in_dry_run()
        .args(["ls-tree", "HEAD"])
        .arg(relative_path)
        .run_capture_stdout(dwn_ctx.exec_ctx)
        .stdout();

    let actual_hash = recorded
        .split_whitespace()
        .nth(2)
        .unwrap_or_else(|| panic!("unexpected output `{recorded}`"));

    if actual_hash == checked_out_hash {
        // already checked out
        return;
    }

    println!("Updating submodule {relative_path}");

    helpers::git(Some(dwn_ctx.src))
        .allow_failure()
        .run_in_dry_run()
        .args(["submodule", "-q", "sync"])
        .arg(relative_path)
        .run(dwn_ctx.exec_ctx);

    // Try passing `--progress` to start, then run git again without if that fails.
    let update = |progress: bool| {
        // Git is buggy and will try to fetch submodules from the tracking branch for *this* repository,
        // even though that has no relation to the upstream for the submodule.
        let current_branch = helpers::git(Some(dwn_ctx.src))
            .allow_failure()
            .run_in_dry_run()
            .args(["symbolic-ref", "--short", "HEAD"])
            .run_capture(dwn_ctx.exec_ctx);

        let mut git = helpers::git(Some(dwn_ctx.src)).allow_failure();
        git.run_in_dry_run();
        if current_branch.is_success() {
            // If there is a tag named after the current branch, git will try to disambiguate by prepending `heads/` to the branch name.
            // This syntax isn't accepted by `branch.{branch}`. Strip it.
            let branch = current_branch.stdout();
            let branch = branch.trim();
            let branch = branch.strip_prefix("heads/").unwrap_or(branch);
            git.arg("-c").arg(format!("branch.{branch}.remote=origin"));
        }
        git.args(["submodule", "update", "--init", "--recursive", "--depth=1"]);
        if progress {
            git.arg("--progress");
        }
        git.arg(relative_path);
        git
    };
    if !update(true).allow_failure().run(dwn_ctx.exec_ctx) {
        update(false).allow_failure().run(dwn_ctx.exec_ctx);
    }

    // Save any local changes, but avoid running `git stash pop` if there are none (since it will exit with an error).
    // diff-index reports the modifications through the exit status
    let has_local_modifications = !submodule_git()
        .allow_failure()
        .args(["diff-index", "--quiet", "HEAD"])
        .run(dwn_ctx.exec_ctx);
    if has_local_modifications {
        submodule_git().allow_failure().args(["stash", "push"]).run(dwn_ctx.exec_ctx);
    }

    submodule_git().allow_failure().args(["reset", "-q", "--hard"]).run(dwn_ctx.exec_ctx);
    submodule_git().allow_failure().args(["clean", "-qdfx"]).run(dwn_ctx.exec_ctx);

    if has_local_modifications {
        submodule_git().allow_failure().args(["stash", "pop"]).run(dwn_ctx.exec_ctx);
    }
}

pub fn git_info(exec_ctx: &ExecutionContext, omit_git_hash: bool, dir: &Path) -> GitInfo {
    GitInfo::new(omit_git_hash, dir, exec_ctx)
}

pub fn submodules_(submodules: &Option<bool>, rust_info: &channel::GitInfo) -> bool {
    // If not specified in config, the default is to only manage
    // submodules if we're currently inside a git repository.
    submodules.unwrap_or(rust_info.is_managed_git_subrepository())
}

/// Returns `true` if this is an external version of LLVM not managed by bootstrap.
/// In particular, we expect llvm sources to be available when this is false.
///
/// NOTE: this is not the same as `!is_rust_llvm` when `llvm_has_patches` is set.
pub fn is_system_llvm<'a>(
    dwn_ctx: impl AsRef<DownloadContext<'a>>,
    target_config: &HashMap<TargetSelection, Target>,
    llvm_from_ci: bool,
    target: TargetSelection,
) -> bool {
    let dwn_ctx = dwn_ctx.as_ref();
    match target_config.get(&target) {
        Some(Target { llvm_config: Some(_), .. }) => {
            let ci_llvm = llvm_from_ci && is_host_target(&dwn_ctx.host_target, &target);
            !ci_llvm
        }
        // We're building from the in-tree src/llvm-project sources.
        Some(Target { llvm_config: None, .. }) => false,
        None => false,
    }
}

pub fn is_host_target(host_target: &TargetSelection, target: &TargetSelection) -> bool {
    host_target == target
}

pub(crate) fn ci_llvm_root<'a>(
    dwn_ctx: impl AsRef<DownloadContext<'a>>,
    llvm_from_ci: bool,
    out: &Path,
) -> PathBuf {
    let dwn_ctx = dwn_ctx.as_ref();
    assert!(llvm_from_ci);
    out.join(dwn_ctx.host_target).join("ci-llvm")
}

/// Returns the content of the given file at a specific commit.
pub(crate) fn read_file_by_commit<'a>(
    dwn_ctx: impl AsRef<DownloadContext<'a>>,
    rust_info: &channel::GitInfo,
    file: &Path,
    commit: &str,
) -> String {
    let dwn_ctx = dwn_ctx.as_ref();
    assert!(
        rust_info.is_managed_git_subrepository(),
        "`Config::read_file_by_commit` is not supported in non-git sources."
    );

    let mut git = helpers::git(Some(dwn_ctx.src));
    git.arg("show").arg(format!("{commit}:{}", file.to_str().unwrap()));
    git.run_capture_stdout(dwn_ctx.exec_ctx).stdout()
}
