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
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{cmp, env, fs};

use build_helper::ci::CiEnv;
use build_helper::exit;
use build_helper::git::{GitConfig, PathFreshness, check_path_modifications};
use serde::Deserialize;
#[cfg(feature = "tracing")]
use tracing::{instrument, span};

pub use crate::core::config::flags::Subcommand;
use crate::core::config::flags::{Color, Flags};
use crate::core::config::toml::TomlConfig;
use crate::core::config::toml::build::Tool;
use crate::core::config::toml::change_id::ChangeId;
use crate::core::config::toml::rust::{
    LldMode, RustOptimize, check_incompatible_options_for_ci_rustc,
};
use crate::core::config::toml::target::Target;
use crate::core::config::{
    DebuginfoLevel, DryRun, GccCiMode, LlvmLibunwind, Merge, ReplaceOpt, RustcLto, SplitDebuginfo,
    StringOrBool, threads_from_config,
};
use crate::utils::channel;
use crate::utils::exec::ExecutionContext;
use crate::utils::helpers::{exe, get_host_target};
use crate::{GitInfo, OnceLock, TargetSelection, check_ci_llvm, helpers, t};

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
    pub verbose: usize,
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
    /// Whether to use the `c` feature of the `compiler_builtins` crate.
    pub optimized_compiler_builtins: bool,

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
    pub rust_codegen_backends: Vec<String>,
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
    #[cfg_attr(
        feature = "tracing",
        instrument(target = "CONFIG_HANDLING", level = "trace", name = "Config::default_opts")
    )]
    pub fn default_opts() -> Config {
        #[cfg(feature = "tracing")]
        span!(target: "CONFIG_HANDLING", tracing::Level::TRACE, "constructing default config");

        Config {
            bypass_bootstrap_lock: false,
            llvm_optimize: true,
            ninja_in_file: true,
            llvm_static_stdcpp: false,
            llvm_libzstd: false,
            backtrace: true,
            rust_optimize: RustOptimize::Bool(true),
            rust_optimize_tests: true,
            rust_randomize_layout: false,
            submodules: None,
            docs: true,
            docs_minification: true,
            rust_rpath: true,
            rust_strip: false,
            channel: "dev".to_string(),
            codegen_tests: true,
            rust_dist_src: true,
            rust_codegen_backends: vec!["llvm".to_owned()],
            deny_warnings: true,
            bindir: "bin".into(),
            dist_include_mingw_linker: true,
            dist_compression_profile: "fast".into(),

            stdout_is_tty: std::io::stdout().is_terminal(),
            stderr_is_tty: std::io::stderr().is_terminal(),

            // set by build.rs
            host_target: get_host_target(),

            src: {
                let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                // Undo `src/bootstrap`
                manifest_dir.parent().unwrap().parent().unwrap().to_owned()
            },
            out: PathBuf::from("build"),

            // This is needed by codegen_ssa on macOS to ship `llvm-objcopy` aliased to
            // `rust-objcopy` to workaround bad `strip`s on macOS.
            llvm_tools_enabled: true,

            ..Default::default()
        }
    }

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
            paths: mut flags_paths,
            set: flags_set,
            free_args: mut flags_free_args,
            ci: flags_ci,
            skip_std_check_if_no_download_rustc: flags_skip_std_check_if_no_download_rustc,
        } = flags;

        let mut config = Config::default_opts();
        let mut exec_ctx = ExecutionContext::new();
        exec_ctx.set_verbose(flags_verbose);
        exec_ctx.set_fail_fast(flags_cmd.fail_fast());

        config.exec_ctx = exec_ctx;

        let read_toml = |path: &Path| {
            get_toml(path).unwrap_or_else(|e| {
                eprintln!("ERROR: Failed to parse '{}': {e}", path.display());
                exit!(2);
            })
        };

        // Set flags.
        config.paths = std::mem::take(&mut flags_paths);

        #[cfg(feature = "tracing")]
        span!(
            target: "CONFIG_HANDLING",
            tracing::Level::TRACE,
            "collecting paths and path exclusions",
            "flags.paths" = ?flags_paths,
            "flags.skip" = ?flags_skip,
            "flags.exclude" = ?flags_exclude
        );

        #[cfg(feature = "tracing")]
        span!(
            target: "CONFIG_HANDLING",
            tracing::Level::TRACE,
            "normalizing and combining `flag.skip`/`flag.exclude` paths",
            "config.skip" = ?config.skip,
        );

        config.include_default_paths = flags_include_default_paths;
        config.rustc_error_format = flags_rustc_error_format;
        config.json_output = flags_json_output;
        config.compile_time_deps = flags_compile_time_deps;
        config.on_fail = flags_on_fail;
        config.cmd = flags_cmd;
        config.incremental = flags_incremental;
        config.set_dry_run(if flags_dry_run { DryRun::UserSelected } else { DryRun::Disabled });
        config.dump_bootstrap_shims = flags_dump_bootstrap_shims;
        config.keep_stage = flags_keep_stage;
        config.keep_stage_std = flags_keep_stage_std;
        config.color = flags_color;
        config.free_args = std::mem::take(&mut flags_free_args);
        config.llvm_profile_use = flags_llvm_profile_use;
        config.llvm_profile_generate = flags_llvm_profile_generate;
        config.enable_bolt_settings = flags_enable_bolt_settings;
        config.bypass_bootstrap_lock = flags_bypass_bootstrap_lock;
        config.is_running_on_ci = flags_ci.unwrap_or(CiEnv::is_ci());
        config.skip_std_check_if_no_download_rustc = flags_skip_std_check_if_no_download_rustc;

        if let Some(flags_jobs) = flags_jobs {
            config.jobs = Some(threads_from_config(flags_jobs));
        }

        // Infer the rest of the configuration.

        if let Some(src) = flags_src {
            config.src = src
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
            let output = cmd.allow_failure().run_capture_stdout(&config);
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
                    config.src = git_root;
                }
            } else {
                // We're building from a tarball, not git sources.
                // We don't support pre-downloaded bootstrap in this case.
            }
        }

        if cfg!(test) {
            // Use the build directory of the original x.py invocation, so that we can set `initial_rustc` properly.
            config.out = Path::new(
                &env::var_os("CARGO_TARGET_DIR").expect("cargo test directly is not supported"),
            )
            .parent()
            .unwrap()
            .to_path_buf();
        }

        config.stage0_metadata = build_helper::stage0_parser::parse_stage0_file();

        // Locate the configuration file using the following priority (first match wins):
        // 1. `--config <path>` (explicit flag)
        // 2. `RUST_BOOTSTRAP_CONFIG` environment variable
        // 3. `./bootstrap.toml` (local file)
        // 4. `<root>/bootstrap.toml`
        // 5. `./config.toml` (fallback for backward compatibility)
        // 6. `<root>/config.toml`
        let toml_path = flags_config
            .clone()
            .or_else(|| env::var_os("RUST_BOOTSTRAP_CONFIG").map(PathBuf::from));
        let using_default_path = toml_path.is_none();
        let mut toml_path = toml_path.unwrap_or_else(|| PathBuf::from("bootstrap.toml"));

        if using_default_path && !toml_path.exists() {
            toml_path = config.src.join(PathBuf::from("bootstrap.toml"));
            if !toml_path.exists() {
                toml_path = PathBuf::from("config.toml");
                if !toml_path.exists() {
                    toml_path = config.src.join(PathBuf::from("config.toml"));
                }
            }
        }

        // Give a hard error if `--config` or `RUST_BOOTSTRAP_CONFIG` are set to a missing path,
        // but not if `bootstrap.toml` hasn't been created.
        let mut toml = if !using_default_path || toml_path.exists() {
            config.config = Some(if cfg!(not(test)) {
                toml_path = toml_path.canonicalize().unwrap();
                toml_path.clone()
            } else {
                toml_path.clone()
            });
            read_toml(&toml_path)
        } else {
            config.config = None;
            TomlConfig::default()
        };

        if cfg!(test) {
            // When configuring bootstrap for tests, make sure to set the rustc and Cargo to the
            // same ones used to call the tests (if custom ones are not defined in the toml). If we
            // don't do that, bootstrap will use its own detection logic to find a suitable rustc
            // and Cargo, which doesn't work when the caller is specìfying a custom local rustc or
            // Cargo in their bootstrap.toml.
            let build = toml.build.get_or_insert_with(Default::default);
            build.rustc = build.rustc.take().or(std::env::var_os("RUSTC").map(|p| p.into()));
            build.cargo = build.cargo.take().or(std::env::var_os("CARGO").map(|p| p.into()));
        }

        if config.git_info(false, &config.src).is_from_tarball() && toml.profile.is_none() {
            toml.profile = Some("dist".into());
        }

        // Reverse the list to ensure the last added config extension remains the most dominant.
        // For example, given ["a.toml", "b.toml"], "b.toml" should take precedence over "a.toml".
        //
        // This must be handled before applying the `profile` since `include`s should always take
        // precedence over `profile`s.
        for include_path in toml.include.clone().unwrap_or_default().iter().rev() {
            let include_path = toml_path.parent().unwrap().join(include_path);

            let included_toml = read_toml(&include_path);
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
            let mut include_path = config.src.clone();
            include_path.push("src");
            include_path.push("bootstrap");
            include_path.push("defaults");
            include_path.push(format!("bootstrap.{include}.toml"));
            let included_toml = read_toml(&include_path);
            toml.merge(
                Some(include_path),
                &mut Default::default(),
                included_toml,
                ReplaceOpt::IgnoreDuplicate,
            );
        }

        let mut override_toml = TomlConfig::default();
        for option in flags_set.iter() {
            fn get_table(option: &str) -> Result<TomlConfig, toml::de::Error> {
                toml::from_str(option).and_then(|table: toml::Value| TomlConfig::deserialize(table))
            }

            let mut err = match get_table(option) {
                Ok(v) => {
                    override_toml.merge(
                        None,
                        &mut Default::default(),
                        v,
                        ReplaceOpt::ErrorOnDuplicate,
                    );
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

        config.change_id = toml.change_id.inner;

        config.verbose = cmp::max(config.verbose, flags_verbose as usize);

        // Verbose flag is a good default for `rust.verbose-tests`.
        config.verbose_tests = config.is_verbose();

        config.llvm_assertions =
            toml.llvm.as_ref().is_some_and(|llvm| llvm.assertions.unwrap_or(false));

        if let Some(flags_build) = flags_build {
            config.host_target = TargetSelection::from_user(&flags_build);
        } else if let Some(Some(build)) = toml.build.as_ref().map(|build| build.build.clone()) {
            config.host_target = TargetSelection::from_user(&build);
        }

        let file_content = t!(fs::read_to_string(config.src.join("src/ci/channel")));
        let ci_channel = file_content.trim_end();

        let toml_channel = toml.rust.as_ref().and_then(|r| r.channel.clone());
        let is_user_configured_rust_channel = match toml_channel {
            Some(channel) if channel == "auto-detect" => {
                config.channel = ci_channel.into();
                true
            }
            Some(channel) => {
                config.channel = channel;
                true
            }
            None => false,
        };

        let default = config.channel == "dev";
        config.omit_git_hash = toml.rust.as_ref().and_then(|r| r.omit_git_hash).unwrap_or(default);

        config.rust_info = config.git_info(config.omit_git_hash, &config.src);
        config.cargo_info =
            config.git_info(config.omit_git_hash, &config.src.join("src/tools/cargo"));
        config.rust_analyzer_info =
            config.git_info(config.omit_git_hash, &config.src.join("src/tools/rust-analyzer"));
        config.clippy_info =
            config.git_info(config.omit_git_hash, &config.src.join("src/tools/clippy"));
        config.miri_info =
            config.git_info(config.omit_git_hash, &config.src.join("src/tools/miri"));
        config.rustfmt_info =
            config.git_info(config.omit_git_hash, &config.src.join("src/tools/rustfmt"));
        config.enzyme_info =
            config.git_info(config.omit_git_hash, &config.src.join("src/tools/enzyme"));
        config.in_tree_llvm_info = config.git_info(false, &config.src.join("src/llvm-project"));
        config.in_tree_gcc_info = config.git_info(false, &config.src.join("src/gcc"));

        toml.rust.as_ref().map(|rust| {
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
            let debug_assertions_requested = matches!(rust.rustc_debug_assertions, Some(true))
                || (matches!(rust.debug, Some(true))
                    && !matches!(rust.rustc_debug_assertions, Some(false)));

            if debug_assertions_requested
                && let Some(ref opt) = rust.download_rustc
                && opt.is_string_or_true()
            {
                eprintln!(
                    "WARN: currently no CI rustc builds have rustc debug assertions \
                            enabled. Please either set `rust.debug-assertions` to `false` if you \
                            want to use download CI rustc or set `rust.download-rustc` to `false`."
                );
            }

            config.download_rustc_commit = config.download_ci_rustc_commit(
                rust.download_rustc.clone(),
                debug_assertions_requested,
                config.llvm_assertions,
            );
        });

        if !is_user_configured_rust_channel && config.rust_info.is_from_tarball() {
            config.channel = ci_channel.into();
        }

        config.rust_profile_use = flags_rust_profile_use;
        config.rust_profile_generate = flags_rust_profile_generate;

        config.reproducible_artifacts = flags_reproducible_artifact;

        // We need to override `rust.channel` if it's manually specified when using the CI rustc.
        // This is because if the compiler uses a different channel than the one specified in bootstrap.toml,
        // tests may fail due to using a different channel than the one used by the compiler during tests.
        if let Some(commit) = &config.download_rustc_commit
            && is_user_configured_rust_channel
        {
            println!(
                "WARNING: `rust.download-rustc` is enabled. The `rust.channel` option will be overridden by the CI rustc's channel."
            );

            let channel =
                config.read_file_by_commit(Path::new("src/ci/channel"), commit).trim().to_owned();

            config.channel = channel;
        }

        config.explicit_stage_from_cli = flags_stage.is_some();

        config.skip.extend(
            flags_skip
                .into_iter()
                .chain(flags_exclude)
                .map(|p| {
                    // Never return top-level path here as it would break `--skip`
                    // logic on rustc's internal test framework which is utilized
                    // by compiletest.
                    if cfg!(windows) {
                        PathBuf::from(p.to_str().unwrap().replace('/', "\\"))
                    } else {
                        p
                    }
                })
                .collect::<Vec<PathBuf>>(),
        );

        config.apply_install_config(toml.install);
        config.apply_gcc_config(toml.gcc);
        config.apply_dist_config(toml.dist);

        config.apply_build_config(
            toml.build,
            flags_skip_stage0_validation,
            flags_stage,
            flags_host,
            flags_target,
            flags_build_dir,
        );
        config.apply_target_config(toml.target);
        config.apply_rust_config(toml.rust, flags_warnings);
        config.apply_llvm_config(toml.llvm);

        if config.llvm_from_ci {
            let triple = &config.host_target.triple;
            let ci_llvm_bin = config.ci_llvm_root().join("bin");
            let build_target = config
                .target_config
                .entry(config.host_target)
                .or_insert_with(|| Target::from_triple(triple));

            check_ci_llvm!(build_target.llvm_config);
            check_ci_llvm!(build_target.llvm_filecheck);
            build_target.llvm_config =
                Some(ci_llvm_bin.join(exe("llvm-config", config.host_target)));
            build_target.llvm_filecheck =
                Some(ci_llvm_bin.join(exe("FileCheck", config.host_target)));
        }

        if matches!(config.lld_mode, LldMode::SelfContained)
            && !config.lld_enabled
            && flags_stage.unwrap_or(0) > 0
        {
            panic!(
                "Trying to use self-contained lld as a linker, but LLD is not being added to the sysroot. Enable it with rust.lld = true."
            );
        }

        if config.lld_enabled && config.is_system_llvm(config.host_target) {
            panic!("Cannot enable LLD with `rust.lld = true` when using external llvm-config.");
        }

        // Now check that the selected stage makes sense, and if not, print a warning and end
        match (config.stage, &config.cmd) {
            (0, Subcommand::Build) => {
                eprintln!("WARNING: cannot build anything on stage 0. Use at least stage 1.");
                exit!(1);
            }
            (0, Subcommand::Check { .. }) => {
                eprintln!("WARNING: cannot check anything on stage 0. Use at least stage 1.");
                exit!(1);
            }
            _ => {}
        }

        // CI should always run stage 2 builds, unless it specifically states otherwise
        #[cfg(not(test))]
        if flags_stage.is_none() && config.is_running_on_ci {
            match config.cmd {
                Subcommand::Test { .. }
                | Subcommand::Miri { .. }
                | Subcommand::Doc { .. }
                | Subcommand::Build
                | Subcommand::Bench { .. }
                | Subcommand::Dist
                | Subcommand::Install => {
                    assert_eq!(
                        config.stage, 2,
                        "x.py should be run with `--stage 2` on CI, but was run with `--stage {}`",
                        config.stage,
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

        config
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
        assert!(
            self.rust_info.is_managed_git_subrepository(),
            "`Config::read_file_by_commit` is not supported in non-git sources."
        );

        let mut git = helpers::git(Some(&self.src));
        git.arg("show").arg(format!("{commit}:{}", file.to_str().unwrap()));
        git.run_capture_stdout(self).stdout()
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
        assert!(self.llvm_from_ci);
        self.out.join(self.host_target).join("ci-llvm")
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
        if self.rust_info.is_from_tarball() || !self.submodules() {
            return;
        }

        let absolute_path = self.src.join(relative_path);

        // NOTE: This check is required because `jj git clone` doesn't create directories for
        // submodules, they are completely ignored. The code below assumes this directory exists,
        // so create it here.
        if !absolute_path.exists() {
            t!(fs::create_dir_all(&absolute_path));
        }

        // NOTE: The check for the empty directory is here because when running x.py the first time,
        // the submodule won't be checked out. Check it out now so we can build it.
        if !self.git_info(false, &absolute_path).is_managed_git_subrepository()
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
            submodule_git().args(["rev-parse", "HEAD"]).run_capture_stdout(self).stdout();
        let checked_out_hash = checked_out_hash.trim_end();
        // Determine commit that the submodule *should* have.
        let recorded = helpers::git(Some(&self.src))
            .run_in_dry_run()
            .args(["ls-tree", "HEAD"])
            .arg(relative_path)
            .run_capture_stdout(self)
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

        helpers::git(Some(&self.src))
            .allow_failure()
            .run_in_dry_run()
            .args(["submodule", "-q", "sync"])
            .arg(relative_path)
            .run(self);

        // Try passing `--progress` to start, then run git again without if that fails.
        let update = |progress: bool| {
            // Git is buggy and will try to fetch submodules from the tracking branch for *this* repository,
            // even though that has no relation to the upstream for the submodule.
            let current_branch = helpers::git(Some(&self.src))
                .allow_failure()
                .run_in_dry_run()
                .args(["symbolic-ref", "--short", "HEAD"])
                .run_capture(self);

            let mut git = helpers::git(Some(&self.src)).allow_failure();
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
        if !update(true).allow_failure().run(self) {
            update(false).allow_failure().run(self);
        }

        // Save any local changes, but avoid running `git stash pop` if there are none (since it will exit with an error).
        // diff-index reports the modifications through the exit status
        let has_local_modifications =
            !submodule_git().allow_failure().args(["diff-index", "--quiet", "HEAD"]).run(self);
        if has_local_modifications {
            submodule_git().allow_failure().args(["stash", "push"]).run(self);
        }

        submodule_git().allow_failure().args(["reset", "-q", "--hard"]).run(self);
        submodule_git().allow_failure().args(["clean", "-qdfx"]).run(self);

        if has_local_modifications {
            submodule_git().allow_failure().args(["stash", "pop"]).run(self);
        }
    }

    /// Returns true if any of the `paths` have been modified locally.
    pub fn has_changes_from_upstream(&self, paths: &[&'static str]) -> bool {
        match self.check_path_modifications(paths) {
            PathFreshness::LastModifiedUpstream { .. } => false,
            PathFreshness::HasLocalModifications { .. } | PathFreshness::MissingUpstream => true,
        }
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

    pub fn ci_env(&self) -> CiEnv {
        if self.is_running_on_ci { CiEnv::GitHubActions } else { CiEnv::None }
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

    pub fn codegen_backends(&self, target: TargetSelection) -> &[String] {
        self.target_config
            .get(&target)
            .and_then(|cfg| cfg.codegen_backends.as_deref())
            .unwrap_or(&self.rust_codegen_backends)
    }

    pub fn jemalloc(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).and_then(|cfg| cfg.jemalloc).unwrap_or(self.jemalloc)
    }

    pub fn default_codegen_backend(&self, target: TargetSelection) -> Option<String> {
        self.codegen_backends(target).first().cloned()
    }

    pub fn rpath_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).and_then(|t| t.rpath).unwrap_or(self.rust_rpath)
    }

    pub fn optimized_compiler_builtins(&self, target: TargetSelection) -> bool {
        self.target_config
            .get(&target)
            .and_then(|t| t.optimized_compiler_builtins)
            .unwrap_or(self.optimized_compiler_builtins)
    }

    pub fn llvm_enabled(&self, target: TargetSelection) -> bool {
        self.codegen_backends(target).contains(&"llvm".to_owned())
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
        match self.target_config.get(&target) {
            Some(Target { llvm_config: Some(_), .. }) => {
                let ci_llvm = self.llvm_from_ci && self.is_host_target(target);
                !ci_llvm
            }
            // We're building from the in-tree src/llvm-project sources.
            Some(Target { llvm_config: None, .. }) => false,
            None => false,
        }
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
