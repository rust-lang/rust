//! This module defines the central `Config` struct, which aggregates all components
//! of the bootstrap configuration into a single unit.
//!
//! It serves as the primary public interface for accessing bootstrap's configuration data.
//! This module orchestrates the overall configuration parsing process by leveraging
//! logic from `parsing.rs` and provides top-level methods for initializing (`Config::parse()`)
//! and querying the comprehensive configuration state. It also contains various utility
//! methods that operate on the overall configuration.

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{env, fs};

use build_helper::ci::CiEnv;
use build_helper::exit;
use build_helper::git::{GitConfig, PathFreshness, check_path_modifications, output_result};
#[cfg(feature = "tracing")]
use tracing::{instrument, span};

use crate::core::build_steps::llvm;
use crate::core::build_steps::llvm::LLVM_INVALIDATION_PATHS;
use crate::core::config::flags::Color;
pub use crate::core::config::flags::Subcommand;
use crate::core::config::toml::change_id::ChangeId;
use crate::core::config::toml::common::{
    DebuginfoLevel, DryRun, GccCiMode, LlvmLibunwind, RustcLto, SplitDebuginfo, StringOrBool,
};
use crate::core::config::toml::rust::{
    LldMode, RustOptimize, check_incompatible_options_for_ci_rustc,
};
use crate::core::config::toml::target::Target;
use crate::core::download::is_download_ci_available;
use crate::utils::channel;
use crate::{Command, Flags, GitInfo, OnceLock, TargetSelection, helpers, output, t};

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
    pub sanitizers: bool,
    pub profiler: bool,
    pub omit_git_hash: bool,
    pub skip: Vec<PathBuf>,
    pub include_default_paths: bool,
    pub rustc_error_format: Option<String>,
    pub json_output: bool,
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
    pub dry_run: DryRun,
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

    pub build: TargetSelection,
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

    pub is_running_on_ci: bool,

    /// Cache for determining path modifications
    pub path_modification_cache: Arc<Mutex<HashMap<Vec<&'static str>, PathFreshness>>>,
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
            build: TargetSelection::from_user(env!("BUILD_TRIPLE")),

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

    #[cfg_attr(
        feature = "tracing",
        instrument(target = "CONFIG_HANDLING", level = "trace", name = "Config::parse", skip_all)
    )]
    pub fn parse(flags: Flags) -> Config {
        Self::parse_inner(flags, Self::get_toml)
    }

    pub fn dry_run(&self) -> bool {
        match self.dry_run {
            DryRun::Disabled => false,
            DryRun::SelfCheck | DryRun::UserSelected => true,
        }
    }

    pub fn is_explicit_stage(&self) -> bool {
        self.explicit_stage_from_cli || self.explicit_stage_from_config
    }

    /// Runs a command, printing out nice contextual information if it fails.
    /// Exits if the command failed to execute at all, otherwise returns its
    /// `status.success()`.
    #[deprecated = "use `Builder::try_run` instead where possible"]
    pub(crate) fn try_run(&self, cmd: &mut Command) -> Result<(), ()> {
        if self.dry_run() {
            return Ok(());
        }
        self.verbose(|| println!("running: {cmd:?}"));
        build_helper::util::try_run(cmd, self.is_verbose())
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
        output(git.as_command_mut())
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
            if let Some(prefix) = &self.prefix {
                if let Ok(stripped) = bindir.strip_prefix(prefix) {
                    return stripped;
                }
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
        self.out.join(self.build).join("ci-llvm")
    }

    /// Directory where the extracted `rustc-dev` component is stored.
    pub(crate) fn ci_rustc_dir(&self) -> PathBuf {
        assert!(self.download_rustc());
        self.out.join(self.build).join("ci-rustc")
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
                            self.build,
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
        if self.is_verbose() {
            f()
        }
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
        if !GitInfo::new(false, &absolute_path).is_managed_git_subrepository()
            && !helpers::dir_is_empty(&absolute_path)
        {
            return;
        }

        // Submodule updating actually happens during in the dry run mode. We need to make sure that
        // all the git commands below are actually executed, because some follow-up code
        // in bootstrap might depend on the submodules being checked out. Furthermore, not all
        // the command executions below work with an empty output (produced during dry run).
        // Therefore, all commands below are marked with `run_always()`, so that they also run in
        // dry run mode.
        let submodule_git = || {
            let mut cmd = helpers::git(Some(&absolute_path));
            cmd.run_always();
            cmd
        };

        // Determine commit checked out in submodule.
        let checked_out_hash = output(submodule_git().args(["rev-parse", "HEAD"]).as_command_mut());
        let checked_out_hash = checked_out_hash.trim_end();
        // Determine commit that the submodule *should* have.
        let recorded = output(
            helpers::git(Some(&self.src))
                .run_always()
                .args(["ls-tree", "HEAD"])
                .arg(relative_path)
                .as_command_mut(),
        );

        let actual_hash = recorded
            .split_whitespace()
            .nth(2)
            .unwrap_or_else(|| panic!("unexpected output `{recorded}`"));

        if actual_hash == checked_out_hash {
            // already checked out
            return;
        }

        println!("Updating submodule {relative_path}");
        self.check_run(
            helpers::git(Some(&self.src))
                .run_always()
                .args(["submodule", "-q", "sync"])
                .arg(relative_path),
        );

        // Try passing `--progress` to start, then run git again without if that fails.
        let update = |progress: bool| {
            // Git is buggy and will try to fetch submodules from the tracking branch for *this* repository,
            // even though that has no relation to the upstream for the submodule.
            let current_branch = output_result(
                helpers::git(Some(&self.src))
                    .allow_failure()
                    .run_always()
                    .args(["symbolic-ref", "--short", "HEAD"])
                    .as_command_mut(),
            )
            .map(|b| b.trim().to_owned());

            let mut git = helpers::git(Some(&self.src)).allow_failure();
            git.run_always();
            if let Ok(branch) = current_branch {
                // If there is a tag named after the current branch, git will try to disambiguate by prepending `heads/` to the branch name.
                // This syntax isn't accepted by `branch.{branch}`. Strip it.
                let branch = branch.strip_prefix("heads/").unwrap_or(&branch);
                git.arg("-c").arg(format!("branch.{branch}.remote=origin"));
            }
            git.args(["submodule", "update", "--init", "--recursive", "--depth=1"]);
            if progress {
                git.arg("--progress");
            }
            git.arg(relative_path);
            git
        };
        if !self.check_run(&mut update(true)) {
            self.check_run(&mut update(false));
        }

        // Save any local changes, but avoid running `git stash pop` if there are none (since it will exit with an error).
        // diff-index reports the modifications through the exit status
        let has_local_modifications = !self.check_run(submodule_git().allow_failure().args([
            "diff-index",
            "--quiet",
            "HEAD",
        ]));
        if has_local_modifications {
            self.check_run(submodule_git().args(["stash", "push"]));
        }

        self.check_run(submodule_git().args(["reset", "-q", "--hard"]));
        self.check_run(submodule_git().args(["clean", "-qdfx"]));

        if has_local_modifications {
            self.check_run(submodule_git().args(["stash", "pop"]));
        }
    }

    #[cfg(test)]
    pub fn check_stage0_version(&self, _program_path: &Path, _component_name: &'static str) {}

    /// check rustc/cargo version is same or lower with 1 apart from the building one
    #[cfg(not(test))]
    pub fn check_stage0_version(&self, program_path: &Path, component_name: &'static str) {
        use build_helper::util::fail;

        if self.dry_run() {
            return;
        }

        let stage0_output = output(Command::new(program_path).arg("--version"));
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
        let source_version = semver::Version::parse(
            fs::read_to_string(self.src.join("src/version")).unwrap().trim(),
        )
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

    /// Returns the commit to download, or `None` if we shouldn't download CI artifacts.
    pub fn download_ci_rustc_commit(
        &self,
        download_rustc: Option<StringOrBool>,
        debug_assertions_requested: bool,
        llvm_assertions: bool,
    ) -> Option<String> {
        if !is_download_ci_available(&self.build.triple, llvm_assertions) {
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
                if !self.rust_info.is_managed_git_subrepository() {
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

        // RUSTC_IF_UNCHANGED_ALLOWED_PATHS
        let mut allowed_paths = RUSTC_IF_UNCHANGED_ALLOWED_PATHS.to_vec();

        // In CI, disable ci-rustc if there are changes in the library tree. But for non-CI, allow
        // these changes to speed up the build process for library developers. This provides consistent
        // functionality for library developers between `download-rustc=true` and `download-rustc="if-unchanged"`
        // options.
        //
        // If you update "library" logic here, update `builder::tests::ci_rustc_if_unchanged_logic` test
        // logic accordingly.
        if !self.is_running_on_ci {
            allowed_paths.push(":!library");
        }

        let commit = if self.rust_info.is_managed_git_subrepository() {
            // Look for a version to compare to based on the current commit.
            // Only commits merged by bors will have CI artifacts.
            let freshness = self.check_path_modifications(&allowed_paths);
            self.verbose(|| {
                eprintln!("rustc freshness: {freshness:?}");
            });
            match freshness {
                PathFreshness::LastModifiedUpstream { upstream } => upstream,
                PathFreshness::HasLocalModifications { upstream } => {
                    if if_unchanged {
                        return None;
                    }

                    if self.is_running_on_ci {
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
            channel::read_commit_info_file(&self.src)
                .map(|info| info.sha.trim().to_owned())
                .expect("git-commit-info is missing in the project root")
        };

        if debug_assertions_requested {
            eprintln!(
                "WARN: `rust.debug-assertions = true` will prevent downloading CI rustc as alt CI \
                rustc is not currently built with debug assertions."
            );
            return None;
        }

        Some(commit)
    }

    pub fn parse_download_ci_llvm(
        &self,
        download_ci_llvm: Option<StringOrBool>,
        asserts: bool,
    ) -> bool {
        // We don't ever want to use `true` on CI, as we should not
        // download upstream artifacts if there are any local modifications.
        let default = if self.is_running_on_ci {
            StringOrBool::String("if-unchanged".to_string())
        } else {
            StringOrBool::Bool(true)
        };
        let download_ci_llvm = download_ci_llvm.unwrap_or(default);

        let if_unchanged = || {
            if self.rust_info.is_from_tarball() {
                // Git is needed for running "if-unchanged" logic.
                println!("ERROR: 'if-unchanged' is only compatible with Git managed sources.");
                crate::exit!(1);
            }

            // Fetching the LLVM submodule is unnecessary for self-tests.
            #[cfg(not(test))]
            self.update_submodule("src/llvm-project");

            // Check for untracked changes in `src/llvm-project` and other important places.
            let has_changes = self.has_changes_from_upstream(LLVM_INVALIDATION_PATHS);

            // Return false if there are untracked changes, otherwise check if CI LLVM is available.
            if has_changes { false } else { llvm::is_ci_llvm_available_for_target(self, asserts) }
        };

        match download_ci_llvm {
            StringOrBool::Bool(b) => {
                if !b && self.download_rustc_commit.is_some() {
                    panic!(
                        "`llvm.download-ci-llvm` cannot be set to `false` if `rust.download-rustc` is set to `true` or `if-unchanged`."
                    );
                }

                if b && self.is_running_on_ci {
                    // On CI, we must always rebuild LLVM if there were any modifications to it
                    panic!(
                        "`llvm.download-ci-llvm` cannot be set to `true` on CI. Use `if-unchanged` instead."
                    );
                }

                // If download-ci-llvm=true we also want to check that CI llvm is available
                b && llvm::is_ci_llvm_available_for_target(self, asserts)
            }
            StringOrBool::String(s) if s == "if-unchanged" => if_unchanged(),
            StringOrBool::String(other) => {
                panic!("unrecognized option for download-ci-llvm: {other:?}")
            }
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
        self.build == target
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
}
