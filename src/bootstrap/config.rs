//! Serialized configuration of a build.
//!
//! This module implements parsing `config.toml` configuration files to tweak
//! how the build runs.

use std::cmp;
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::cache::{Interned, INTERNER};
pub use crate::flags::Subcommand;
use crate::flags::{Color, Flags};
use crate::util::exe;
use build_helper::t;
use merge::Merge;
use serde::Deserialize;

macro_rules! check_ci_llvm {
    ($name:expr) => {
        assert!(
            $name.is_none(),
            "setting {} is incompatible with download-ci-llvm.",
            stringify!($name)
        );
    };
}

/// Global configuration for the entire build and/or bootstrap.
///
/// This structure is derived from a combination of both `config.toml` and
/// `config.mk`. As of the time of this writing it's unlikely that `config.toml`
/// is used all that much, so this is primarily filled out by `config.mk` which
/// is generated from `./configure`.
///
/// Note that this structure is not decoded directly into, but rather it is
/// filled out from the decoded forms of the structs below. For documentation
/// each field, see the corresponding fields in
/// `config.toml.example`.
#[derive(Default)]
pub struct Config {
    pub changelog_seen: Option<usize>,
    pub ccache: Option<String>,
    /// Call Build::ninja() instead of this.
    pub ninja_in_file: bool,
    pub verbose: usize,
    pub submodules: bool,
    pub fast_submodules: bool,
    pub compiler_docs: bool,
    pub docs: bool,
    pub locked_deps: bool,
    pub vendor: bool,
    pub target_config: HashMap<TargetSelection, Target>,
    pub full_bootstrap: bool,
    pub extended: bool,
    pub tools: Option<HashSet<String>>,
    pub sanitizers: bool,
    pub profiler: bool,
    pub ignore_git: bool,
    pub exclude: Vec<PathBuf>,
    pub include_default_paths: bool,
    pub rustc_error_format: Option<String>,
    pub json_output: bool,
    pub test_compare_mode: bool,
    pub llvm_libunwind: Option<LlvmLibunwind>,
    pub color: Color,

    pub on_fail: Option<String>,
    pub stage: u32,
    pub keep_stage: Vec<u32>,
    pub keep_stage_std: Vec<u32>,
    pub src: PathBuf,
    // defaults to `config.toml`
    pub config: PathBuf,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,
    pub dry_run: bool,

    pub deny_warnings: bool,
    pub backtrace_on_ice: bool,

    // llvm codegen options
    pub llvm_skip_rebuild: bool,
    pub llvm_assertions: bool,
    pub llvm_optimize: bool,
    pub llvm_thin_lto: bool,
    pub llvm_release_debuginfo: bool,
    pub llvm_version_check: bool,
    pub llvm_static_stdcpp: bool,
    pub llvm_link_shared: bool,
    pub llvm_clang_cl: Option<String>,
    pub llvm_targets: Option<String>,
    pub llvm_experimental_targets: Option<String>,
    pub llvm_link_jobs: Option<u32>,
    pub llvm_version_suffix: Option<String>,
    pub llvm_use_linker: Option<String>,
    pub llvm_allow_old_toolchain: Option<bool>,
    pub llvm_polly: Option<bool>,
    pub llvm_from_ci: bool,

    pub use_lld: bool,
    pub lld_enabled: bool,
    pub llvm_tools_enabled: bool,

    pub llvm_cflags: Option<String>,
    pub llvm_cxxflags: Option<String>,
    pub llvm_ldflags: Option<String>,
    pub llvm_use_libcxx: bool,

    // rust codegen options
    pub rust_optimize: bool,
    pub rust_codegen_units: Option<u32>,
    pub rust_codegen_units_std: Option<u32>,
    pub rust_debug_assertions: bool,
    pub rust_debug_assertions_std: bool,
    pub rust_debug_logging: bool,
    pub rust_debuginfo_level_rustc: u32,
    pub rust_debuginfo_level_std: u32,
    pub rust_debuginfo_level_tools: u32,
    pub rust_debuginfo_level_tests: u32,
    pub rust_run_dsymutil: bool,
    pub rust_rpath: bool,
    pub rustc_parallel: bool,
    pub rustc_default_linker: Option<String>,
    pub rust_optimize_tests: bool,
    pub rust_dist_src: bool,
    pub rust_codegen_backends: Vec<Interned<String>>,
    pub rust_verify_llvm_ir: bool,
    pub rust_thin_lto_import_instr_limit: Option<u32>,
    pub rust_remap_debuginfo: bool,
    pub rust_new_symbol_mangling: bool,
    pub rust_profile_use: Option<String>,
    pub rust_profile_generate: Option<String>,

    pub build: TargetSelection,
    pub hosts: Vec<TargetSelection>,
    pub targets: Vec<TargetSelection>,
    pub local_rebuild: bool,
    pub jemalloc: bool,
    pub control_flow_guard: bool,

    // dist misc
    pub dist_sign_folder: Option<PathBuf>,
    pub dist_upload_addr: Option<String>,
    pub dist_gpg_password_file: Option<PathBuf>,
    pub dist_compression_formats: Option<Vec<String>>,

    // libstd features
    pub backtrace: bool, // support for RUST_BACKTRACE

    // misc
    pub low_priority: bool,
    pub channel: String,
    pub description: Option<String>,
    pub verbose_tests: bool,
    pub save_toolstates: Option<PathBuf>,
    pub print_step_timings: bool,
    pub missing_tools: bool,

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
    pub gdb: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub cargo_native_static: bool,
    pub configure_args: Vec<String>,

    // These are either the stage0 downloaded binaries or the locally installed ones.
    pub initial_cargo: PathBuf,
    pub initial_rustc: PathBuf,
    pub initial_rustfmt: Option<PathBuf>,
    pub out: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LlvmLibunwind {
    No,
    InTree,
    System,
}

impl Default for LlvmLibunwind {
    fn default() -> Self {
        Self::No
    }
}

impl FromStr for LlvmLibunwind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "no" => Ok(Self::No),
            "in-tree" => Ok(Self::InTree),
            "system" => Ok(Self::System),
            invalid => Err(format!("Invalid value '{}' for rust.llvm-libunwind config.", invalid)),
        }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetSelection {
    pub triple: Interned<String>,
    file: Option<Interned<String>>,
}

impl TargetSelection {
    pub fn from_user(selection: &str) -> Self {
        let path = Path::new(selection);

        let (triple, file) = if path.exists() {
            let triple = path
                .file_stem()
                .expect("Target specification file has no file stem")
                .to_str()
                .expect("Target specification file stem is not UTF-8");

            (triple, Some(selection))
        } else {
            (selection, None)
        };

        let triple = INTERNER.intern_str(triple);
        let file = file.map(|f| INTERNER.intern_str(f));

        Self { triple, file }
    }

    pub fn rustc_target_arg(&self) -> &str {
        self.file.as_ref().unwrap_or(&self.triple)
    }

    pub fn contains(&self, needle: &str) -> bool {
        self.triple.contains(needle)
    }

    pub fn starts_with(&self, needle: &str) -> bool {
        self.triple.starts_with(needle)
    }

    pub fn ends_with(&self, needle: &str) -> bool {
        self.triple.ends_with(needle)
    }
}

impl fmt::Display for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.triple)?;
        if let Some(file) = self.file {
            write!(f, "({})", file)?;
        }
        Ok(())
    }
}

impl PartialEq<&str> for TargetSelection {
    fn eq(&self, other: &&str) -> bool {
        self.triple == *other
    }
}

/// Per-target configuration stored in the global configuration structure.
#[derive(Default)]
pub struct Target {
    /// Some(path to llvm-config) if using an external LLVM.
    pub llvm_config: Option<PathBuf>,
    /// Some(path to FileCheck) if one was specified.
    pub llvm_filecheck: Option<PathBuf>,
    pub cc: Option<PathBuf>,
    pub cxx: Option<PathBuf>,
    pub ar: Option<PathBuf>,
    pub ranlib: Option<PathBuf>,
    pub linker: Option<PathBuf>,
    pub ndk: Option<PathBuf>,
    pub sanitizers: Option<bool>,
    pub profiler: Option<bool>,
    pub crt_static: Option<bool>,
    pub musl_root: Option<PathBuf>,
    pub musl_libdir: Option<PathBuf>,
    pub wasi_root: Option<PathBuf>,
    pub qemu_rootfs: Option<PathBuf>,
    pub no_std: bool,
}

impl Target {
    pub fn from_triple(triple: &str) -> Self {
        let mut target: Self = Default::default();
        if triple.contains("-none") || triple.contains("nvptx") {
            target.no_std = true;
        }
        target
    }
}
/// Structure of the `config.toml` file that configuration is read from.
///
/// This structure uses `Decodable` to automatically decode a TOML configuration
/// file into this format, and then this is traversed and written into the above
/// `Config` structure.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct TomlConfig {
    changelog_seen: Option<usize>,
    build: Option<Build>,
    install: Option<Install>,
    llvm: Option<Llvm>,
    rust: Option<Rust>,
    target: Option<HashMap<String, TomlTarget>>,
    dist: Option<Dist>,
    profile: Option<String>,
}

impl Merge for TomlConfig {
    fn merge(
        &mut self,
        TomlConfig { build, install, llvm, rust, dist, target, profile: _, changelog_seen: _ }: Self,
    ) {
        fn do_merge<T: Merge>(x: &mut Option<T>, y: Option<T>) {
            if let Some(new) = y {
                if let Some(original) = x {
                    original.merge(new);
                } else {
                    *x = Some(new);
                }
            }
        };
        do_merge(&mut self.build, build);
        do_merge(&mut self.install, install);
        do_merge(&mut self.llvm, llvm);
        do_merge(&mut self.rust, rust);
        do_merge(&mut self.dist, dist);
        assert!(target.is_none(), "merging target-specific config is not currently supported");
    }
}

/// TOML representation of various global build decisions.
#[derive(Deserialize, Default, Clone, Merge)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Build {
    build: Option<String>,
    host: Option<Vec<String>>,
    target: Option<Vec<String>>,
    // This is ignored, the rust code always gets the build directory from the `BUILD_DIR` env variable
    build_dir: Option<String>,
    cargo: Option<String>,
    rustc: Option<String>,
    rustfmt: Option<PathBuf>,
    docs: Option<bool>,
    compiler_docs: Option<bool>,
    submodules: Option<bool>,
    fast_submodules: Option<bool>,
    gdb: Option<String>,
    nodejs: Option<String>,
    python: Option<String>,
    locked_deps: Option<bool>,
    vendor: Option<bool>,
    full_bootstrap: Option<bool>,
    extended: Option<bool>,
    tools: Option<HashSet<String>>,
    verbose: Option<usize>,
    sanitizers: Option<bool>,
    profiler: Option<bool>,
    cargo_native_static: Option<bool>,
    low_priority: Option<bool>,
    configure_args: Option<Vec<String>>,
    local_rebuild: Option<bool>,
    print_step_timings: Option<bool>,
    doc_stage: Option<u32>,
    build_stage: Option<u32>,
    test_stage: Option<u32>,
    install_stage: Option<u32>,
    dist_stage: Option<u32>,
    bench_stage: Option<u32>,
}

/// TOML representation of various global install decisions.
#[derive(Deserialize, Default, Clone, Merge)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Install {
    prefix: Option<String>,
    sysconfdir: Option<String>,
    docdir: Option<String>,
    bindir: Option<String>,
    libdir: Option<String>,
    mandir: Option<String>,
    datadir: Option<String>,

    // standard paths, currently unused
    infodir: Option<String>,
    localstatedir: Option<String>,
}

/// TOML representation of how the LLVM build is configured.
#[derive(Deserialize, Default, Merge)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Llvm {
    skip_rebuild: Option<bool>,
    optimize: Option<bool>,
    thin_lto: Option<bool>,
    release_debuginfo: Option<bool>,
    assertions: Option<bool>,
    ccache: Option<StringOrBool>,
    version_check: Option<bool>,
    static_libstdcpp: Option<bool>,
    ninja: Option<bool>,
    targets: Option<String>,
    experimental_targets: Option<String>,
    link_jobs: Option<u32>,
    link_shared: Option<bool>,
    version_suffix: Option<String>,
    clang_cl: Option<String>,
    cflags: Option<String>,
    cxxflags: Option<String>,
    ldflags: Option<String>,
    use_libcxx: Option<bool>,
    use_linker: Option<String>,
    allow_old_toolchain: Option<bool>,
    polly: Option<bool>,
    download_ci_llvm: Option<StringOrBool>,
}

#[derive(Deserialize, Default, Clone, Merge)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Dist {
    sign_folder: Option<String>,
    gpg_password_file: Option<String>,
    upload_addr: Option<String>,
    src_tarball: Option<bool>,
    missing_tools: Option<bool>,
    compression_formats: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrBool {
    String(String),
    Bool(bool),
}

impl Default for StringOrBool {
    fn default() -> StringOrBool {
        StringOrBool::Bool(false)
    }
}

/// TOML representation of how the Rust build is configured.
#[derive(Deserialize, Default, Merge)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Rust {
    optimize: Option<bool>,
    debug: Option<bool>,
    codegen_units: Option<u32>,
    codegen_units_std: Option<u32>,
    debug_assertions: Option<bool>,
    debug_assertions_std: Option<bool>,
    debug_logging: Option<bool>,
    debuginfo_level: Option<u32>,
    debuginfo_level_rustc: Option<u32>,
    debuginfo_level_std: Option<u32>,
    debuginfo_level_tools: Option<u32>,
    debuginfo_level_tests: Option<u32>,
    run_dsymutil: Option<bool>,
    backtrace: Option<bool>,
    incremental: Option<bool>,
    parallel_compiler: Option<bool>,
    default_linker: Option<String>,
    channel: Option<String>,
    description: Option<String>,
    musl_root: Option<String>,
    rpath: Option<bool>,
    verbose_tests: Option<bool>,
    optimize_tests: Option<bool>,
    codegen_tests: Option<bool>,
    ignore_git: Option<bool>,
    dist_src: Option<bool>,
    save_toolstates: Option<String>,
    codegen_backends: Option<Vec<String>>,
    lld: Option<bool>,
    use_lld: Option<bool>,
    llvm_tools: Option<bool>,
    deny_warnings: Option<bool>,
    backtrace_on_ice: Option<bool>,
    verify_llvm_ir: Option<bool>,
    thin_lto_import_instr_limit: Option<u32>,
    remap_debuginfo: Option<bool>,
    jemalloc: Option<bool>,
    test_compare_mode: Option<bool>,
    llvm_libunwind: Option<String>,
    control_flow_guard: Option<bool>,
    new_symbol_mangling: Option<bool>,
    profile_generate: Option<String>,
    profile_use: Option<String>,
}

/// TOML representation of how each build target is configured.
#[derive(Deserialize, Default, Merge)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct TomlTarget {
    cc: Option<String>,
    cxx: Option<String>,
    ar: Option<String>,
    ranlib: Option<String>,
    linker: Option<String>,
    llvm_config: Option<String>,
    llvm_filecheck: Option<String>,
    android_ndk: Option<String>,
    sanitizers: Option<bool>,
    profiler: Option<bool>,
    crt_static: Option<bool>,
    musl_root: Option<String>,
    musl_libdir: Option<String>,
    wasi_root: Option<String>,
    qemu_rootfs: Option<String>,
    no_std: Option<bool>,
}

impl Config {
    fn path_from_python(var_key: &str) -> PathBuf {
        match env::var_os(var_key) {
            Some(var_val) => Self::normalize_python_path(var_val),
            _ => panic!("expected '{}' to be set", var_key),
        }
    }

    /// Normalizes paths from Python slightly. We don't trust paths from Python (#49785).
    fn normalize_python_path(path: OsString) -> PathBuf {
        Path::new(&path).components().collect()
    }

    pub fn default_opts() -> Config {
        let mut config = Config::default();
        config.llvm_optimize = true;
        config.ninja_in_file = true;
        config.llvm_version_check = true;
        config.backtrace = true;
        config.rust_optimize = true;
        config.rust_optimize_tests = true;
        config.submodules = true;
        config.fast_submodules = true;
        config.docs = true;
        config.rust_rpath = true;
        config.channel = "dev".to_string();
        config.codegen_tests = true;
        config.ignore_git = false;
        config.rust_dist_src = true;
        config.rust_codegen_backends = vec![INTERNER.intern_str("llvm")];
        config.deny_warnings = true;
        config.missing_tools = false;

        // set by build.rs
        config.build = TargetSelection::from_user(&env!("BUILD_TRIPLE"));
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // Undo `src/bootstrap`
        config.src = manifest_dir.parent().unwrap().parent().unwrap().to_owned();
        config.out = Config::path_from_python("BUILD_DIR");

        config.initial_cargo = PathBuf::from(env!("CARGO"));
        config.initial_rustc = PathBuf::from(env!("RUSTC"));

        config
    }

    pub fn parse(args: &[String]) -> Config {
        let flags = Flags::parse(&args);

        let mut config = Config::default_opts();
        config.exclude = flags.exclude;
        config.include_default_paths = flags.include_default_paths;
        config.rustc_error_format = flags.rustc_error_format;
        config.json_output = flags.json_output;
        config.on_fail = flags.on_fail;
        config.jobs = flags.jobs.map(threads_from_config);
        config.cmd = flags.cmd;
        config.incremental = flags.incremental;
        config.dry_run = flags.dry_run;
        config.keep_stage = flags.keep_stage;
        config.keep_stage_std = flags.keep_stage_std;
        config.bindir = "bin".into(); // default
        config.color = flags.color;
        if let Some(value) = flags.deny_warnings {
            config.deny_warnings = value;
        }

        if config.dry_run {
            let dir = config.out.join("tmp-dry-run");
            t!(fs::create_dir_all(&dir));
            config.out = dir;
        }

        #[cfg(test)]
        let get_toml = |_| TomlConfig::default();
        #[cfg(not(test))]
        let get_toml = |file: &Path| {
            use std::process;

            let contents = t!(fs::read_to_string(file), "`include` config not found");
            match toml::from_str(&contents) {
                Ok(table) => table,
                Err(err) => {
                    println!("failed to parse TOML configuration '{}': {}", file.display(), err);
                    process::exit(2);
                }
            }
        };

        let mut toml = flags.config.as_deref().map(get_toml).unwrap_or_else(TomlConfig::default);
        if let Some(include) = &toml.profile {
            let mut include_path = config.src.clone();
            include_path.push("src");
            include_path.push("bootstrap");
            include_path.push("defaults");
            include_path.push(format!("config.{}.toml", include));
            let included_toml = get_toml(&include_path);
            toml.merge(included_toml);
        }

        config.changelog_seen = toml.changelog_seen;
        if let Some(cfg) = flags.config {
            config.config = cfg;
        }

        let build = toml.build.unwrap_or_default();

        config.hosts = if let Some(arg_host) = flags.host {
            arg_host
        } else if let Some(file_host) = build.host {
            file_host.iter().map(|h| TargetSelection::from_user(h)).collect()
        } else {
            vec![config.build]
        };
        config.targets = if let Some(arg_target) = flags.target {
            arg_target
        } else if let Some(file_target) = build.target {
            file_target.iter().map(|h| TargetSelection::from_user(h)).collect()
        } else {
            // If target is *not* configured, then default to the host
            // toolchains.
            config.hosts.clone()
        };

        config.nodejs = build.nodejs.map(PathBuf::from);
        config.gdb = build.gdb.map(PathBuf::from);
        config.python = build.python.map(PathBuf::from);
        set(&mut config.low_priority, build.low_priority);
        set(&mut config.compiler_docs, build.compiler_docs);
        set(&mut config.docs, build.docs);
        set(&mut config.submodules, build.submodules);
        set(&mut config.fast_submodules, build.fast_submodules);
        set(&mut config.locked_deps, build.locked_deps);
        set(&mut config.vendor, build.vendor);
        set(&mut config.full_bootstrap, build.full_bootstrap);
        set(&mut config.extended, build.extended);
        config.tools = build.tools;
        if build.rustfmt.is_some() {
            config.initial_rustfmt = build.rustfmt;
        }
        set(&mut config.verbose, build.verbose);
        set(&mut config.sanitizers, build.sanitizers);
        set(&mut config.profiler, build.profiler);
        set(&mut config.cargo_native_static, build.cargo_native_static);
        set(&mut config.configure_args, build.configure_args);
        set(&mut config.local_rebuild, build.local_rebuild);
        set(&mut config.print_step_timings, build.print_step_timings);

        // See https://github.com/rust-lang/compiler-team/issues/326
        config.stage = match config.cmd {
            Subcommand::Doc { .. } => flags.stage.or(build.doc_stage).unwrap_or(0),
            Subcommand::Build { .. } => flags.stage.or(build.build_stage).unwrap_or(1),
            Subcommand::Test { .. } => flags.stage.or(build.test_stage).unwrap_or(1),
            Subcommand::Bench { .. } => flags.stage.or(build.bench_stage).unwrap_or(2),
            Subcommand::Dist { .. } => flags.stage.or(build.dist_stage).unwrap_or(2),
            Subcommand::Install { .. } => flags.stage.or(build.install_stage).unwrap_or(2),
            // These are all bootstrap tools, which don't depend on the compiler.
            // The stage we pass shouldn't matter, but use 0 just in case.
            Subcommand::Clean { .. }
            | Subcommand::Check { .. }
            | Subcommand::Clippy { .. }
            | Subcommand::Fix { .. }
            | Subcommand::Run { .. }
            | Subcommand::Setup { .. }
            | Subcommand::Format { .. } => flags.stage.unwrap_or(0),
        };

        // CI should always run stage 2 builds, unless it specifically states otherwise
        #[cfg(not(test))]
        if flags.stage.is_none() && crate::CiEnv::current() != crate::CiEnv::None {
            match config.cmd {
                Subcommand::Test { .. }
                | Subcommand::Doc { .. }
                | Subcommand::Build { .. }
                | Subcommand::Bench { .. }
                | Subcommand::Dist { .. }
                | Subcommand::Install { .. } => {
                    assert_eq!(
                        config.stage, 2,
                        "x.py should be run with `--stage 2` on CI, but was run with `--stage {}`",
                        config.stage,
                    );
                }
                Subcommand::Clean { .. }
                | Subcommand::Check { .. }
                | Subcommand::Clippy { .. }
                | Subcommand::Fix { .. }
                | Subcommand::Run { .. }
                | Subcommand::Setup { .. }
                | Subcommand::Format { .. } => {}
            }
        }

        config.verbose = cmp::max(config.verbose, flags.verbose);

        if let Some(install) = toml.install {
            config.prefix = install.prefix.map(PathBuf::from);
            config.sysconfdir = install.sysconfdir.map(PathBuf::from);
            config.datadir = install.datadir.map(PathBuf::from);
            config.docdir = install.docdir.map(PathBuf::from);
            set(&mut config.bindir, install.bindir.map(PathBuf::from));
            config.libdir = install.libdir.map(PathBuf::from);
            config.mandir = install.mandir.map(PathBuf::from);
        }

        // We want the llvm-skip-rebuild flag to take precedence over the
        // skip-rebuild config.toml option so we store it separately
        // so that we can infer the right value
        let mut llvm_skip_rebuild = flags.llvm_skip_rebuild;

        // Store off these values as options because if they're not provided
        // we'll infer default values for them later
        let mut llvm_assertions = None;
        let mut debug = None;
        let mut debug_assertions = None;
        let mut debug_assertions_std = None;
        let mut debug_logging = None;
        let mut debuginfo_level = None;
        let mut debuginfo_level_rustc = None;
        let mut debuginfo_level_std = None;
        let mut debuginfo_level_tools = None;
        let mut debuginfo_level_tests = None;
        let mut optimize = None;
        let mut ignore_git = None;

        if let Some(llvm) = toml.llvm {
            match llvm.ccache {
                Some(StringOrBool::String(ref s)) => config.ccache = Some(s.to_string()),
                Some(StringOrBool::Bool(true)) => {
                    config.ccache = Some("ccache".to_string());
                }
                Some(StringOrBool::Bool(false)) | None => {}
            }
            set(&mut config.ninja_in_file, llvm.ninja);
            llvm_assertions = llvm.assertions;
            llvm_skip_rebuild = llvm_skip_rebuild.or(llvm.skip_rebuild);
            set(&mut config.llvm_optimize, llvm.optimize);
            set(&mut config.llvm_thin_lto, llvm.thin_lto);
            set(&mut config.llvm_release_debuginfo, llvm.release_debuginfo);
            set(&mut config.llvm_version_check, llvm.version_check);
            set(&mut config.llvm_static_stdcpp, llvm.static_libstdcpp);
            set(&mut config.llvm_link_shared, llvm.link_shared);
            config.llvm_targets = llvm.targets.clone();
            config.llvm_experimental_targets = llvm.experimental_targets.clone();
            config.llvm_link_jobs = llvm.link_jobs;
            config.llvm_version_suffix = llvm.version_suffix.clone();
            config.llvm_clang_cl = llvm.clang_cl.clone();

            config.llvm_cflags = llvm.cflags.clone();
            config.llvm_cxxflags = llvm.cxxflags.clone();
            config.llvm_ldflags = llvm.ldflags.clone();
            set(&mut config.llvm_use_libcxx, llvm.use_libcxx);
            config.llvm_use_linker = llvm.use_linker.clone();
            config.llvm_allow_old_toolchain = llvm.allow_old_toolchain;
            config.llvm_polly = llvm.polly;
            config.llvm_from_ci = match llvm.download_ci_llvm {
                Some(StringOrBool::String(s)) => {
                    assert!(s == "if-available", "unknown option `{}` for download-ci-llvm", s);
                    config.build.triple == "x86_64-unknown-linux-gnu"
                }
                Some(StringOrBool::Bool(b)) => b,
                None => false,
            };

            if config.llvm_from_ci {
                // None of the LLVM options, except assertions, are supported
                // when using downloaded LLVM. We could just ignore these but
                // that's potentially confusing, so force them to not be
                // explicitly set. The defaults and CI defaults don't
                // necessarily match but forcing people to match (somewhat
                // arbitrary) CI configuration locally seems bad/hard.
                check_ci_llvm!(llvm.optimize);
                check_ci_llvm!(llvm.thin_lto);
                check_ci_llvm!(llvm.release_debuginfo);
                check_ci_llvm!(llvm.link_shared);
                check_ci_llvm!(llvm.static_libstdcpp);
                check_ci_llvm!(llvm.targets);
                check_ci_llvm!(llvm.experimental_targets);
                check_ci_llvm!(llvm.link_jobs);
                check_ci_llvm!(llvm.link_shared);
                check_ci_llvm!(llvm.clang_cl);
                check_ci_llvm!(llvm.version_suffix);
                check_ci_llvm!(llvm.cflags);
                check_ci_llvm!(llvm.cxxflags);
                check_ci_llvm!(llvm.ldflags);
                check_ci_llvm!(llvm.use_libcxx);
                check_ci_llvm!(llvm.use_linker);
                check_ci_llvm!(llvm.allow_old_toolchain);
                check_ci_llvm!(llvm.polly);

                // CI-built LLVM is shared
                config.llvm_link_shared = true;
            }

            if config.llvm_thin_lto {
                // If we're building with ThinLTO on, we want to link to LLVM
                // shared, to avoid re-doing ThinLTO (which happens in the link
                // step) with each stage.
                config.llvm_link_shared = true;
            }
        }

        if let Some(rust) = toml.rust {
            debug = rust.debug;
            debug_assertions = rust.debug_assertions;
            debug_assertions_std = rust.debug_assertions_std;
            debug_logging = rust.debug_logging;
            debuginfo_level = rust.debuginfo_level;
            debuginfo_level_rustc = rust.debuginfo_level_rustc;
            debuginfo_level_std = rust.debuginfo_level_std;
            debuginfo_level_tools = rust.debuginfo_level_tools;
            debuginfo_level_tests = rust.debuginfo_level_tests;
            config.rust_run_dsymutil = rust.run_dsymutil.unwrap_or(false);
            optimize = rust.optimize;
            ignore_git = rust.ignore_git;
            set(&mut config.rust_new_symbol_mangling, rust.new_symbol_mangling);
            set(&mut config.rust_optimize_tests, rust.optimize_tests);
            set(&mut config.codegen_tests, rust.codegen_tests);
            set(&mut config.rust_rpath, rust.rpath);
            set(&mut config.jemalloc, rust.jemalloc);
            set(&mut config.test_compare_mode, rust.test_compare_mode);
            config.llvm_libunwind = rust
                .llvm_libunwind
                .map(|v| v.parse().expect("failed to parse rust.llvm-libunwind"));
            set(&mut config.backtrace, rust.backtrace);
            set(&mut config.channel, rust.channel);
            config.description = rust.description;
            set(&mut config.rust_dist_src, rust.dist_src);
            set(&mut config.verbose_tests, rust.verbose_tests);
            // in the case "false" is set explicitly, do not overwrite the command line args
            if let Some(true) = rust.incremental {
                config.incremental = true;
            }
            set(&mut config.use_lld, rust.use_lld);
            set(&mut config.lld_enabled, rust.lld);
            set(&mut config.llvm_tools_enabled, rust.llvm_tools);
            config.rustc_parallel = rust.parallel_compiler.unwrap_or(false);
            config.rustc_default_linker = rust.default_linker;
            config.musl_root = rust.musl_root.map(PathBuf::from);
            config.save_toolstates = rust.save_toolstates.map(PathBuf::from);
            set(&mut config.deny_warnings, flags.deny_warnings.or(rust.deny_warnings));
            set(&mut config.backtrace_on_ice, rust.backtrace_on_ice);
            set(&mut config.rust_verify_llvm_ir, rust.verify_llvm_ir);
            config.rust_thin_lto_import_instr_limit = rust.thin_lto_import_instr_limit;
            set(&mut config.rust_remap_debuginfo, rust.remap_debuginfo);
            set(&mut config.control_flow_guard, rust.control_flow_guard);

            if let Some(ref backends) = rust.codegen_backends {
                config.rust_codegen_backends =
                    backends.iter().map(|s| INTERNER.intern_str(s)).collect();
            }

            config.rust_codegen_units = rust.codegen_units.map(threads_from_config);
            config.rust_codegen_units_std = rust.codegen_units_std.map(threads_from_config);
            config.rust_profile_use = flags.rust_profile_use.or(rust.profile_use);
            config.rust_profile_generate = flags.rust_profile_generate.or(rust.profile_generate);
        } else {
            config.rust_profile_use = flags.rust_profile_use;
            config.rust_profile_generate = flags.rust_profile_generate;
        }

        if let Some(t) = toml.target {
            for (triple, cfg) in t {
                let mut target = Target::from_triple(&triple);

                if let Some(ref s) = cfg.llvm_config {
                    target.llvm_config = Some(config.src.join(s));
                }
                if let Some(ref s) = cfg.llvm_filecheck {
                    target.llvm_filecheck = Some(config.src.join(s));
                }
                if let Some(ref s) = cfg.android_ndk {
                    target.ndk = Some(config.src.join(s));
                }
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
                target.sanitizers = cfg.sanitizers;
                target.profiler = cfg.profiler;

                config.target_config.insert(TargetSelection::from_user(&triple), target);
            }
        }

        if config.llvm_from_ci {
            let triple = &config.build.triple;
            let mut build_target = config
                .target_config
                .entry(config.build)
                .or_insert_with(|| Target::from_triple(&triple));

            check_ci_llvm!(build_target.llvm_config);
            check_ci_llvm!(build_target.llvm_filecheck);
            let ci_llvm_bin = config.out.join(&*config.build.triple).join("ci-llvm/bin");
            build_target.llvm_config = Some(ci_llvm_bin.join(exe("llvm-config", config.build)));
            build_target.llvm_filecheck = Some(ci_llvm_bin.join(exe("FileCheck", config.build)));
        }

        if let Some(t) = toml.dist {
            config.dist_sign_folder = t.sign_folder.map(PathBuf::from);
            config.dist_gpg_password_file = t.gpg_password_file.map(PathBuf::from);
            config.dist_upload_addr = t.upload_addr;
            config.dist_compression_formats = t.compression_formats;
            set(&mut config.rust_dist_src, t.src_tarball);
            set(&mut config.missing_tools, t.missing_tools);
        }

        config.initial_rustfmt = config.initial_rustfmt.or_else({
            let build = config.build;
            let initial_rustc = &config.initial_rustc;

            move || {
                // Cargo does not provide a RUSTFMT environment variable, so we
                // synthesize it manually.
                let rustfmt = initial_rustc.with_file_name(exe("rustfmt", build));

                if rustfmt.exists() { Some(rustfmt) } else { None }
            }
        });

        // Now that we've reached the end of our configuration, infer the
        // default values for all options that we haven't otherwise stored yet.

        config.llvm_skip_rebuild = llvm_skip_rebuild.unwrap_or(false);

        let default = false;
        config.llvm_assertions = llvm_assertions.unwrap_or(default);

        let default = true;
        config.rust_optimize = optimize.unwrap_or(default);

        let default = debug == Some(true);
        config.rust_debug_assertions = debug_assertions.unwrap_or(default);
        config.rust_debug_assertions_std =
            debug_assertions_std.unwrap_or(config.rust_debug_assertions);

        config.rust_debug_logging = debug_logging.unwrap_or(config.rust_debug_assertions);

        let with_defaults = |debuginfo_level_specific: Option<u32>| {
            debuginfo_level_specific.or(debuginfo_level).unwrap_or(if debug == Some(true) {
                1
            } else {
                0
            })
        };
        config.rust_debuginfo_level_rustc = with_defaults(debuginfo_level_rustc);
        config.rust_debuginfo_level_std = with_defaults(debuginfo_level_std);
        config.rust_debuginfo_level_tools = with_defaults(debuginfo_level_tools);
        config.rust_debuginfo_level_tests = debuginfo_level_tests.unwrap_or(0);

        let default = config.channel == "dev";
        config.ignore_git = ignore_git.unwrap_or(default);

        config
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

    pub fn verbose(&self) -> bool {
        self.verbose > 0
    }

    pub fn very_verbose(&self) -> bool {
        self.verbose > 1
    }

    pub fn sanitizers_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).map(|t| t.sanitizers).flatten().unwrap_or(self.sanitizers)
    }

    pub fn any_sanitizers_enabled(&self) -> bool {
        self.target_config.values().any(|t| t.sanitizers == Some(true)) || self.sanitizers
    }

    pub fn profiler_enabled(&self, target: TargetSelection) -> bool {
        self.target_config.get(&target).map(|t| t.profiler).flatten().unwrap_or(self.profiler)
    }

    pub fn any_profiler_enabled(&self) -> bool {
        self.target_config.values().any(|t| t.profiler == Some(true)) || self.profiler
    }

    pub fn llvm_enabled(&self) -> bool {
        self.rust_codegen_backends.contains(&INTERNER.intern_str("llvm"))
    }
}

fn set<T>(field: &mut T, val: Option<T>) {
    if let Some(v) = val {
        *field = v;
    }
}

fn threads_from_config(v: u32) -> u32 {
    match v {
        0 => num_cpus::get() as u32,
        n => n,
    }
}
