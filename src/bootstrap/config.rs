//! Serialized configuration of a build.
//!
//! This module implements parsing `config.toml` configuration files to tweak
//! how the build runs.

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::cmp;

use build_helper::t;
use toml;
use serde::Deserialize;
use crate::cache::{INTERNER, Interned};
use crate::flags::Flags;
pub use crate::flags::Subcommand;

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
    pub ccache: Option<String>,
    pub ninja: bool,
    pub verbose: usize,
    pub submodules: bool,
    pub fast_submodules: bool,
    pub compiler_docs: bool,
    pub docs: bool,
    pub locked_deps: bool,
    pub vendor: bool,
    pub target_config: HashMap<Interned<String>, Target>,
    pub full_bootstrap: bool,
    pub extended: bool,
    pub tools: Option<HashSet<String>>,
    pub sanitizers: bool,
    pub profiler: bool,
    pub ignore_git: bool,
    pub exclude: Vec<PathBuf>,
    pub rustc_error_format: Option<String>,
    pub test_compare_mode: bool,
    pub llvm_libunwind: bool,

    pub skip_only_host_steps: bool,

    pub on_fail: Option<String>,
    pub stage: Option<u32>,
    pub keep_stage: Vec<u32>,
    pub src: PathBuf,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,
    pub dry_run: bool,

    pub deny_warnings: bool,
    pub backtrace_on_ice: bool,

    // llvm codegen options
    pub llvm_assertions: bool,
    pub llvm_optimize: bool,
    pub llvm_thin_lto: bool,
    pub llvm_release_debuginfo: bool,
    pub llvm_version_check: bool,
    pub llvm_static_stdcpp: bool,
    pub llvm_link_shared: bool,
    pub llvm_clang_cl: Option<String>,
    pub llvm_targets: Option<String>,
    pub llvm_experimental_targets: String,
    pub llvm_link_jobs: Option<u32>,
    pub llvm_version_suffix: Option<String>,
    pub llvm_use_linker: Option<String>,
    pub llvm_allow_old_toolchain: Option<bool>,

    pub lld_enabled: bool,
    pub lldb_enabled: bool,
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
    pub rust_debuginfo_level_rustc: u32,
    pub rust_debuginfo_level_std: u32,
    pub rust_debuginfo_level_tools: u32,
    pub rust_debuginfo_level_tests: u32,
    pub rust_rpath: bool,
    pub rustc_parallel: bool,
    pub rustc_default_linker: Option<String>,
    pub rust_optimize_tests: bool,
    pub rust_dist_src: bool,
    pub rust_codegen_backends: Vec<Interned<String>>,
    pub rust_codegen_backends_dir: String,
    pub rust_verify_llvm_ir: bool,
    pub rust_remap_debuginfo: bool,

    pub build: Interned<String>,
    pub hosts: Vec<Interned<String>>,
    pub targets: Vec<Interned<String>>,
    pub local_rebuild: bool,
    pub jemalloc: bool,

    // dist misc
    pub dist_sign_folder: Option<PathBuf>,
    pub dist_upload_addr: Option<String>,
    pub dist_gpg_password_file: Option<PathBuf>,

    // libstd features
    pub backtrace: bool, // support for RUST_BACKTRACE
    pub wasm_syscall: bool,

    // misc
    pub low_priority: bool,
    pub channel: String,
    pub verbose_tests: bool,
    pub test_miri: bool,
    pub save_toolstates: Option<PathBuf>,
    pub print_step_timings: bool,
    pub missing_tools: bool,

    // Fallback musl-root for all targets
    pub musl_root: Option<PathBuf>,
    pub prefix: Option<PathBuf>,
    pub sysconfdir: Option<PathBuf>,
    pub datadir: Option<PathBuf>,
    pub docdir: Option<PathBuf>,
    pub bindir: Option<PathBuf>,
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
    pub out: PathBuf,
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
    pub crt_static: Option<bool>,
    pub musl_root: Option<PathBuf>,
    pub wasi_root: Option<PathBuf>,
    pub qemu_rootfs: Option<PathBuf>,
    pub no_std: bool,
}

/// Structure of the `config.toml` file that configuration is read from.
///
/// This structure uses `Decodable` to automatically decode a TOML configuration
/// file into this format, and then this is traversed and written into the above
/// `Config` structure.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct TomlConfig {
    build: Option<Build>,
    install: Option<Install>,
    llvm: Option<Llvm>,
    rust: Option<Rust>,
    target: Option<HashMap<String, TomlTarget>>,
    dist: Option<Dist>,
}

/// TOML representation of various global build decisions.
#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Build {
    build: Option<String>,
    #[serde(default)]
    host: Vec<String>,
    #[serde(default)]
    target: Vec<String>,
    cargo: Option<String>,
    rustc: Option<String>,
    low_priority: Option<bool>,
    compiler_docs: Option<bool>,
    docs: Option<bool>,
    submodules: Option<bool>,
    fast_submodules: Option<bool>,
    gdb: Option<String>,
    locked_deps: Option<bool>,
    vendor: Option<bool>,
    nodejs: Option<String>,
    python: Option<String>,
    full_bootstrap: Option<bool>,
    extended: Option<bool>,
    tools: Option<HashSet<String>>,
    verbose: Option<usize>,
    sanitizers: Option<bool>,
    profiler: Option<bool>,
    cargo_native_static: Option<bool>,
    configure_args: Option<Vec<String>>,
    local_rebuild: Option<bool>,
    print_step_timings: Option<bool>,
}

/// TOML representation of various global install decisions.
#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Install {
    prefix: Option<String>,
    sysconfdir: Option<String>,
    datadir: Option<String>,
    docdir: Option<String>,
    bindir: Option<String>,
    libdir: Option<String>,
    mandir: Option<String>,

    // standard paths, currently unused
    infodir: Option<String>,
    localstatedir: Option<String>,
}

/// TOML representation of how the LLVM build is configured.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Llvm {
    ccache: Option<StringOrBool>,
    ninja: Option<bool>,
    assertions: Option<bool>,
    optimize: Option<bool>,
    thin_lto: Option<bool>,
    release_debuginfo: Option<bool>,
    version_check: Option<bool>,
    static_libstdcpp: Option<bool>,
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
}

#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Dist {
    sign_folder: Option<String>,
    gpg_password_file: Option<String>,
    upload_addr: Option<String>,
    src_tarball: Option<bool>,
    missing_tools: Option<bool>,
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
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Rust {
    optimize: Option<bool>,
    codegen_units: Option<u32>,
    codegen_units_std: Option<u32>,
    debug_assertions: Option<bool>,
    debuginfo_level: Option<u32>,
    debuginfo_level_rustc: Option<u32>,
    debuginfo_level_std: Option<u32>,
    debuginfo_level_tools: Option<u32>,
    debuginfo_level_tests: Option<u32>,
    parallel_compiler: Option<bool>,
    backtrace: Option<bool>,
    default_linker: Option<String>,
    channel: Option<String>,
    musl_root: Option<String>,
    rpath: Option<bool>,
    optimize_tests: Option<bool>,
    codegen_tests: Option<bool>,
    ignore_git: Option<bool>,
    debug: Option<bool>,
    dist_src: Option<bool>,
    verbose_tests: Option<bool>,
    test_miri: Option<bool>,
    incremental: Option<bool>,
    save_toolstates: Option<String>,
    codegen_backends: Option<Vec<String>>,
    codegen_backends_dir: Option<String>,
    wasm_syscall: Option<bool>,
    lld: Option<bool>,
    lldb: Option<bool>,
    llvm_tools: Option<bool>,
    deny_warnings: Option<bool>,
    backtrace_on_ice: Option<bool>,
    verify_llvm_ir: Option<bool>,
    remap_debuginfo: Option<bool>,
    jemalloc: Option<bool>,
    test_compare_mode: Option<bool>,
    llvm_libunwind: Option<bool>,
}

/// TOML representation of how each build target is configured.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct TomlTarget {
    llvm_config: Option<String>,
    llvm_filecheck: Option<String>,
    cc: Option<String>,
    cxx: Option<String>,
    ar: Option<String>,
    ranlib: Option<String>,
    linker: Option<String>,
    android_ndk: Option<String>,
    crt_static: Option<bool>,
    musl_root: Option<String>,
    wasi_root: Option<String>,
    qemu_rootfs: Option<String>,
}

impl Config {
    fn path_from_python(var_key: &str) -> PathBuf {
        match env::var_os(var_key) {
            // Do not trust paths from Python and normalize them slightly (#49785).
            Some(var_val) => Path::new(&var_val).components().collect(),
            _ => panic!("expected '{}' to be set", var_key),
        }
    }

    pub fn default_opts() -> Config {
        let mut config = Config::default();
        config.llvm_optimize = true;
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
        config.test_miri = false;
        config.rust_codegen_backends = vec![INTERNER.intern_str("llvm")];
        config.rust_codegen_backends_dir = "codegen-backends".to_owned();
        config.deny_warnings = true;
        config.missing_tools = false;

        // set by bootstrap.py
        config.build = INTERNER.intern_str(&env::var("BUILD").expect("'BUILD' to be set"));
        config.src = Config::path_from_python("SRC");
        config.out = Config::path_from_python("BUILD_DIR");

        config.initial_rustc = Config::path_from_python("RUSTC");
        config.initial_cargo = Config::path_from_python("CARGO");

        config
    }

    pub fn parse(args: &[String]) -> Config {
        let flags = Flags::parse(&args);
        let file = flags.config.clone();
        let mut config = Config::default_opts();
        config.exclude = flags.exclude;
        config.rustc_error_format = flags.rustc_error_format;
        config.on_fail = flags.on_fail;
        config.stage = flags.stage;
        config.jobs = flags.jobs.map(threads_from_config);
        config.cmd = flags.cmd;
        config.incremental = flags.incremental;
        config.dry_run = flags.dry_run;
        config.keep_stage = flags.keep_stage;
        if let Some(value) = flags.warnings {
            config.deny_warnings = value;
        }

        if config.dry_run {
            let dir = config.out.join("tmp-dry-run");
            t!(fs::create_dir_all(&dir));
            config.out = dir;
        }

        // If --target was specified but --host wasn't specified, don't run any host-only tests.
        let has_hosts = !flags.host.is_empty();
        let has_targets = !flags.target.is_empty();
        config.skip_only_host_steps = !has_hosts && has_targets;

        let toml = file.map(|file| {
            let contents = t!(fs::read_to_string(&file));
            match toml::from_str(&contents) {
                Ok(table) => table,
                Err(err) => {
                    println!("failed to parse TOML configuration '{}': {}",
                        file.display(), err);
                    process::exit(2);
                }
            }
        }).unwrap_or_else(|| TomlConfig::default());

        let build = toml.build.clone().unwrap_or_default();
        // set by bootstrap.py
        config.hosts.push(config.build.clone());
        for host in build.host.iter() {
            let host = INTERNER.intern_str(host);
            if !config.hosts.contains(&host) {
                config.hosts.push(host);
            }
        }
        for target in config.hosts.iter().cloned()
            .chain(build.target.iter().map(|s| INTERNER.intern_str(s)))
        {
            if !config.targets.contains(&target) {
                config.targets.push(target);
            }
        }
        config.hosts = if !flags.host.is_empty() {
            flags.host
        } else {
            config.hosts
        };
        config.targets = if !flags.target.is_empty() {
            flags.target
        } else {
            config.targets
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
        set(&mut config.verbose, build.verbose);
        set(&mut config.sanitizers, build.sanitizers);
        set(&mut config.profiler, build.profiler);
        set(&mut config.cargo_native_static, build.cargo_native_static);
        set(&mut config.configure_args, build.configure_args);
        set(&mut config.local_rebuild, build.local_rebuild);
        set(&mut config.print_step_timings, build.print_step_timings);
        config.verbose = cmp::max(config.verbose, flags.verbose);

        if let Some(ref install) = toml.install {
            config.prefix = install.prefix.clone().map(PathBuf::from);
            config.sysconfdir = install.sysconfdir.clone().map(PathBuf::from);
            config.datadir = install.datadir.clone().map(PathBuf::from);
            config.docdir = install.docdir.clone().map(PathBuf::from);
            config.bindir = install.bindir.clone().map(PathBuf::from);
            config.libdir = install.libdir.clone().map(PathBuf::from);
            config.mandir = install.mandir.clone().map(PathBuf::from);
        }

        // Store off these values as options because if they're not provided
        // we'll infer default values for them later
        let mut llvm_assertions = None;
        let mut debug = None;
        let mut debug_assertions = None;
        let mut debuginfo_level = None;
        let mut debuginfo_level_rustc = None;
        let mut debuginfo_level_std = None;
        let mut debuginfo_level_tools = None;
        let mut debuginfo_level_tests = None;
        let mut optimize = None;
        let mut ignore_git = None;

        if let Some(ref llvm) = toml.llvm {
            match llvm.ccache {
                Some(StringOrBool::String(ref s)) => {
                    config.ccache = Some(s.to_string())
                }
                Some(StringOrBool::Bool(true)) => {
                    config.ccache = Some("ccache".to_string());
                }
                Some(StringOrBool::Bool(false)) | None => {}
            }
            set(&mut config.ninja, llvm.ninja);
            llvm_assertions = llvm.assertions;
            set(&mut config.llvm_optimize, llvm.optimize);
            set(&mut config.llvm_thin_lto, llvm.thin_lto);
            set(&mut config.llvm_release_debuginfo, llvm.release_debuginfo);
            set(&mut config.llvm_version_check, llvm.version_check);
            set(&mut config.llvm_static_stdcpp, llvm.static_libstdcpp);
            set(&mut config.llvm_link_shared, llvm.link_shared);
            config.llvm_targets = llvm.targets.clone();
            config.llvm_experimental_targets = llvm.experimental_targets.clone()
                .unwrap_or_else(|| "WebAssembly;RISCV".to_string());
            config.llvm_link_jobs = llvm.link_jobs;
            config.llvm_version_suffix = llvm.version_suffix.clone();
            config.llvm_clang_cl = llvm.clang_cl.clone();

            config.llvm_cflags = llvm.cflags.clone();
            config.llvm_cxxflags = llvm.cxxflags.clone();
            config.llvm_ldflags = llvm.ldflags.clone();
            set(&mut config.llvm_use_libcxx, llvm.use_libcxx);
            config.llvm_use_linker = llvm.use_linker.clone();
            config.llvm_allow_old_toolchain = llvm.allow_old_toolchain.clone();
        }

        if let Some(ref rust) = toml.rust {
            debug = rust.debug;
            debug_assertions = rust.debug_assertions;
            debuginfo_level = rust.debuginfo_level;
            debuginfo_level_rustc = rust.debuginfo_level_rustc;
            debuginfo_level_std = rust.debuginfo_level_std;
            debuginfo_level_tools = rust.debuginfo_level_tools;
            debuginfo_level_tests = rust.debuginfo_level_tests;
            optimize = rust.optimize;
            ignore_git = rust.ignore_git;
            set(&mut config.rust_optimize_tests, rust.optimize_tests);
            set(&mut config.codegen_tests, rust.codegen_tests);
            set(&mut config.rust_rpath, rust.rpath);
            set(&mut config.jemalloc, rust.jemalloc);
            set(&mut config.test_compare_mode, rust.test_compare_mode);
            set(&mut config.llvm_libunwind, rust.llvm_libunwind);
            set(&mut config.backtrace, rust.backtrace);
            set(&mut config.channel, rust.channel.clone());
            set(&mut config.rust_dist_src, rust.dist_src);
            set(&mut config.verbose_tests, rust.verbose_tests);
            set(&mut config.test_miri, rust.test_miri);
            // in the case "false" is set explicitly, do not overwrite the command line args
            if let Some(true) = rust.incremental {
                config.incremental = true;
            }
            set(&mut config.wasm_syscall, rust.wasm_syscall);
            set(&mut config.lld_enabled, rust.lld);
            set(&mut config.lldb_enabled, rust.lldb);
            set(&mut config.llvm_tools_enabled, rust.llvm_tools);
            config.rustc_parallel = rust.parallel_compiler.unwrap_or(false);
            config.rustc_default_linker = rust.default_linker.clone();
            config.musl_root = rust.musl_root.clone().map(PathBuf::from);
            config.save_toolstates = rust.save_toolstates.clone().map(PathBuf::from);
            set(&mut config.deny_warnings, rust.deny_warnings.or(flags.warnings));
            set(&mut config.backtrace_on_ice, rust.backtrace_on_ice);
            set(&mut config.rust_verify_llvm_ir, rust.verify_llvm_ir);
            set(&mut config.rust_remap_debuginfo, rust.remap_debuginfo);

            if let Some(ref backends) = rust.codegen_backends {
                config.rust_codegen_backends = backends.iter()
                    .map(|s| INTERNER.intern_str(s))
                    .collect();
            }

            set(&mut config.rust_codegen_backends_dir, rust.codegen_backends_dir.clone());

            config.rust_codegen_units = rust.codegen_units.map(threads_from_config);
            config.rust_codegen_units_std = rust.codegen_units_std.map(threads_from_config);
        }

        if let Some(ref t) = toml.target {
            for (triple, cfg) in t {
                let mut target = Target::default();

                if let Some(ref s) = cfg.llvm_config {
                    target.llvm_config = Some(config.src.join(s));
                }
                if let Some(ref s) = cfg.llvm_filecheck {
                    target.llvm_filecheck = Some(config.src.join(s));
                }
                if let Some(ref s) = cfg.android_ndk {
                    target.ndk = Some(config.src.join(s));
                }
                target.cc = cfg.cc.clone().map(PathBuf::from);
                target.cxx = cfg.cxx.clone().map(PathBuf::from);
                target.ar = cfg.ar.clone().map(PathBuf::from);
                target.ranlib = cfg.ranlib.clone().map(PathBuf::from);
                target.linker = cfg.linker.clone().map(PathBuf::from);
                target.crt_static = cfg.crt_static.clone();
                target.musl_root = cfg.musl_root.clone().map(PathBuf::from);
                target.wasi_root = cfg.wasi_root.clone().map(PathBuf::from);
                target.qemu_rootfs = cfg.qemu_rootfs.clone().map(PathBuf::from);

                config.target_config.insert(INTERNER.intern_string(triple.clone()), target);
            }
        }

        if let Some(ref t) = toml.dist {
            config.dist_sign_folder = t.sign_folder.clone().map(PathBuf::from);
            config.dist_gpg_password_file = t.gpg_password_file.clone().map(PathBuf::from);
            config.dist_upload_addr = t.upload_addr.clone();
            set(&mut config.rust_dist_src, t.src_tarball);
            set(&mut config.missing_tools, t.missing_tools);
        }

        // Now that we've reached the end of our configuration, infer the
        // default values for all options that we haven't otherwise stored yet.

        set(&mut config.initial_rustc, build.rustc.map(PathBuf::from));
        set(&mut config.initial_cargo, build.cargo.map(PathBuf::from));

        let default = false;
        config.llvm_assertions = llvm_assertions.unwrap_or(default);

        let default = true;
        config.rust_optimize = optimize.unwrap_or(default);

        let default = debug == Some(true);
        config.rust_debug_assertions = debug_assertions.unwrap_or(default);

        let with_defaults = |debuginfo_level_specific: Option<u32>| {
            debuginfo_level_specific
                .or(debuginfo_level)
                .unwrap_or(if debug == Some(true) { 2 } else { 0 })
        };
        config.rust_debuginfo_level_rustc = with_defaults(debuginfo_level_rustc);
        config.rust_debuginfo_level_std = with_defaults(debuginfo_level_std);
        config.rust_debuginfo_level_tools = with_defaults(debuginfo_level_tools);
        config.rust_debuginfo_level_tests = debuginfo_level_tests.unwrap_or(0);

        let default = config.channel == "dev";
        config.ignore_git = ignore_git.unwrap_or(default);

        config
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

    pub fn llvm_enabled(&self) -> bool {
        self.rust_codegen_backends.contains(&INTERNER.intern_str("llvm"))
        || self.rust_codegen_backends.contains(&INTERNER.intern_str("emscripten"))
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
