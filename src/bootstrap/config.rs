// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Serialized configuration of a build.
//!
//! This module implements parsing `config.toml` configuration files to tweak
//! how the build runs.

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::process;
use std::cmp;

use num_cpus;
use toml;
use util::exe;
use cache::{INTERNER, Interned};
use flags::Flags;
pub use flags::Subcommand;

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
    pub compiler_docs: bool,
    pub docs: bool,
    pub locked_deps: bool,
    pub vendor: bool,
    pub target_config: HashMap<Interned<String>, Target>,
    pub full_bootstrap: bool,
    pub extended: bool,
    pub sanitizers: bool,
    pub profiler: bool,
    pub ignore_git: bool,

    pub run_host_only: bool,

    pub on_fail: Option<String>,
    pub stage: Option<u32>,
    pub keep_stage: Option<u32>,
    pub src: PathBuf,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub incremental: bool,

    // llvm codegen options
    pub llvm_enabled: bool,
    pub llvm_assertions: bool,
    pub llvm_optimize: bool,
    pub llvm_release_debuginfo: bool,
    pub llvm_version_check: bool,
    pub llvm_static_stdcpp: bool,
    pub llvm_link_shared: bool,
    pub llvm_targets: Option<String>,
    pub llvm_experimental_targets: Option<String>,
    pub llvm_link_jobs: Option<u32>,

    // rust codegen options
    pub rust_optimize: bool,
    pub rust_codegen_units: u32,
    pub rust_debug_assertions: bool,
    pub rust_debuginfo: bool,
    pub rust_debuginfo_lines: bool,
    pub rust_debuginfo_only_std: bool,
    pub rust_rpath: bool,
    pub rustc_default_linker: Option<String>,
    pub rustc_default_ar: Option<String>,
    pub rust_optimize_tests: bool,
    pub rust_debuginfo_tests: bool,
    pub rust_dist_src: bool,

    pub build: Interned<String>,
    pub hosts: Vec<Interned<String>>,
    pub targets: Vec<Interned<String>>,
    pub local_rebuild: bool,

    // dist misc
    pub dist_sign_folder: Option<PathBuf>,
    pub dist_upload_addr: Option<String>,
    pub dist_gpg_password_file: Option<PathBuf>,

    // libstd features
    pub debug_jemalloc: bool,
    pub use_jemalloc: bool,
    pub backtrace: bool, // support for RUST_BACKTRACE

    // misc
    pub low_priority: bool,
    pub channel: String,
    pub quiet_tests: bool,
    // Fallback musl-root for all targets
    pub musl_root: Option<PathBuf>,
    pub prefix: Option<PathBuf>,
    pub sysconfdir: Option<PathBuf>,
    pub docdir: Option<PathBuf>,
    pub bindir: Option<PathBuf>,
    pub libdir: Option<PathBuf>,
    pub libdir_relative: Option<PathBuf>,
    pub mandir: Option<PathBuf>,
    pub codegen_tests: bool,
    pub nodejs: Option<PathBuf>,
    pub gdb: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub openssl_static: bool,
    pub configure_args: Vec<String>,

    // These are either the stage0 downloaded binaries or the locally installed ones.
    pub initial_cargo: PathBuf,
    pub initial_rustc: PathBuf,
}

/// Per-target configuration stored in the global configuration structure.
#[derive(Default)]
pub struct Target {
    /// Some(path to llvm-config) if using an external LLVM.
    pub llvm_config: Option<PathBuf>,
    pub jemalloc: Option<PathBuf>,
    pub cc: Option<PathBuf>,
    pub cxx: Option<PathBuf>,
    pub ndk: Option<PathBuf>,
    pub crt_static: Option<bool>,
    pub musl_root: Option<PathBuf>,
    pub qemu_rootfs: Option<PathBuf>,
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
    gdb: Option<String>,
    locked_deps: Option<bool>,
    vendor: Option<bool>,
    nodejs: Option<String>,
    python: Option<String>,
    full_bootstrap: Option<bool>,
    extended: Option<bool>,
    verbose: Option<usize>,
    sanitizers: Option<bool>,
    profiler: Option<bool>,
    openssl_static: Option<bool>,
    configure_args: Option<Vec<String>>,
    local_rebuild: Option<bool>,
}

/// TOML representation of various global install decisions.
#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Install {
    prefix: Option<String>,
    sysconfdir: Option<String>,
    docdir: Option<String>,
    bindir: Option<String>,
    libdir: Option<String>,
    mandir: Option<String>,
}

/// TOML representation of how the LLVM build is configured.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Llvm {
    enabled: Option<bool>,
    ccache: Option<StringOrBool>,
    ninja: Option<bool>,
    assertions: Option<bool>,
    optimize: Option<bool>,
    release_debuginfo: Option<bool>,
    version_check: Option<bool>,
    static_libstdcpp: Option<bool>,
    targets: Option<String>,
    experimental_targets: Option<String>,
    link_jobs: Option<u32>,
    link_shared: Option<bool>,
}

#[derive(Deserialize, Default, Clone)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct Dist {
    sign_folder: Option<String>,
    gpg_password_file: Option<String>,
    upload_addr: Option<String>,
    src_tarball: Option<bool>,
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
    debug_assertions: Option<bool>,
    debuginfo: Option<bool>,
    debuginfo_lines: Option<bool>,
    debuginfo_only_std: Option<bool>,
    debug_jemalloc: Option<bool>,
    use_jemalloc: Option<bool>,
    backtrace: Option<bool>,
    default_linker: Option<String>,
    default_ar: Option<String>,
    channel: Option<String>,
    musl_root: Option<String>,
    rpath: Option<bool>,
    optimize_tests: Option<bool>,
    debuginfo_tests: Option<bool>,
    codegen_tests: Option<bool>,
    ignore_git: Option<bool>,
    debug: Option<bool>,
    dist_src: Option<bool>,
    quiet_tests: Option<bool>,
}

/// TOML representation of how each build target is configured.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct TomlTarget {
    llvm_config: Option<String>,
    jemalloc: Option<String>,
    cc: Option<String>,
    cxx: Option<String>,
    android_ndk: Option<String>,
    crt_static: Option<bool>,
    musl_root: Option<String>,
    qemu_rootfs: Option<String>,
}

impl Config {
    pub fn parse(args: &[String]) -> Config {
        let flags = Flags::parse(&args);
        let file = flags.config.clone();
        let mut config = Config::default();
        config.llvm_enabled = true;
        config.llvm_optimize = true;
        config.use_jemalloc = true;
        config.backtrace = true;
        config.rust_optimize = true;
        config.rust_optimize_tests = true;
        config.submodules = true;
        config.docs = true;
        config.rust_rpath = true;
        config.rust_codegen_units = 1;
        config.channel = "dev".to_string();
        config.codegen_tests = true;
        config.ignore_git = false;
        config.rust_dist_src = true;

        config.on_fail = flags.on_fail;
        config.stage = flags.stage;
        config.src = flags.src;
        config.jobs = flags.jobs;
        config.cmd = flags.cmd;
        config.incremental = flags.incremental;
        config.keep_stage = flags.keep_stage;

        // If --target was specified but --host wasn't specified, don't run any host-only tests.
        config.run_host_only = flags.host.is_empty() && !flags.target.is_empty();

        let toml = file.map(|file| {
            let mut f = t!(File::open(&file));
            let mut contents = String::new();
            t!(f.read_to_string(&mut contents));
            match toml::from_str(&contents) {
                Ok(table) => table,
                Err(err) => {
                    println!("failed to parse TOML configuration '{}': {}",
                        file.display(), err);
                    process::exit(2);
                }
            }
        }).unwrap_or_else(|| TomlConfig::default());

        let build = toml.build.clone().unwrap_or(Build::default());
        set(&mut config.build, build.build.clone().map(|x| INTERNER.intern_string(x)));
        set(&mut config.build, flags.build);
        if config.build.is_empty() {
            // set by bootstrap.py
            config.build = INTERNER.intern_str(&env::var("BUILD").unwrap());
        }
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
        set(&mut config.locked_deps, build.locked_deps);
        set(&mut config.vendor, build.vendor);
        set(&mut config.full_bootstrap, build.full_bootstrap);
        set(&mut config.extended, build.extended);
        set(&mut config.verbose, build.verbose);
        set(&mut config.sanitizers, build.sanitizers);
        set(&mut config.profiler, build.profiler);
        set(&mut config.openssl_static, build.openssl_static);
        set(&mut config.configure_args, build.configure_args);
        set(&mut config.local_rebuild, build.local_rebuild);
        config.verbose = cmp::max(config.verbose, flags.verbose);

        if let Some(ref install) = toml.install {
            config.prefix = install.prefix.clone().map(PathBuf::from);
            config.sysconfdir = install.sysconfdir.clone().map(PathBuf::from);
            config.docdir = install.docdir.clone().map(PathBuf::from);
            config.bindir = install.bindir.clone().map(PathBuf::from);
            config.libdir = install.libdir.clone().map(PathBuf::from);
            config.mandir = install.mandir.clone().map(PathBuf::from);
        }

        // Store off these values as options because if they're not provided
        // we'll infer default values for them later
        let mut llvm_assertions = None;
        let mut debuginfo_lines = None;
        let mut debuginfo_only_std = None;
        let mut debug = None;
        let mut debug_jemalloc = None;
        let mut debuginfo = None;
        let mut debug_assertions = None;
        let mut optimize = None;

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
            set(&mut config.llvm_enabled, llvm.enabled);
            llvm_assertions = llvm.assertions;
            set(&mut config.llvm_optimize, llvm.optimize);
            set(&mut config.llvm_release_debuginfo, llvm.release_debuginfo);
            set(&mut config.llvm_version_check, llvm.version_check);
            set(&mut config.llvm_static_stdcpp, llvm.static_libstdcpp);
            set(&mut config.llvm_link_shared, llvm.link_shared);
            config.llvm_targets = llvm.targets.clone();
            config.llvm_experimental_targets = llvm.experimental_targets.clone();
            config.llvm_link_jobs = llvm.link_jobs;
        }

        if let Some(ref rust) = toml.rust {
            debug = rust.debug;
            debug_assertions = rust.debug_assertions;
            debuginfo = rust.debuginfo;
            debuginfo_lines = rust.debuginfo_lines;
            debuginfo_only_std = rust.debuginfo_only_std;
            optimize = rust.optimize;
            debug_jemalloc = rust.debug_jemalloc;
            set(&mut config.rust_optimize_tests, rust.optimize_tests);
            set(&mut config.rust_debuginfo_tests, rust.debuginfo_tests);
            set(&mut config.codegen_tests, rust.codegen_tests);
            set(&mut config.rust_rpath, rust.rpath);
            set(&mut config.use_jemalloc, rust.use_jemalloc);
            set(&mut config.backtrace, rust.backtrace);
            set(&mut config.channel, rust.channel.clone());
            set(&mut config.ignore_git, rust.ignore_git);
            set(&mut config.rust_dist_src, rust.dist_src);
            set(&mut config.quiet_tests, rust.quiet_tests);
            config.rustc_default_linker = rust.default_linker.clone();
            config.rustc_default_ar = rust.default_ar.clone();
            config.musl_root = rust.musl_root.clone().map(PathBuf::from);

            match rust.codegen_units {
                Some(0) => config.rust_codegen_units = num_cpus::get() as u32,
                Some(n) => config.rust_codegen_units = n,
                None => {}
            }
        }

        if let Some(ref t) = toml.target {
            for (triple, cfg) in t {
                let mut target = Target::default();

                if let Some(ref s) = cfg.llvm_config {
                    target.llvm_config = Some(env::current_dir().unwrap().join(s));
                }
                if let Some(ref s) = cfg.jemalloc {
                    target.jemalloc = Some(env::current_dir().unwrap().join(s));
                }
                if let Some(ref s) = cfg.android_ndk {
                    target.ndk = Some(env::current_dir().unwrap().join(s));
                }
                target.cxx = cfg.cxx.clone().map(PathBuf::from);
                target.cc = cfg.cc.clone().map(PathBuf::from);
                target.crt_static = cfg.crt_static.clone();
                target.musl_root = cfg.musl_root.clone().map(PathBuf::from);
                target.qemu_rootfs = cfg.qemu_rootfs.clone().map(PathBuf::from);

                config.target_config.insert(INTERNER.intern_string(triple.clone()), target);
            }
        }

        if let Some(ref t) = toml.dist {
            config.dist_sign_folder = t.sign_folder.clone().map(PathBuf::from);
            config.dist_gpg_password_file = t.gpg_password_file.clone().map(PathBuf::from);
            config.dist_upload_addr = t.upload_addr.clone();
            set(&mut config.rust_dist_src, t.src_tarball);
        }

        let cwd = t!(env::current_dir());
        let out = cwd.join("build");

        let stage0_root = out.join(&config.build).join("stage0/bin");
        config.initial_rustc = match build.rustc {
            Some(s) => PathBuf::from(s),
            None => stage0_root.join(exe("rustc", &config.build)),
        };
        config.initial_cargo = match build.cargo {
            Some(s) => PathBuf::from(s),
            None => stage0_root.join(exe("cargo", &config.build)),
        };

        // Now that we've reached the end of our configuration, infer the
        // default values for all options that we haven't otherwise stored yet.

        let default = config.channel == "nightly";
        config.llvm_assertions = llvm_assertions.unwrap_or(default);

        let default = match &config.channel[..] {
            "stable" | "beta" | "nightly" => true,
            _ => false,
        };
        config.rust_debuginfo_lines = debuginfo_lines.unwrap_or(default);
        config.rust_debuginfo_only_std = debuginfo_only_std.unwrap_or(default);

        let default = debug == Some(true);
        config.debug_jemalloc = debug_jemalloc.unwrap_or(default);
        config.rust_debuginfo = debuginfo.unwrap_or(default);
        config.rust_debug_assertions = debug_assertions.unwrap_or(default);
        config.rust_optimize = optimize.unwrap_or(!default);

        config
    }

    pub fn verbose(&self) -> bool {
        self.verbose > 0
    }

    pub fn very_verbose(&self) -> bool {
        self.verbose > 1
    }
}

fn set<T>(field: &mut T, val: Option<T>) {
    if let Some(v) = val {
        *field = v;
    }
}
