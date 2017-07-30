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
//! This module implements parsing `config.mk` and `config.toml` configuration
//! files to tweak how the build runs.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::PathBuf;
use std::process;
use std::cmp;

use num_cpus;
use toml;
use util::{exe, push_exe_path};
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
    pub configure_args: Vec<String>,
    pub openssl_static: bool,


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
        config.build = flags.build;
        config.channel = "dev".to_string();
        config.codegen_tests = true;
        config.rust_dist_src = true;

        config.on_fail = flags.on_fail;
        config.stage = flags.stage;
        config.src = flags.src;
        config.jobs = flags.jobs;
        config.cmd = flags.cmd;
        config.incremental = flags.incremental;
        config.keep_stage = flags.keep_stage;

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
        config.verbose = cmp::max(config.verbose, flags.verbose);

        if let Some(ref install) = toml.install {
            config.prefix = install.prefix.clone().map(PathBuf::from);
            config.sysconfdir = install.sysconfdir.clone().map(PathBuf::from);
            config.docdir = install.docdir.clone().map(PathBuf::from);
            config.bindir = install.bindir.clone().map(PathBuf::from);
            config.libdir = install.libdir.clone().map(PathBuf::from);
            config.mandir = install.mandir.clone().map(PathBuf::from);
        }

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
            set(&mut config.llvm_assertions, llvm.assertions);
            set(&mut config.llvm_optimize, llvm.optimize);
            set(&mut config.llvm_release_debuginfo, llvm.release_debuginfo);
            set(&mut config.llvm_version_check, llvm.version_check);
            set(&mut config.llvm_static_stdcpp, llvm.static_libstdcpp);
            config.llvm_targets = llvm.targets.clone();
            config.llvm_experimental_targets = llvm.experimental_targets.clone();
            config.llvm_link_jobs = llvm.link_jobs;
        }

        if let Some(ref rust) = toml.rust {
            set(&mut config.rust_debug_assertions, rust.debug_assertions);
            set(&mut config.rust_debuginfo, rust.debuginfo);
            set(&mut config.rust_debuginfo_lines, rust.debuginfo_lines);
            set(&mut config.rust_debuginfo_only_std, rust.debuginfo_only_std);
            set(&mut config.rust_optimize, rust.optimize);
            set(&mut config.rust_optimize_tests, rust.optimize_tests);
            set(&mut config.rust_debuginfo_tests, rust.debuginfo_tests);
            set(&mut config.codegen_tests, rust.codegen_tests);
            set(&mut config.rust_rpath, rust.rpath);
            set(&mut config.debug_jemalloc, rust.debug_jemalloc);
            set(&mut config.use_jemalloc, rust.use_jemalloc);
            set(&mut config.backtrace, rust.backtrace);
            set(&mut config.channel, rust.channel.clone());
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

        // compat with `./configure` while we're still using that
        if fs::metadata("config.mk").is_ok() {
            config.update_with_config_mk();
        }

        config
    }

    /// "Temporary" routine to parse `config.mk` into this configuration.
    ///
    /// While we still have `./configure` this implements the ability to decode
    /// that configuration into this. This isn't exactly a full-blown makefile
    /// parser, but hey it gets the job done!
    fn update_with_config_mk(&mut self) {
        let mut config = String::new();
        File::open("config.mk").unwrap().read_to_string(&mut config).unwrap();
        for line in config.lines() {
            let mut parts = line.splitn(2, ":=").map(|s| s.trim());
            let key = parts.next().unwrap();
            let value = match parts.next() {
                Some(n) if n.starts_with('\"') => &n[1..n.len() - 1],
                Some(n) => n,
                None => continue
            };

            macro_rules! check {
                ($(($name:expr, $val:expr),)*) => {
                    if value == "1" {
                        $(
                            if key == concat!("CFG_ENABLE_", $name) {
                                $val = true;
                                continue
                            }
                            if key == concat!("CFG_DISABLE_", $name) {
                                $val = false;
                                continue
                            }
                        )*
                    }
                }
            }

            check! {
                ("MANAGE_SUBMODULES", self.submodules),
                ("COMPILER_DOCS", self.compiler_docs),
                ("DOCS", self.docs),
                ("LLVM_ASSERTIONS", self.llvm_assertions),
                ("LLVM_RELEASE_DEBUGINFO", self.llvm_release_debuginfo),
                ("OPTIMIZE_LLVM", self.llvm_optimize),
                ("LLVM_VERSION_CHECK", self.llvm_version_check),
                ("LLVM_STATIC_STDCPP", self.llvm_static_stdcpp),
                ("LLVM_LINK_SHARED", self.llvm_link_shared),
                ("OPTIMIZE", self.rust_optimize),
                ("DEBUG_ASSERTIONS", self.rust_debug_assertions),
                ("DEBUGINFO", self.rust_debuginfo),
                ("DEBUGINFO_LINES", self.rust_debuginfo_lines),
                ("DEBUGINFO_ONLY_STD", self.rust_debuginfo_only_std),
                ("JEMALLOC", self.use_jemalloc),
                ("DEBUG_JEMALLOC", self.debug_jemalloc),
                ("RPATH", self.rust_rpath),
                ("OPTIMIZE_TESTS", self.rust_optimize_tests),
                ("DEBUGINFO_TESTS", self.rust_debuginfo_tests),
                ("QUIET_TESTS", self.quiet_tests),
                ("LOCAL_REBUILD", self.local_rebuild),
                ("NINJA", self.ninja),
                ("CODEGEN_TESTS", self.codegen_tests),
                ("LOCKED_DEPS", self.locked_deps),
                ("VENDOR", self.vendor),
                ("FULL_BOOTSTRAP", self.full_bootstrap),
                ("EXTENDED", self.extended),
                ("SANITIZERS", self.sanitizers),
                ("PROFILER", self.profiler),
                ("DIST_SRC", self.rust_dist_src),
                ("CARGO_OPENSSL_STATIC", self.openssl_static),
            }

            match key {
                "CFG_BUILD" if value.len() > 0 => self.build = INTERNER.intern_str(value),
                "CFG_HOST" if value.len() > 0 => {
                    self.hosts.extend(value.split(" ").map(|s| INTERNER.intern_str(s)));

                }
                "CFG_TARGET" if value.len() > 0 => {
                    self.targets.extend(value.split(" ").map(|s| INTERNER.intern_str(s)));
                }
                "CFG_EXPERIMENTAL_TARGETS" if value.len() > 0 => {
                    self.llvm_experimental_targets = Some(value.to_string());
                }
                "CFG_MUSL_ROOT" if value.len() > 0 => {
                    self.musl_root = Some(parse_configure_path(value));
                }
                "CFG_MUSL_ROOT_X86_64" if value.len() > 0 => {
                    let target = INTERNER.intern_str("x86_64-unknown-linux-musl");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.musl_root = Some(parse_configure_path(value));
                }
                "CFG_MUSL_ROOT_I686" if value.len() > 0 => {
                    let target = INTERNER.intern_str("i686-unknown-linux-musl");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.musl_root = Some(parse_configure_path(value));
                }
                "CFG_MUSL_ROOT_ARM" if value.len() > 0 => {
                    let target = INTERNER.intern_str("arm-unknown-linux-musleabi");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.musl_root = Some(parse_configure_path(value));
                }
                "CFG_MUSL_ROOT_ARMHF" if value.len() > 0 => {
                    let target = INTERNER.intern_str("arm-unknown-linux-musleabihf");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.musl_root = Some(parse_configure_path(value));
                }
                "CFG_MUSL_ROOT_ARMV7" if value.len() > 0 => {
                    let target = INTERNER.intern_str("armv7-unknown-linux-musleabihf");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.musl_root = Some(parse_configure_path(value));
                }
                "CFG_DEFAULT_AR" if value.len() > 0 => {
                    self.rustc_default_ar = Some(value.to_string());
                }
                "CFG_DEFAULT_LINKER" if value.len() > 0 => {
                    self.rustc_default_linker = Some(value.to_string());
                }
                "CFG_GDB" if value.len() > 0 => {
                    self.gdb = Some(parse_configure_path(value));
                }
                "CFG_RELEASE_CHANNEL" => {
                    self.channel = value.to_string();
                }
                "CFG_PREFIX" => {
                    self.prefix = Some(PathBuf::from(value));
                }
                "CFG_SYSCONFDIR" => {
                    self.sysconfdir = Some(PathBuf::from(value));
                }
                "CFG_DOCDIR" => {
                    self.docdir = Some(PathBuf::from(value));
                }
                "CFG_BINDIR" => {
                    self.bindir = Some(PathBuf::from(value));
                }
                "CFG_LIBDIR" => {
                    self.libdir = Some(PathBuf::from(value));
                }
                "CFG_LIBDIR_RELATIVE" => {
                    self.libdir_relative = Some(PathBuf::from(value));
                }
                "CFG_MANDIR" => {
                    self.mandir = Some(PathBuf::from(value));
                }
                "CFG_LLVM_ROOT" if value.len() > 0 => {
                    let target = self.target_config.entry(self.build.clone())
                                     .or_insert(Target::default());
                    let root = parse_configure_path(value);
                    target.llvm_config = Some(push_exe_path(root, &["bin", "llvm-config"]));
                }
                "CFG_JEMALLOC_ROOT" if value.len() > 0 => {
                    let target = self.target_config.entry(self.build.clone())
                                     .or_insert(Target::default());
                    target.jemalloc = Some(parse_configure_path(value).join("libjemalloc_pic.a"));
                }
                "CFG_ARM_LINUX_ANDROIDEABI_NDK" if value.len() > 0 => {
                    let target = INTERNER.intern_str("arm-linux-androideabi");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.ndk = Some(parse_configure_path(value));
                }
                "CFG_ARMV7_LINUX_ANDROIDEABI_NDK" if value.len() > 0 => {
                    let target = INTERNER.intern_str("armv7-linux-androideabi");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.ndk = Some(parse_configure_path(value));
                }
                "CFG_I686_LINUX_ANDROID_NDK" if value.len() > 0 => {
                    let target = INTERNER.intern_str("i686-linux-android");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.ndk = Some(parse_configure_path(value));
                }
                "CFG_AARCH64_LINUX_ANDROID_NDK" if value.len() > 0 => {
                    let target = INTERNER.intern_str("aarch64-linux-android");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.ndk = Some(parse_configure_path(value));
                }
                "CFG_X86_64_LINUX_ANDROID_NDK" if value.len() > 0 => {
                    let target = INTERNER.intern_str("x86_64-linux-android");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.ndk = Some(parse_configure_path(value));
                }
                "CFG_LOCAL_RUST_ROOT" if value.len() > 0 => {
                    let path = parse_configure_path(value);
                    self.initial_rustc = push_exe_path(path.clone(), &["bin", "rustc"]);
                    self.initial_cargo = push_exe_path(path, &["bin", "cargo"]);
                }
                "CFG_PYTHON" if value.len() > 0 => {
                    let path = parse_configure_path(value);
                    self.python = Some(path);
                }
                "CFG_ENABLE_CCACHE" if value == "1" => {
                    self.ccache = Some(exe("ccache", &self.build));
                }
                "CFG_ENABLE_SCCACHE" if value == "1" => {
                    self.ccache = Some(exe("sccache", &self.build));
                }
                "CFG_CONFIGURE_ARGS" if value.len() > 0 => {
                    self.configure_args = value.split_whitespace()
                                               .map(|s| s.to_string())
                                               .collect();
                }
                "CFG_QEMU_ARMHF_ROOTFS" if value.len() > 0 => {
                    let target = INTERNER.intern_str("arm-unknown-linux-gnueabihf");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.qemu_rootfs = Some(parse_configure_path(value));
                }
                "CFG_QEMU_AARCH64_ROOTFS" if value.len() > 0 => {
                    let target = INTERNER.intern_str("aarch64-unknown-linux-gnu");
                    let target = self.target_config.entry(target).or_insert(Target::default());
                    target.qemu_rootfs = Some(parse_configure_path(value));
                }
                _ => {}
            }
        }
    }

    pub fn verbose(&self) -> bool {
        self.verbose > 0
    }

    pub fn very_verbose(&self) -> bool {
        self.verbose > 1
    }
}

#[cfg(not(windows))]
fn parse_configure_path(path: &str) -> PathBuf {
    path.into()
}

#[cfg(windows)]
fn parse_configure_path(path: &str) -> PathBuf {
    // on windows, configure produces unix style paths e.g. /c/some/path but we
    // only want real windows paths

    use std::process::Command;
    use build_helper;

    // '/' is invalid in windows paths, so we can detect unix paths by the presence of it
    if !path.contains('/') {
        return path.into();
    }

    let win_path = build_helper::output(Command::new("cygpath").arg("-w").arg(path));
    let win_path = win_path.trim();

    win_path.into()
}

fn set<T>(field: &mut T, val: Option<T>) {
    if let Some(v) = val {
        *field = v;
    }
}
