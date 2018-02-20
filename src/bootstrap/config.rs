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

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::cmp;
use std::iter;

use num_cpus;
use channel;
use toml;
use util::exe;
use cache::{INTERNER, Interned};
use flags::Flags;
use build_helper::output;
pub use flags::Subcommand;

/// Global configuration for the entire build and/or bootstrap.
///
/// This structure is derived from `config.toml`.
///
/// Note that this structure is not decoded directly into, but rather it is
/// filled out from the decoded forms of the structs below. For documentation
/// each field, see the corresponding fields in
/// `config.toml.example`.
#[derive(Default)]
pub struct Config {
    pub rustc_error_format: Option<String>,
    pub build: Interned<String>,
    pub hosts: Vec<Interned<String>>,
    pub targets: Vec<Interned<String>>,

    // These are either the stage0 downloaded binaries or the locally installed ones.
    pub initial_cargo: PathBuf,
    pub initial_rustc: PathBuf,

    pub run_host_only: bool,
    pub is_sudo: bool,
    pub out: PathBuf,

    pub exclude: Vec<PathBuf>,
    pub on_fail: Option<String>,
    pub stage: Option<u32>,
    pub keep_stage: Option<u32>,
    pub src: PathBuf,
    pub jobs: Option<u32>,
    pub cmd: Subcommand,
    pub paths: Vec<PathBuf>,
    pub incremental: bool,

    pub llvm: Llvm,
    pub rust: Rust,
    pub dist: Dist,

    pub target_config: HashMap<Interned<String>, Target>,
    pub install: Install,
    pub general: Build,
}

/// Per-target configuration stored in the global configuration structure.
#[derive(Default)]
pub struct Target {
    /// Some(path to llvm-config) if using an external LLVM.
    pub llvm_config: Option<PathBuf>,
    pub jemalloc: Option<PathBuf>,
    pub cc: Option<PathBuf>,
    pub cxx: Option<PathBuf>,
    pub ar: Option<PathBuf>,
    pub linker: Option<PathBuf>,
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
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct TomlConfig {
    build: Build,
    install: Install,
    llvm: Llvm,
    rust: Rust,
    target: HashMap<String, TomlTarget>,
    dist: Dist,
}

/// TOML representation of various global build decisions.
#[derive(Deserialize, Clone)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Build {
    #[serde(skip)]
    // We get build from the BUILD env-var, provided by bootstrap.py
    build: String,
    host: Vec<String>,
    target: Vec<String>,

    #[serde(rename = "cargo")]
    pub initial_cargo: Option<PathBuf>,
    #[serde(rename = "rustc")]
    pub initial_rustc: Option<PathBuf>,

    pub low_priority: bool,
    pub compiler_docs: bool,
    pub docs: bool,
    pub submodules: bool,
    pub gdb: Option<PathBuf>,
    pub locked_deps: bool,
    pub vendor: bool,
    pub nodejs: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub full_bootstrap: bool,
    pub extended: bool,
    pub tools: Option<HashSet<String>>,
    pub verbose: usize,
    pub sanitizers: bool,
    pub profiler: bool,
    pub openssl_static: bool,
    pub configure_args: Vec<String>,
    pub local_rebuild: bool,
}

impl Default for Build {
    fn default() -> Build {
        Build {
            build: env::var("BUILD").unwrap(),
            host: Vec::new(),
            target: Vec::new(),
            initial_cargo: None,
            initial_rustc: None,
            low_priority: false,
            compiler_docs: false,
            docs: true,
            submodules: true,
            gdb: None,
            locked_deps: false,
            vendor: false,
            nodejs: None,
            python: None,
            full_bootstrap: false,
            extended: false,
            tools: None,
            verbose: 0,
            sanitizers: false,
            profiler: false,
            openssl_static: false,
            configure_args: Vec::new(),
            local_rebuild: false,
        }
    }
}

/// TOML representation of various global install decisions.
#[derive(Deserialize, Clone)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Install {
    pub prefix: PathBuf,
    pub sysconfdir: PathBuf,
    pub docdir: PathBuf,
    pub bindir: PathBuf,
    pub libdir: PathBuf,
    pub mandir: PathBuf,

    // standard paths, currently unused
    datadir: Option<PathBuf>,
    infodir: Option<PathBuf>,
    localstatedir: Option<PathBuf>,
}

impl Default for Install {
    fn default() -> Install {
        Install {
            prefix: PathBuf::from("/usr/local"),
            sysconfdir: PathBuf::from("/etc"),
            docdir: PathBuf::from("share/doc/rust"),
            bindir: PathBuf::from("bin"),
            libdir: PathBuf::from("lib"),
            mandir: PathBuf::from("share/man"),

            datadir: None,
            infodir: None,
            localstatedir: None,
        }
    }
}

/// TOML representation of how the LLVM build is configured.
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Llvm {
    pub enabled: bool,
    ccache: StringOrBool,
    pub ninja: bool,
    pub assertions: bool,
    pub optimize: bool,
    pub release_debuginfo: bool,
    pub version_check: bool,
    pub static_libstdcpp: bool,
    pub targets: String,
    pub experimental_targets: String,
    pub link_jobs: u32,
    pub link_shared: bool,
}

impl Llvm {
    pub fn ccache(&self) -> Option<String> {
        match self.ccache {
            StringOrBool::String(ref s) => {
                Some(s.to_string())
            }
            StringOrBool::Bool(true) => {
                Some("ccache".to_string())
            }
            StringOrBool::Bool(false) => None,
        }
    }
}

impl Default for Llvm {
    fn default() -> Llvm {
        Llvm {
            enabled: true,
            ccache: StringOrBool::Bool(false),
            ninja: false,
            assertions: false,
            optimize: true,
            release_debuginfo: false,
            version_check: true,
            targets:
                String::from("X86;ARM;AArch64;Mips;PowerPC;SystemZ;MSP430;Sparc;NVPTX;Hexagon"),
            experimental_targets: String::from("WebAssembly"),
            link_jobs: 0,
            static_libstdcpp: false,
            link_shared: false,
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Dist {
    pub sign_folder: Option<PathBuf>,
    pub gpg_password_file: Option<PathBuf>,
    pub upload_addr: Option<String>,
    pub src_tarball: bool,
}

impl Default for Dist {
    fn default() -> Dist {
        Dist {
            sign_folder: None,
            gpg_password_file: None,
            upload_addr: None,
            src_tarball: true,
        }
    }
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
#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Rust {
    pub thinlto: bool,
    pub rpath: bool,
    pub experimental_parallel_queries: bool,
    pub use_jemalloc: bool,
    pub backtrace: bool, // RUST_BACKTRACE support
    pub default_linker: Option<String>,
    pub channel: String,
    // Fallback musl-root for all targets
    pub musl_root: Option<PathBuf>,
    pub optimize_tests: bool,
    pub debuginfo_tests: bool,
    pub codegen_tests: bool,
    pub quiet_tests: bool,
    pub test_miri: bool,
    pub save_toolstates: Option<PathBuf>,
    pub codegen_backends: Vec<String>,
    pub wasm_syscall: bool,
    optimize: Option<bool>,
    codegen_units: Option<u32>,
    debug_assertions: Option<bool>,
    debuginfo: Option<bool>,
    debuginfo_lines: Option<bool>,
    debuginfo_only_std: Option<bool>,
    debug_jemalloc: Option<bool>,
    ignore_git: Option<bool>,
    debug: bool,
}

impl Rust {
    pub fn debug_jemalloc(&self) -> bool {
        self.debug_jemalloc.unwrap_or(self.debug)
    }

    pub fn debuginfo(&self) -> bool {
        self.debuginfo.unwrap_or(self.debug)
    }

    fn is_dist_channel(&self) -> bool {
        match &self.channel[..] {
            "stable" | "beta" | "nightly" => true,
            _ => false,
        }
    }

    pub fn debuginfo_lines(&self) -> bool {
        self.debuginfo_lines.unwrap_or(self.is_dist_channel())
    }

    pub fn debuginfo_only_std(&self) -> bool {
        self.debuginfo_only_std.unwrap_or(self.is_dist_channel())
    }

    pub fn ignore_git(&self) -> bool {
        self.ignore_git.unwrap_or(!self.is_dist_channel())
    }

    pub fn debug_assertions(&self) -> bool {
        self.debug_assertions.unwrap_or(self.debug)
    }

    pub fn optimize(&self) -> bool {
        self.optimize.unwrap_or(!self.debug)
    }

    pub fn codegen_units(&self) -> Option<u32> {
        match self.codegen_units {
            Some(0) => Some(num_cpus::get() as u32),
            Some(n) => Some(n),
            None => None,
        }
    }
}

impl Default for Rust {
    fn default() -> Rust {
        Rust {
            debug: false,
            debug_assertions: None,
            debuginfo: None,
            debuginfo_lines: None,
            debuginfo_only_std: None,
            optimize: None,
            ignore_git: None,
            debug_jemalloc: None,
            thinlto: true,
            optimize_tests: true,
            debuginfo_tests: false,
            codegen_tests: true,
            rpath: true,
            use_jemalloc: true,
            backtrace: true,
            channel: String::from("dev"),
            quiet_tests: false,
            test_miri: false,
            wasm_syscall: false,
            codegen_backends: vec![String::from("llvm")],
            codegen_units: None,
            default_linker: None,
            experimental_parallel_queries: false,
            musl_root: None,
            save_toolstates: None,
        }
    }
}

/// TOML representation of how each build target is configured.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct TomlTarget {
    llvm_config: Option<PathBuf>,
    jemalloc: Option<PathBuf>,
    cc: Option<PathBuf>,
    cxx: Option<PathBuf>,
    ar: Option<PathBuf>,
    linker: Option<PathBuf>,
    android_ndk: Option<PathBuf>,
    crt_static: Option<bool>,
    musl_root: Option<PathBuf>,
    qemu_rootfs: Option<PathBuf>,
}

impl Config {
    pub fn parse(args: &[String]) -> Config {
        let flags = Flags::parse(&args);
        let file = flags.config.clone();
        let mut config = Config::default();
        config.exclude = flags.exclude;
        config.paths = flags.paths;
        config.rustc_error_format = flags.rustc_error_format;
        config.on_fail = flags.on_fail;
        config.stage = flags.stage;
        config.src = flags.src;
        config.jobs = flags.jobs;
        config.cmd = flags.cmd;
        config.incremental = flags.incremental;
        config.keep_stage = flags.keep_stage;

        let cwd = t!(env::current_dir());
        let out = cwd.join("build");
        config.out = out.clone();

        // If --target was specified but --host wasn't specified, don't run any host-only tests.
        config.run_host_only = !(flags.host.is_empty() && !flags.target.is_empty());

        config.is_sudo = match env::var_os("SUDO_USER") {
            Some(sudo_user) => {
                match env::var_os("USER") {
                    Some(user) => user != sudo_user,
                    None => false,
                }
            }
            None => false,
        };

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

        config.general = toml.build.clone();
        // bootstrap.py already handles this fully -- checks flags, toml, and default-generates
        config.build = INTERNER.intern_str(&toml.build.build);
        let mut hosts = toml.build.host.into_iter()
            .map(|h| INTERNER.intern_string(h))
            .chain(iter::once(config.build.clone()))
            .collect::<Vec<_>>();
        hosts.sort_by(|a, b| a.cmp(&b));
        hosts.dedup_by(|a, b| a == b);
        config.hosts = hosts;
        let mut targets = toml.build.target.into_iter()
            .map(|t| INTERNER.intern_string(t))
            .chain(config.hosts.clone())
            .collect::<Vec<_>>();
        targets.sort_by(|a, b| a.cmp(&b));
        targets.dedup_by(|a, b| a == b);
        config.targets = targets;
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

        config.general.verbose = cmp::max(toml.build.verbose, flags.verbose);

        config.install = toml.install;
        config.llvm = toml.llvm;

        config.rust = toml.rust.clone();

        for (triple, cfg) in toml.target {
            let cwd = t!(env::current_dir());
            config.target_config.insert(INTERNER.intern_string(triple.clone()), Target {
                llvm_config: cfg.llvm_config.map(|p| cwd.join(p)),
                jemalloc: cfg.jemalloc.map(|p| cwd.join(p)),
                ndk: cfg.android_ndk.map(|p| cwd.join(p)),
                cc: cfg.cc,
                cxx: cfg.cxx,
                ar: cfg.ar,
                linker: cfg.linker,
                crt_static: cfg.crt_static,
                musl_root: cfg.musl_root,
                qemu_rootfs: cfg.qemu_rootfs,
            });
        }

        config.dist = toml.dist;

        let stage0_root = out.join(&config.build).join("stage0/bin");
        config.initial_rustc = match toml.build.initial_rustc {
            Some(s) => PathBuf::from(s),
            None => stage0_root.join(exe("rustc", &config.build)),
        };
        config.initial_cargo = match toml.build.initial_cargo {
            Some(s) => PathBuf::from(s),
            None => stage0_root.join(exe("cargo", &config.build)),
        };

        // If local-rust is the same major.minor as the current version, then force a local-rebuild
        let local_version_verbose = output(
            Command::new(&config.initial_rustc).arg("--version").arg("--verbose"));
        let local_release = local_version_verbose
            .lines().filter(|x| x.starts_with("release:"))
            .next().unwrap().trim_left_matches("release:").trim();
        let my_version = channel::CFG_RELEASE_NUM;
        if local_release.split('.').take(2).eq(my_version.split('.').take(2)) {
            eprintln!("auto-detected local rebuild");
            config.general.local_rebuild = true;
        }

        // The msvc hosts don't use jemalloc, turn it off globally to
        // avoid packaging the dummy liballoc_jemalloc on that platform.
        if config.hosts.iter().any(|host| host.contains("msvc")) {
            config.rust.use_jemalloc = false;
        }

        config
    }

    /// Try to find the relative path of `libdir`.
    pub fn libdir_relative(&self) -> Option<&Path> {
        let libdir = self.install.libdir.as_ref()?;
        if libdir.is_relative() {
            Some(libdir)
        } else {
            // Try to make it relative to the prefix.
            libdir.strip_prefix(self.prefix.as_ref()?).ok()
        }
    }

    pub fn verbose(&self) -> bool {
        self.general.verbose > 0
    }

    pub fn very_verbose(&self) -> bool {
        self.general.verbose > 1
    }
}
