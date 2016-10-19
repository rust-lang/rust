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
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::process;

use num_cpus;
use rustc_serialize::Decodable;
use toml::{Parser, Decoder, Value};

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
/// `src/bootstrap/config.toml.example`.
#[derive(Default)]
pub struct Config {
    pub ccache: bool,
    pub ninja: bool,
    pub verbose: bool,
    pub submodules: bool,
    pub compiler_docs: bool,
    pub docs: bool,
    pub target_config: HashMap<String, Target>,

    // llvm codegen options
    pub llvm_assertions: bool,
    pub llvm_optimize: bool,
    pub llvm_version_check: bool,
    pub llvm_static_stdcpp: bool,

    // rust codegen options
    pub rust_optimize: bool,
    pub rust_codegen_units: u32,
    pub rust_debug_assertions: bool,
    pub rust_debuginfo: bool,
    pub rust_rpath: bool,
    pub rustc_default_linker: Option<String>,
    pub rustc_default_ar: Option<String>,
    pub rust_optimize_tests: bool,
    pub rust_debuginfo_tests: bool,

    pub build: String,
    pub host: Vec<String>,
    pub target: Vec<String>,
    pub rustc: Option<PathBuf>,
    pub cargo: Option<PathBuf>,
    pub local_rebuild: bool,

    // libstd features
    pub debug_jemalloc: bool,
    pub use_jemalloc: bool,
    pub backtrace: bool, // support for RUST_BACKTRACE

    // misc
    pub channel: String,
    // Fallback musl-root for all targets
    pub musl_root: Option<PathBuf>,
    pub prefix: Option<String>,
    pub docdir: Option<String>,
    pub libdir: Option<String>,
    pub mandir: Option<String>,
    pub codegen_tests: bool,
    pub nodejs: Option<PathBuf>,
}

/// Per-target configuration stored in the global configuration structure.
#[derive(Default)]
pub struct Target {
    pub llvm_config: Option<PathBuf>,
    pub jemalloc: Option<PathBuf>,
    pub cc: Option<PathBuf>,
    pub cxx: Option<PathBuf>,
    pub ndk: Option<PathBuf>,
    pub musl_root: Option<PathBuf>,
}

/// Structure of the `config.toml` file that configuration is read from.
///
/// This structure uses `Decodable` to automatically decode a TOML configuration
/// file into this format, and then this is traversed and written into the above
/// `Config` structure.
#[derive(RustcDecodable, Default)]
struct TomlConfig {
    build: Option<Build>,
    llvm: Option<Llvm>,
    rust: Option<Rust>,
    target: Option<HashMap<String, TomlTarget>>,
}

/// TOML representation of various global build decisions.
#[derive(RustcDecodable, Default, Clone)]
struct Build {
    build: Option<String>,
    host: Vec<String>,
    target: Vec<String>,
    cargo: Option<String>,
    rustc: Option<String>,
    compiler_docs: Option<bool>,
    docs: Option<bool>,
    submodules: Option<bool>,
}

/// TOML representation of how the LLVM build is configured.
#[derive(RustcDecodable, Default)]
struct Llvm {
    ccache: Option<bool>,
    ninja: Option<bool>,
    assertions: Option<bool>,
    optimize: Option<bool>,
    version_check: Option<bool>,
    static_libstdcpp: Option<bool>,
}

/// TOML representation of how the Rust build is configured.
#[derive(RustcDecodable, Default)]
struct Rust {
    optimize: Option<bool>,
    codegen_units: Option<u32>,
    debug_assertions: Option<bool>,
    debuginfo: Option<bool>,
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
#[derive(RustcDecodable, Default)]
struct TomlTarget {
    llvm_config: Option<String>,
    jemalloc: Option<String>,
    cc: Option<String>,
    cxx: Option<String>,
    android_ndk: Option<String>,
    musl_root: Option<String>,
}

impl Config {
    pub fn parse(build: &str, file: Option<PathBuf>) -> Config {
        let mut config = Config::default();
        config.llvm_optimize = true;
        config.use_jemalloc = true;
        config.backtrace = true;
        config.rust_optimize = true;
        config.rust_optimize_tests = true;
        config.submodules = true;
        config.docs = true;
        config.rust_rpath = true;
        config.rust_codegen_units = 1;
        config.build = build.to_string();
        config.channel = "dev".to_string();
        config.codegen_tests = true;

        let toml = file.map(|file| {
            let mut f = t!(File::open(&file));
            let mut toml = String::new();
            t!(f.read_to_string(&mut toml));
            let mut p = Parser::new(&toml);
            let table = match p.parse() {
                Some(table) => table,
                None => {
                    println!("failed to parse TOML configuration:");
                    for err in p.errors.iter() {
                        let (loline, locol) = p.to_linecol(err.lo);
                        let (hiline, hicol) = p.to_linecol(err.hi);
                        println!("{}:{}-{}:{}: {}", loline, locol, hiline,
                                 hicol, err.desc);
                    }
                    process::exit(2);
                }
            };
            let mut d = Decoder::new(Value::Table(table));
            match Decodable::decode(&mut d) {
                Ok(cfg) => cfg,
                Err(e) => {
                    println!("failed to decode TOML: {}", e);
                    process::exit(2);
                }
            }
        }).unwrap_or_else(|| TomlConfig::default());

        let build = toml.build.clone().unwrap_or(Build::default());
        set(&mut config.build, build.build.clone());
        config.host.push(config.build.clone());
        for host in build.host.iter() {
            if !config.host.contains(host) {
                config.host.push(host.clone());
            }
        }
        for target in config.host.iter().chain(&build.target) {
            if !config.target.contains(target) {
                config.target.push(target.clone());
            }
        }
        config.rustc = build.rustc.map(PathBuf::from);
        config.cargo = build.cargo.map(PathBuf::from);
        set(&mut config.compiler_docs, build.compiler_docs);
        set(&mut config.docs, build.docs);
        set(&mut config.submodules, build.submodules);

        if let Some(ref llvm) = toml.llvm {
            set(&mut config.ccache, llvm.ccache);
            set(&mut config.ninja, llvm.ninja);
            set(&mut config.llvm_assertions, llvm.assertions);
            set(&mut config.llvm_optimize, llvm.optimize);
            set(&mut config.llvm_version_check, llvm.version_check);
            set(&mut config.llvm_static_stdcpp, llvm.static_libstdcpp);
        }
        if let Some(ref rust) = toml.rust {
            set(&mut config.rust_debug_assertions, rust.debug_assertions);
            set(&mut config.rust_debuginfo, rust.debuginfo);
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

                config.target_config.insert(triple.clone(), target);
            }
        }

        return config
    }

    /// "Temporary" routine to parse `config.mk` into this configuration.
    ///
    /// While we still have `./configure` this implements the ability to decode
    /// that configuration into this. This isn't exactly a full-blown makefile
    /// parser, but hey it gets the job done!
    pub fn update_with_config_mk(&mut self) {
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
                ("CCACHE", self.ccache),
                ("MANAGE_SUBMODULES", self.submodules),
                ("COMPILER_DOCS", self.compiler_docs),
                ("DOCS", self.docs),
                ("LLVM_ASSERTIONS", self.llvm_assertions),
                ("OPTIMIZE_LLVM", self.llvm_optimize),
                ("LLVM_VERSION_CHECK", self.llvm_version_check),
                ("LLVM_STATIC_STDCPP", self.llvm_static_stdcpp),
                ("OPTIMIZE", self.rust_optimize),
                ("DEBUG_ASSERTIONS", self.rust_debug_assertions),
                ("DEBUGINFO", self.rust_debuginfo),
                ("JEMALLOC", self.use_jemalloc),
                ("DEBUG_JEMALLOC", self.debug_jemalloc),
                ("RPATH", self.rust_rpath),
                ("OPTIMIZE_TESTS", self.rust_optimize_tests),
                ("DEBUGINFO_TESTS", self.rust_debuginfo_tests),
                ("LOCAL_REBUILD", self.local_rebuild),
                ("NINJA", self.ninja),
                ("CODEGEN_TESTS", self.codegen_tests),
            }

            match key {
                "CFG_BUILD" => self.build = value.to_string(),
                "CFG_HOST" => {
                    self.host = value.split(" ").map(|s| s.to_string())
                                     .collect();
                }
                "CFG_TARGET" => {
                    self.target = value.split(" ").map(|s| s.to_string())
                                       .collect();
                }
                "CFG_MUSL_ROOT" if value.len() > 0 => {
                    self.musl_root = Some(PathBuf::from(value));
                }
                "CFG_MUSL_ROOT_X86_64" if value.len() > 0 => {
                    let target = "x86_64-unknown-linux-musl".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.musl_root = Some(PathBuf::from(value));
                }
                "CFG_MUSL_ROOT_I686" if value.len() > 0 => {
                    let target = "i686-unknown-linux-musl".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.musl_root = Some(PathBuf::from(value));
                }
                "CFG_MUSL_ROOT_ARM" if value.len() > 0 => {
                    let target = "arm-unknown-linux-musleabi".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.musl_root = Some(PathBuf::from(value));
                }
                "CFG_MUSL_ROOT_ARMHF" if value.len() > 0 => {
                    let target = "arm-unknown-linux-musleabihf".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.musl_root = Some(PathBuf::from(value));
                }
                "CFG_MUSL_ROOT_ARMV7" if value.len() > 0 => {
                    let target = "armv7-unknown-linux-musleabihf".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.musl_root = Some(PathBuf::from(value));
                }
                "CFG_DEFAULT_AR" if value.len() > 0 => {
                    self.rustc_default_ar = Some(value.to_string());
                }
                "CFG_DEFAULT_LINKER" if value.len() > 0 => {
                    self.rustc_default_linker = Some(value.to_string());
                }
                "CFG_RELEASE_CHANNEL" => {
                    self.channel = value.to_string();
                }
                "CFG_PREFIX" => {
                    self.prefix = Some(value.to_string());
                }
                "CFG_DOCDIR" => {
                    self.docdir = Some(value.to_string());
                }
                "CFG_LIBDIR" => {
                    self.libdir = Some(value.to_string());
                }
                "CFG_MANDIR" => {
                    self.mandir = Some(value.to_string());
                }
                "CFG_LLVM_ROOT" if value.len() > 0 => {
                    let target = self.target_config.entry(self.build.clone())
                                     .or_insert(Target::default());
                    let root = PathBuf::from(value);
                    target.llvm_config = Some(root.join("bin/llvm-config"));
                }
                "CFG_JEMALLOC_ROOT" if value.len() > 0 => {
                    let target = self.target_config.entry(self.build.clone())
                                     .or_insert(Target::default());
                    target.jemalloc = Some(PathBuf::from(value));
                }
                "CFG_ARM_LINUX_ANDROIDEABI_NDK" if value.len() > 0 => {
                    let target = "arm-linux-androideabi".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.ndk = Some(PathBuf::from(value));
                }
                "CFG_ARMV7_LINUX_ANDROIDEABI_NDK" if value.len() > 0 => {
                    let target = "armv7-linux-androideabi".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.ndk = Some(PathBuf::from(value));
                }
                "CFG_I686_LINUX_ANDROID_NDK" if value.len() > 0 => {
                    let target = "i686-linux-android".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.ndk = Some(PathBuf::from(value));
                }
                "CFG_AARCH64_LINUX_ANDROID_NDK" if value.len() > 0 => {
                    let target = "aarch64-linux-android".to_string();
                    let target = self.target_config.entry(target)
                                     .or_insert(Target::default());
                    target.ndk = Some(PathBuf::from(value));
                }
                "CFG_LOCAL_RUST_ROOT" if value.len() > 0 => {
                    self.rustc = Some(PathBuf::from(value).join("bin/rustc"));
                    self.cargo = Some(PathBuf::from(value).join("bin/cargo"));
                }
                _ => {}
            }
        }
    }
}

fn set<T>(field: &mut T, val: Option<T>) {
    if let Some(v) = val {
        *field = v;
    }
}
