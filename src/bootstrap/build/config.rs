// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
/// filled out from the decoded forms of the structs below.
#[derive(Default)]
pub struct Config {
    pub ccache: bool,
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

    pub build: String,
    pub host: Vec<String>,
    pub target: Vec<String>,
    pub rustc: Option<String>,
    pub cargo: Option<String>,

    // libstd features
    pub debug_jemalloc: bool,
    pub use_jemalloc: bool,

    // misc
    pub channel: String,
    pub musl_root: Option<PathBuf>,
}

/// Per-target configuration stored in the global configuration structure.
#[derive(Default)]
pub struct Target {
    pub llvm_config: Option<PathBuf>,
    pub jemalloc: Option<PathBuf>,
    pub cc: Option<PathBuf>,
    pub cxx: Option<PathBuf>,
    pub ndk: Option<PathBuf>,
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
}

/// TOML representation of how the LLVM build is configured.
#[derive(RustcDecodable, Default)]
struct Llvm {
    ccache: Option<bool>,
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
    default_linker: Option<String>,
    default_ar: Option<String>,
    channel: Option<String>,
    musl_root: Option<String>,
    rpath: Option<bool>,
}

/// TOML representation of how each build target is configured.
#[derive(RustcDecodable, Default)]
struct TomlTarget {
    llvm_config: Option<String>,
    jemalloc: Option<String>,
    cc: Option<String>,
    cxx: Option<String>,
    android_ndk: Option<String>,
}

impl Config {
    pub fn parse(build: &str, file: Option<PathBuf>) -> Config {
        let mut config = Config::default();
        config.llvm_optimize = true;
        config.use_jemalloc = true;
        config.rust_optimize = true;
        config.submodules = true;
        config.docs = true;
        config.rust_rpath = true;
        config.rust_codegen_units = 1;
        config.build = build.to_string();
        config.channel = "dev".to_string();

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
        config.rustc = build.rustc;
        config.cargo = build.cargo;
        set(&mut config.compiler_docs, build.compiler_docs);
        set(&mut config.docs, build.docs);

        if let Some(ref llvm) = toml.llvm {
            set(&mut config.ccache, llvm.ccache);
            set(&mut config.llvm_assertions, llvm.assertions);
            set(&mut config.llvm_optimize, llvm.optimize);
            set(&mut config.llvm_optimize, llvm.optimize);
            set(&mut config.llvm_version_check, llvm.version_check);
            set(&mut config.llvm_static_stdcpp, llvm.static_libstdcpp);
        }
        if let Some(ref rust) = toml.rust {
            set(&mut config.rust_debug_assertions, rust.debug_assertions);
            set(&mut config.rust_debuginfo, rust.debuginfo);
            set(&mut config.rust_optimize, rust.optimize);
            set(&mut config.rust_rpath, rust.rpath);
            set(&mut config.debug_jemalloc, rust.debug_jemalloc);
            set(&mut config.use_jemalloc, rust.use_jemalloc);
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

                config.target_config.insert(triple.clone(), target);
            }
        }

        return config
    }
}

fn set<T>(field: &mut T, val: Option<T>) {
    if let Some(v) = val {
        *field = v;
    }
}
