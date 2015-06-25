// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Build the runtime libraries for the compiler. There are two types
//! of runtime libraries, `RtLib` and `ExtRtLib`. `ExtRtLib`s have
//! external build system which we will simply run to build them.
//! For `RtLib`s we invoke the toolchain directly (through `mod cc`)
//! and ignore its build system even if it has one (for compiler-rt).
//! We also generate the `llvmdeps.rs` and `rustc_llvm.def` file.

use std::process::Command;
use std::fs::{create_dir_all, copy};
use std::path::PathBuf;
use std::ffi::OsString;
use build_state::*;
use configure::*;
use cc::{Triple, build_static_lib};
use log::Tee;

fn rt_src_dir(args : &ConfigArgs) -> PathBuf {
    args.src_dir().join("rt")
}

pub fn rt_build_dir(args : &ConfigArgs, triple : &Triple) -> PathBuf {
    args.target_build_dir(triple).join("rt")
}

struct RtLib {
    name : &'static str,
    needed : bool,
    src_files : Vec<&'static str>,
    inc_dirs : Vec<&'static str>
}

fn runtime_libraries(target : &Triple) -> Vec<RtLib> {
    vec![
        RtLib {
            name : "hoedown",
            needed : true,
            src_files : vec!["hoedown/src"],
            inc_dirs : vec!["hoedown/src"],
        },
        RtLib {
            name : "miniz",
            needed : true,
            src_files : vec!["miniz.c"],
            inc_dirs : vec![],
        },
        RtLib {
            name : "rust_builtin",
            needed : true,
            src_files : vec!["rust_builtin.c", "rust_android_dummy.c"],
            inc_dirs : vec![],
        },
        RtLib {
            name : "rustrt_native",
            needed : true,
            src_files : {
                let mut v = vec!["rust_try.ll"];
                if target.is_linux() {
                    v.push("arch/{arch}/record_sp.S");
                }
                v
            },
            inc_dirs : vec![],
        },
        RtLib {
            name : "rust_test_helpers",
            needed : true,
            src_files : vec!["rust_test_helpers.c"],
            inc_dirs : vec![],
        },
        RtLib {
            name : "morestack",
            needed : !target.is_windows(),
            src_files : vec!["arch/{arch}/morestack.S"],
            inc_dirs : vec![],
        },
        RtLib {
            name : "compiler-rt",
            needed : !target.is_msvc(),
            src_files : vec!["../compiler-rt/lib/builtins",
                             "../compiler-rt/lib/builtins/{arch}"],
            inc_dirs : vec!["../compiler-rt/lib/builtins",
                            "../compiler-rt/SDKS/{os}/usr/include"],
        }
        ]
}

fn parse_dir(s : &str, tgt : &Triple) -> PathBuf {
    let arch = if tgt.is_i686() {
        "i386"
    } else {
        tgt.arch()
    };
    let os = if tgt.is_windows() {
        "win"
    } else {
        tgt.os()
    };
    PathBuf::from(s).iter().map(|d| {
        if d == "{arch}" {
            PathBuf::from(arch)
        } else if d == "{os}" {
            PathBuf::from(os)
        } else {
            PathBuf::from(d)
        }
    }).collect()
}

fn build_rt_lib(args : &ConfigArgs,
                triple : &Triple,
                rtlib : &RtLib)
                -> BuildState<()> {
    if !rtlib.needed {
        return continue_build();
    }
    let name = format!("lib{}", rtlib.name);
    println!("Building {} for target triple {}...", name, triple);
    let src_dir = rt_src_dir(args);
    let build_dir = rt_build_dir(args, triple);
    let src_files : Vec<PathBuf> = rtlib.src_files.iter()
        .map(|d| parse_dir(d, triple)).collect();
    let inc_dirs : Vec<PathBuf> = rtlib.inc_dirs.iter()
        .map(|d| parse_dir(d, triple))
        .map(|d| src_dir.join(d)).collect();
    let logger = args.get_logger(triple, &name);
    build_static_lib(args, triple)
        .set_src_dir(&src_dir)
        .set_build_dir(&build_dir)
        .files(&src_files)
        .include_dirs(&inc_dirs)
        .compile(&rtlib.name, &logger)
}

struct ExtRtLib {
    name : &'static str,
    needed : bool,
    env_vars : Vec<(OsString, OsString)>,
    config_cmd : &'static str,
    config_args : Vec<OsString>,
    make_cmd : &'static str,
    make_args : Vec<OsString>,
    build_artefact_src : PathBuf,
    build_artefact_dest : PathBuf
}

fn jemalloc_config_args(cfg : &ConfigArgs, target : &Triple)
                        -> Vec<OsString> {
    vec![ cfg.src_dir().join("jemalloc").join("configure").into(),
          "--with-jemalloc-prefix=je_".into(),
          "--disable-fill".into(),
          format!("--build={}", cfg.build_triple()).into(),
          format!("--host={}", target).into() ]
}

fn libbacktrace_src_dir(cfg : &ConfigArgs) -> PathBuf {
    cfg.src_dir().join("libbacktrace")
}

fn libbacktrace_config_args(cfg : &ConfigArgs,
                            target : &Triple) -> Vec<OsString> {
    vec![ libbacktrace_src_dir(cfg).join("configure").into(),
          format!("--host={}", cfg.build_triple()).into(),
          format!("--target={}", target).into() ]
}

fn external_rt_libs(cfg : &ConfigArgs, triple : &Triple) -> Vec<ExtRtLib> {
    vec![
        ExtRtLib {
            name : "jemalloc",
            needed : !triple.is_windows(),
            env_vars : vec![],
            config_cmd : "bash",
            config_args : jemalloc_config_args(cfg, triple),
            make_cmd : "make",
            make_args : vec![cfg.jnproc()],
            build_artefact_src : PathBuf::from("lib")
                .join("libjemalloc_pic.a"),
            build_artefact_dest : PathBuf::from("libjemalloc.a")
        },
        ExtRtLib {
            name : "libbacktrace",
            needed : triple.is_linux(),
            env_vars : vec![("CFLAGS".into(),
                             "-fPIC -fno-stack-protector".into())],
            config_cmd : "bash",
            config_args : libbacktrace_config_args(cfg, triple),
            make_cmd : "make",
            make_args : vec![cfg.jnproc(),
                             { let mut s = OsString::new();
                               s.push("INCDIR=");
                               s.push(libbacktrace_src_dir(cfg));
                               s
                             }],
            build_artefact_src : PathBuf::from(".libs").join("libbacktrace.a"),
            build_artefact_dest : PathBuf::from("libbacktrace.a")
        }
        ]
}

fn build_external_rt_lib(args : &ConfigArgs,
                         triple : &Triple,
                         rtlib : &ExtRtLib)
                         -> BuildState<()> {
    let name = rtlib.name;
    let build_dir = rt_build_dir(args, triple).join(name);
    let logger = args.get_logger(triple, name);
    let _ = create_dir_all(&build_dir); // errors ignored
    if rtlib.config_cmd != "" {
        println!("Configuring {} for target triple {}...", name, triple);
        let mut cfg_cmd = Command::new(rtlib.config_cmd);
        cfg_cmd.args(&rtlib.config_args)
            .current_dir(&build_dir);
        for &(ref k, ref v) in &rtlib.env_vars {
            cfg_cmd.env(&k, &v);
        }
        try!(cfg_cmd.tee(&logger));
    }
    if rtlib.make_cmd != "" {
        println!("Building {} for target triple {}...", name, triple);
        try!(Command::new(rtlib.make_cmd)
             .args(&rtlib.make_args)
             .current_dir(&build_dir)
             .tee(&logger));
    }
    try!(copy(&build_dir.join(&rtlib.build_artefact_src),
              &rt_build_dir(args, triple).join(&rtlib.build_artefact_dest))
         .map_err(|e| format!("Failed to copy build artefact for {}: {}",
                              name, e)));
    continue_build()
}

fn build_rustllvm(cfg : &ConfigArgs, target : &Triple) -> BuildState<()> {
    println!("Building librustllvm for target triple {}...", target);
    let logger = cfg.get_logger(target, "rustllvm");
    let build_dir = rt_build_dir(cfg, target);
    let src_dir = cfg.src_dir().join("rustllvm");
    let src_files = vec!["ExecutionEngineWrapper.cpp",
                         "PassWrapper.cpp", "RustWrapper.cpp"];
    build_static_lib(cfg, target)
        .set_src_dir(&src_dir)
        .set_build_dir(&build_dir)
        .files(&src_files)
        .set_llvm_cxxflags()
        .compile("rustllvm", &logger)
}

pub fn llvmdeps_path(cfg : &ConfigArgs, target : &Triple) -> PathBuf {
    rt_build_dir(cfg, target).join("llvmdeps.rs")
}

fn generate_llvmdeps(cfg : &ConfigArgs, target : &Triple)
                     -> BuildState<()> {
    println!("Generating llvmdeps.rs for target triple {}...", target);
    let logger = cfg.get_logger(target, "llvmdeps");
    let script = cfg.src_dir().join("etc").join("mklldeps.py");
    let dest = llvmdeps_path(cfg, target);
    let llvm_components = "x86 arm aarch64 mips powerpc ipo bitreader bitwriter linker asmparser mcjit interpreter instrumentation";
    let llvm_enable_static_libcpp = ""; // FIXME : add support
    Command::new("python")
        .arg(&script)
        .arg(&dest)
        .arg(llvm_components)
        .arg(llvm_enable_static_libcpp)
        .arg(&cfg.llvm_tools(target).path_to_llvm_config())
        .tee(&logger)
}

pub fn llvmdef_path(cfg : &ConfigArgs, target : &Triple) -> PathBuf {
    rt_build_dir(cfg, target).join("rustc_llvm.def")
}

fn generate_llvmdef(cfg : &ConfigArgs, target : &Triple)
                    -> BuildState<()> {
    println!("Generating rustc_llvm.def for target triple {}...", target);
    let logger = cfg.get_logger(target, "llvmdef");
    let script = cfg.src_dir().join("etc").join("mklldef.py");
    let src = cfg.src_dir().join("librustc_llvm").join("lib.rs");
    let dest = llvmdef_path(cfg, target);
    let arg = format!("rustc_llvm-{}", cfg.get_git_hash());
    Command::new("python")
        .arg(&script)
        .arg(&src)
        .arg(&dest)
        .arg(&arg)
        .tee(&logger)
}

pub fn build_native_libs(args : &ConfigArgs) -> BuildState<()> {
    let mut triples : Vec<Triple> = vec![];
    let _ : Vec<_> = args.target_triples().iter()
        .map(|t| triples.push(t.clone())).collect();
    if triples.iter().filter(|&t| t == args.build_triple()).count() == 0 {
        triples.push(args.build_triple().clone());
    }
    for triple in &triples {
        for extlib in &external_rt_libs(args, triple) {
            if extlib.needed {
                try!(build_external_rt_lib(args, triple, extlib));
            }
        }
        for rtlib in &runtime_libraries(triple) {
            try!(build_rt_lib(args, triple, rtlib));
        }
        try!(build_rustllvm(args, triple));
        try!(generate_llvmdeps(args, triple));
        if triple.is_msvc() {
            try!(generate_llvmdef(args, triple));
        }
    }
    continue_build()
}
