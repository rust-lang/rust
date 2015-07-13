extern crate build_helper;

use std::path::PathBuf;
use std::process::Command;
use build_helper::{Config, Run, Triple, build_static_lib};

fn main() {
    build_rt_libraries();
    build_backtrace();

    let cfg = Config::new();
    match &cfg.profile()[..] {
        "bench" | "release" => { println!("cargo:rustc-cfg=rtopt"); }
        _ => {}
    }
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
            name : "rust_builtin",
            needed : true,
            src_files : vec!["rt/rust_builtin.c", "rt/rust_android_dummy.c"],
            inc_dirs : vec![]
        },
        RtLib {
            name : "rustrt_native",
            needed : true,
            src_files : {
                let mut v = vec!["rt/rust_try.ll"];
                if target.is_linux() {
                    v.push("rt/arch/{arch}/record_sp.S");
                }
                v
            },
            inc_dirs : vec![]
        },
        RtLib {
            name : "morestack",
            needed : !target.is_windows(),
            src_files : vec!["rt/arch/{arch}/morestack.S"],
            inc_dirs : vec![],
        },
        RtLib {
            name : "compiler-rt",
            needed : !target.is_msvc(), // FIXME: Fix MSVC build for compiler-rt
            src_files : vec!["compiler-rt/lib/builtins",
                             "compiler-rt/lib/builtins/{arch}"],
            inc_dirs : vec!["compiler-rt/lib/builtins",
                            "compiler-rt/SDKS/{os}/usr/include"],
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

fn build_rt_lib(cfg : &Config, rtlib : &RtLib) {
    let target = cfg.target();
    let src_files : Vec<PathBuf> = rtlib.src_files.iter()
        .map(|d| parse_dir(d, target)).collect();
    let inc_dirs : Vec<PathBuf> = rtlib.inc_dirs.iter()
        .map(|d| parse_dir(d, target)).collect();
    build_static_lib(cfg)
        .include_dirs(&inc_dirs)
        .files(&src_files)
        .compile(rtlib.name);
}

fn build_rt_libraries() {
    let cfg = Config::new();
    let libs = runtime_libraries(cfg.target());
    for lib in &libs {
        if lib.needed {
            build_rt_lib(&cfg, lib);
        }
    }
}

fn build_backtrace() {
    let cfg = Config::new();
    if !cfg.target().is_linux() {
        return
    }

    let src_dir = cfg.src_dir().join("libbacktrace");
    let build_dir = cfg.out_dir().join("libbacktrace");
    let _ = std::fs::create_dir_all(&build_dir);
    Command::new(src_dir.join("configure"))
        .current_dir(&build_dir)
        .arg("--with-pic")
        .arg("--disable-multilib")
        .arg("--disable-shared")
        .arg("--disable-host-shared")
        .arg(format!("--build={}", cfg.host()))
        .arg(format!("--host={}", cfg.target()))
        .run();
    Command::new("make")
        .current_dir(&build_dir)
        .arg(format!("-j{}", cfg.njobs()))
        .arg(format!("INCDIR={}", src_dir.display()))
        .run();

    println!("cargo:rustc-link-search=native={}/.libs", build_dir.display());
}
