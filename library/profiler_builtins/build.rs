//! Compiles the profiler part of the `compiler-rt` library.
//!
//! See the build.rs for libcompiler_builtins crate for details.

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=LLVM_PROFILER_RT_LIB");
    if let Ok(rt) = env::var("LLVM_PROFILER_RT_LIB") {
        println!("cargo:rustc-link-lib=static:+verbatim={rt}");
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS was not set");
    let target_env = env::var("CARGO_CFG_TARGET_ENV").expect("CARGO_CFG_TARGET_ENV was not set");
    let cfg = &mut cc::Build::new();

    // FIXME: `rerun-if-changed` directives are not currently emitted and the build script
    // will not rerun on changes in these source files or headers included into them.
    let mut profile_sources = vec![
        "GCDAProfiling.c",
        "InstrProfiling.c",
        "InstrProfilingBuffer.c",
        "InstrProfilingFile.c",
        "InstrProfilingMerge.c",
        "InstrProfilingMergeFile.c",
        "InstrProfilingNameVar.c",
        "InstrProfilingPlatformDarwin.c",
        "InstrProfilingPlatformFuchsia.c",
        "InstrProfilingPlatformLinux.c",
        "InstrProfilingPlatformOther.c",
        "InstrProfilingPlatformWindows.c",
        "InstrProfilingRuntime.cpp",
        "InstrProfilingUtil.c",
        "InstrProfilingValue.c",
        "InstrProfilingVersionVar.c",
        "InstrProfilingWriter.c",
        // These files were added in LLVM 11.
        "InstrProfilingInternal.c",
        "InstrProfilingBiasVar.c",
    ];

    if target_env == "msvc" {
        // Don't pull in extra libraries on MSVC
        cfg.flag("/Zl");
        profile_sources.push("WindowsMMap.c");
        cfg.define("strdup", Some("_strdup"));
        cfg.define("open", Some("_open"));
        cfg.define("fdopen", Some("_fdopen"));
        cfg.define("getpid", Some("_getpid"));
        cfg.define("fileno", Some("_fileno"));
    } else {
        // Turn off various features of gcc and such, mostly copying
        // compiler-rt's build system already
        cfg.flag("-fno-builtin");
        cfg.flag("-fomit-frame-pointer");
        cfg.define("VISIBILITY_HIDDEN", None);
        if target_os != "windows" {
            cfg.flag("-fvisibility=hidden");
            cfg.define("COMPILER_RT_HAS_UNAME", Some("1"));
        } else {
            profile_sources.push("WindowsMMap.c");
        }
    }

    // Assume that the Unixes we are building this for have fnctl() available
    if env::var_os("CARGO_CFG_UNIX").is_some() {
        cfg.define("COMPILER_RT_HAS_FCNTL_LCK", Some("1"));
    }

    // This should be a pretty good heuristic for when to set
    // COMPILER_RT_HAS_ATOMICS
    if env::var_os("CARGO_CFG_TARGET_HAS_ATOMIC")
        .map(|features| features.to_string_lossy().to_lowercase().contains("ptr"))
        .unwrap_or(false)
    {
        cfg.define("COMPILER_RT_HAS_ATOMICS", Some("1"));
    }

    // Note that this should exist if we're going to run (otherwise we just
    // don't build profiler builtins at all).
    let root = Path::new("../../src/llvm-project/compiler-rt");

    let src_root = root.join("lib").join("profile");
    for src in profile_sources {
        let path = src_root.join(src);
        if path.exists() {
            cfg.file(path);
        }
    }

    cfg.include(root.join("include"));
    cfg.warnings(false);
    cfg.compile("profiler-rt");
}
