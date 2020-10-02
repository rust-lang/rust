//! Compiles the profiler part of the `compiler-rt` library.
//!
//! See the build.rs for libcompiler_builtins crate for details.

use std::env;
use std::path::Path;

fn main() {
    let target = env::var("TARGET").expect("TARGET was not set");
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
        "InstrProfilingUtil.c",
        "InstrProfilingValue.c",
        "InstrProfilingWriter.c",
        // This file was renamed in LLVM 10.
        "InstrProfilingRuntime.cc",
        "InstrProfilingRuntime.cpp",
        // These files were added in LLVM 11.
        "InstrProfilingInternal.c",
        "InstrProfilingBiasVar.c",
    ];

    if target.contains("msvc") {
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
        if !target.contains("windows") {
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
