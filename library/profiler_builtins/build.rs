//! Compiles the profiler part of the `compiler-rt` library.
//!
//! Loosely based on:
//! - LLVM's `compiler-rt/lib/profile/CMakeLists.txt`
//! - <https://github.com/rust-lang/compiler-builtins/blob/master/build.rs>.

use std::env;
use std::path::PathBuf;

fn main() {
    if let Ok(rt) = tracked_env_var("LLVM_PROFILER_RT_LIB") {
        println!("cargo::rustc-link-lib=static:+verbatim={rt}");
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS was not set");
    let target_env = env::var("CARGO_CFG_TARGET_ENV").expect("CARGO_CFG_TARGET_ENV was not set");
    let cfg = &mut cc::Build::new();

    let profile_sources = vec![
        // tidy-alphabetical-start
        "GCDAProfiling.c",
        "InstrProfiling.c",
        "InstrProfilingBuffer.c",
        "InstrProfilingFile.c",
        "InstrProfilingInternal.c",
        "InstrProfilingMerge.c",
        "InstrProfilingMergeFile.c",
        "InstrProfilingNameVar.c",
        "InstrProfilingPlatformAIX.c",
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
        "WindowsMMap.c",
        // tidy-alphabetical-end
    ];

    if target_env == "msvc" {
        // Don't pull in extra libraries on MSVC
        cfg.flag("/Zl");
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

    // Get the LLVM `compiler-rt` directory from bootstrap.
    let root = PathBuf::from(tracked_env_var_or_fallback(
        "RUST_COMPILER_RT_FOR_PROFILER",
        "../../src/llvm-project/compiler-rt",
    ));

    let src_root = root.join("lib").join("profile");
    assert!(src_root.exists(), "profiler runtime source directory not found: {src_root:?}");
    println!("cargo::rerun-if-changed={}", src_root.display());
    for file in profile_sources {
        cfg.file(src_root.join(file));
    }

    let include = root.join("include");
    println!("cargo::rerun-if-changed={}", include.display());
    cfg.include(include);

    cfg.warnings(false);
    cfg.compile("profiler-rt");
}

fn tracked_env_var(key: &str) -> Result<String, env::VarError> {
    println!("cargo::rerun-if-env-changed={key}");
    env::var(key)
}
fn tracked_env_var_or_fallback(key: &str, fallback: &str) -> String {
    tracked_env_var(key).unwrap_or_else(|_| {
        println!("cargo::warning={key} was not set; falling back to {fallback:?}");
        fallback.to_owned()
    })
}
