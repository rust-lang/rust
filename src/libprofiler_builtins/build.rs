//! Compiles the profiler part of the `compiler-rt` library.
//!
//! See the build.rs for libcompiler_builtins crate for details.

use std::env;
use std::path::Path;

fn main() {
    let target = env::var("TARGET").expect("TARGET was not set");
    let cfg = &mut cc::Build::new();

    let mut profile_sources = vec!["GCDAProfiling.c",
                                   "InstrProfiling.c",
                                   "InstrProfilingBuffer.c",
                                   "InstrProfilingFile.c",
                                   "InstrProfilingMerge.c",
                                   "InstrProfilingMergeFile.c",
                                   "InstrProfilingNameVar.c",
                                   "InstrProfilingPlatformDarwin.c",
                                   "InstrProfilingPlatformLinux.c",
                                   "InstrProfilingPlatformOther.c",
                                   "InstrProfilingRuntime.cc",
                                   "InstrProfilingUtil.c",
                                   "InstrProfilingValue.c",
                                   "InstrProfilingWriter.c"];

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
        cfg.flag("-fvisibility=hidden");
        cfg.flag("-fomit-frame-pointer");
        cfg.flag("-ffreestanding");
        cfg.define("VISIBILITY_HIDDEN", None);
        cfg.define("COMPILER_RT_HAS_UNAME", Some("1"));
    }

    // The source for `compiler-rt` comes from the `compiler-builtins` crate, so
    // load our env var set by cargo to find the source code.
    let root = env::var_os("DEP_COMPILER_RT_COMPILER_RT").unwrap();
    let root = Path::new(&root);

    for src in profile_sources {
        cfg.file(root.join("lib").join("profile").join(src));
    }

    cfg.warnings(false);
    cfg.compile("profiler-rt");
}
