// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Compiles the profiler part of the `compiler-rt` library.
//!
//! See the build.rs for libcompiler_builtins crate for details.

extern crate gcc;

use std::env;
use std::path::Path;

fn main() {
    let target = env::var("TARGET").expect("TARGET was not set");
    let cfg = &mut gcc::Config::new();

    let mut profile_sources = vec!["GCDAProfiling.c",
                                   "InstrProfiling.c",
                                   "InstrProfilingBuffer.c",
                                   "InstrProfilingFile.c",
                                   "InstrProfilingMerge.c",
                                   "InstrProfilingMergeFile.c",
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
    } else {
        // Turn off various features of gcc and such, mostly copying
        // compiler-rt's build system already
        cfg.flag("-fno-builtin");
        cfg.flag("-fvisibility=hidden");
        cfg.flag("-fomit-frame-pointer");
        cfg.flag("-ffreestanding");
        cfg.define("VISIBILITY_HIDDEN", None);
    }

    for src in profile_sources {
        cfg.file(Path::new("../compiler-rt/lib/profile").join(src));
    }

    cfg.compile("libprofiler-rt.a");
}
