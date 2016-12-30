// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Compilation of native dependencies like LLVM.
//!
//! Native projects like LLVM unfortunately aren't suited just yet for
//! compilation in build scripts that Cargo has. This is because thie
//! compilation takes a *very* long time but also because we don't want to
//! compile LLVM 3 times as part of a normal bootstrap (we want it cached).
//!
//! LLVM and compiler-rt are essentially just wired up to everything else to
//! ensure that they're always in place if needed.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use build_helper::output;
use cmake;
use gcc;

use Build;
use util::{self, up_to_date};

/// Build config for LLVM; mirrors the CMake configuration and can be converted
/// into one.
///
/// The config, after LLVM is successfully built, is serialized and written
/// along with the clean trigger to serve as the stamp. This way LLVM is
/// correctly rebuilt whenever build configuration changes, not only when the
/// clean trigger is touched.
struct LlvmBuildConfig {
    clean_trigger: String,
    path: PathBuf,
    generator: Option<OsString>,
    target: String,
    host: String,
    out_dir: PathBuf,
    profile: String,
    defines: Vec<(OsString, OsString)>,
    build_args: Vec<OsString>,
}

impl From<LlvmBuildConfig> for cmake::Config {
    fn from(mirror: LlvmBuildConfig) -> Self {
        let mut cfg = cmake::Config::new(&mirror.path);

        if let Some(generator) = mirror.generator {
            cfg.generator(&generator);
        }

        cfg.target(&mirror.target)
           .host(&mirror.host)
           .out_dir(&mirror.out_dir)
           .profile(&mirror.profile);

        for (k, v) in mirror.defines {
            cfg.define(k, v);
        }

        for v in mirror.build_args {
            cfg.build_arg(v);
        }

        cfg
    }
}

impl LlvmBuildConfig {
    fn new<S, P>(clean_trigger: S, path: P) -> LlvmBuildConfig
        where S: AsRef<str>, P: AsRef<Path>
    {
        LlvmBuildConfig {
            clean_trigger: clean_trigger.as_ref().to_string(),
            path: path.as_ref().to_path_buf(),
            generator: None,
            target: "".to_string(),
            host: "".to_string(),
            out_dir: PathBuf::new(),
            profile: "".to_string(),
            defines: vec![],
            build_args: vec![],
        }
    }

    fn into_cmake(self) -> cmake::Config {
        self.into()
    }

    fn generator<S: AsRef<OsStr>>(&mut self, v: S) -> &mut LlvmBuildConfig {
        self.generator = Some(v.as_ref().to_owned());
        self
    }

    fn target<S: AsRef<str>>(&mut self, v: S) -> &mut LlvmBuildConfig {
        self.target = v.as_ref().to_string();
        self
    }

    fn host<S: AsRef<str>>(&mut self, v: S) -> &mut LlvmBuildConfig {
        self.host = v.as_ref().to_string();
        self
    }

    fn out_dir<P: AsRef<Path>>(&mut self, v: P) -> &mut LlvmBuildConfig {
        self.out_dir = v.as_ref().to_path_buf();
        self
    }

    fn profile<S: AsRef<str>>(&mut self, v: S) -> &mut LlvmBuildConfig {
        self.profile = v.as_ref().to_string();
        self
    }

    fn define<K, V>(&mut self, k: K, v: V) -> &mut LlvmBuildConfig
        where K: AsRef<OsStr>, V: AsRef<OsStr>
    {
        self.defines.push((k.as_ref().to_owned(), v.as_ref().to_owned()));
        self
    }

    fn build_arg<S: AsRef<OsStr>>(&mut self, v: S) -> &mut LlvmBuildConfig {
        self.build_args.push(v.as_ref().to_owned());
        self
    }
}

impl fmt::Display for LlvmBuildConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Write the force clean trigger first.
        write!(f, "{}", self.clean_trigger)?;
        write!(f, "\n")?;

        // Write out every configuration that should trigger rebuilding if
        // changed.
        writeln!(f, "target: {}", self.target)?;
        writeln!(f, "host: {}", self.host)?;
        writeln!(f, "profile: {}", self.profile)?;

        // Be stable and sort the defines before outputting.
        // Also pay attention to the formatting of OsStr; in order to prevent
        // lossy encoding from happening, the `Debug` trait is used.
        let mut defines = self.defines.clone();
        defines.sort();
        for (k, v) in defines {
            writeln!(f, "define: {:?} = {:?}", k, v)?;
        }

        for v in &self.build_args {
            writeln!(f, "build_arg: {:?}", v)?;
        }

        Ok(())
    }
}

/// Configure LLVM for `target`.
fn configure_llvm(build: &Build, target: &str, dst: &Path, clean_trigger: &str) -> LlvmBuildConfig {
    let assertions = if build.config.llvm_assertions {"ON"} else {"OFF"};

    // http://llvm.org/docs/CMake.html
    let mut cfg = LlvmBuildConfig::new(clean_trigger, build.src.join("src/llvm"));
    if build.config.ninja {
        cfg.generator("Ninja");
    }

    let profile = match (build.config.llvm_optimize, build.config.llvm_release_debuginfo) {
        (false, _) => "Debug",
        (true, false) => "Release",
        (true, true) => "RelWithDebInfo",
    };

    // NOTE: remember to also update `config.toml.example` when changing the defaults!
    let llvm_targets = match build.config.llvm_targets {
        Some(ref s) => s,
        None => "X86;ARM;AArch64;Mips;PowerPC;SystemZ;JSBackend;MSP430;Sparc;NVPTX",
    };

    cfg.target(target)
       .host(&build.config.build)
       .out_dir(dst)
       .profile(profile)
       .define("LLVM_ENABLE_ASSERTIONS", assertions)
       .define("LLVM_TARGETS_TO_BUILD", llvm_targets)
       .define("LLVM_INCLUDE_EXAMPLES", "OFF")
       .define("LLVM_INCLUDE_TESTS", "OFF")
       .define("LLVM_INCLUDE_DOCS", "OFF")
       .define("LLVM_ENABLE_ZLIB", "OFF")
       .define("WITH_POLLY", "OFF")
       .define("LLVM_ENABLE_TERMINFO", "OFF")
       .define("LLVM_ENABLE_LIBEDIT", "OFF")
       .define("LLVM_PARALLEL_COMPILE_JOBS", build.jobs().to_string())
       .define("LLVM_TARGET_ARCH", target.split('-').next().unwrap())
       .define("LLVM_DEFAULT_TARGET_TRIPLE", target);

    if target.starts_with("i686") {
        cfg.define("LLVM_BUILD_32_BITS", "ON");
    }

    // http://llvm.org/docs/HowToCrossCompileLLVM.html
    if target != build.config.build {
        // FIXME: if the llvm root for the build triple is overridden then we
        //        should use llvm-tblgen from there, also should verify that it
        //        actually exists most of the time in normal installs of LLVM.
        let host = build.llvm_out(&build.config.build).join("bin/llvm-tblgen");
        cfg.define("CMAKE_CROSSCOMPILING", "True")
           .define("LLVM_TABLEGEN", &host);
    }

    // MSVC handles compiler business itself
    if !target.contains("msvc") {
        if let Some(ref ccache) = build.config.ccache {
           cfg.define("CMAKE_C_COMPILER", ccache)
              .define("CMAKE_C_COMPILER_ARG1", build.cc(target))
              .define("CMAKE_CXX_COMPILER", ccache)
              .define("CMAKE_CXX_COMPILER_ARG1", build.cxx(target));
        } else {
           cfg.define("CMAKE_C_COMPILER", build.cc(target))
              .define("CMAKE_CXX_COMPILER", build.cxx(target));
        }
        cfg.build_arg("-j").build_arg(build.jobs().to_string());

        cfg.define("CMAKE_C_FLAGS", build.cflags(target).join(" "));
        cfg.define("CMAKE_CXX_FLAGS", build.cflags(target).join(" "));
    }

    cfg
}

/// Compile LLVM for `target`.
pub fn llvm(build: &Build, target: &str) {
    // If we're using a custom LLVM bail out here, but we can only use a
    // custom LLVM for the build triple.
    if let Some(config) = build.config.target_config.get(target) {
        if let Some(ref s) = config.llvm_config {
            return check_llvm_version(build, s);
        }
    }

    // Read the clean trigger for generation and comparison of build stamps
    // below.
    let dst = build.llvm_out(target);
    let trigger = build.src.join("src/rustllvm/llvm-auto-clean-trigger");
    let mut trigger_contents = String::new();
    t!(t!(File::open(&trigger)).read_to_string(&mut trigger_contents));

    // Configure LLVM and generate the expected stamp for this build.
    let cfg = configure_llvm(build, target, &dst, &trigger_contents);
    let stamp_contents = format!("{}", cfg);

    // If any of the following is true, then clean and rebuild LLVM, otherwise
    // do nothing:
    //
    // * the cleaning trigger is newer than our built artifacts,
    // * the artifacts are missing, or
    // * the built LLVM is configured differently.
    let done_stamp = dst.join("llvm-finished-building");
    if done_stamp.exists() {
        let mut done_contents = String::new();
        t!(t!(File::open(&done_stamp)).read_to_string(&mut done_contents));
        if done_contents == stamp_contents {
            return
        }
    }
    drop(fs::remove_dir_all(&dst));

    println!("Building LLVM for {}", target);

    let _time = util::timeit();
    let _ = fs::remove_dir_all(&dst.join("build"));
    t!(fs::create_dir_all(&dst.join("build")));

    // Prepare the CMake config for build.
    let mut cmake_cfg = cfg.into_cmake();

    // FIXME: we don't actually need to build all LLVM tools and all LLVM
    //        libraries here, e.g. we just want a few components and a few
    //        tools. Figure out how to filter them down and only build the right
    //        tools and libs on all platforms.
    cmake_cfg.build();

    t!(t!(File::create(&done_stamp)).write_all(stamp_contents.as_bytes()));
}

fn check_llvm_version(build: &Build, llvm_config: &Path) {
    if !build.config.llvm_version_check {
        return
    }

    let mut cmd = Command::new(llvm_config);
    let version = output(cmd.arg("--version"));
    if version.starts_with("3.5") || version.starts_with("3.6") ||
       version.starts_with("3.7") {
        return
    }
    panic!("\n\nbad LLVM version: {}, need >=3.5\n\n", version)
}

/// Compiles the `rust_test_helpers.c` library which we used in various
/// `run-pass` test suites for ABI testing.
pub fn test_helpers(build: &Build, target: &str) {
    let dst = build.test_helpers_out(target);
    let src = build.src.join("src/rt/rust_test_helpers.c");
    if up_to_date(&src, &dst.join("librust_test_helpers.a")) {
        return
    }

    println!("Building test helpers");
    t!(fs::create_dir_all(&dst));
    let mut cfg = gcc::Config::new();

    // We may have found various cross-compilers a little differently due to our
    // extra configuration, so inform gcc of these compilers. Note, though, that
    // on MSVC we still need gcc's detection of env vars (ugh).
    if !target.contains("msvc") {
        if let Some(ar) = build.ar(target) {
            cfg.archiver(ar);
        }
        cfg.compiler(build.cc(target));
    }

    cfg.cargo_metadata(false)
       .out_dir(&dst)
       .target(target)
       .host(&build.config.build)
       .opt_level(0)
       .debug(false)
       .file(build.src.join("src/rt/rust_test_helpers.c"))
       .compile("librust_test_helpers.a");
}
