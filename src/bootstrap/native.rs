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
//! compilation in build scripts that Cargo has. This is because the
//! compilation takes a *very* long time but also because we don't want to
//! compile LLVM 3 times as part of a normal bootstrap (we want it cached).
//!
//! LLVM and compiler-rt are essentially just wired up to everything else to
//! ensure that they're always in place if needed.

use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;

use build_helper::output;
use cmake;
use cc;

use Build;
use util;
use build_helper::up_to_date;
use builder::{Builder, RunConfig, ShouldRun, Step};
use cache::Interned;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Llvm {
    pub target: Interned<String>,
}

impl Step for Llvm {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/llvm")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Llvm { target: run.target })
    }

    /// Compile LLVM for `target`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;

        // If we're not compiling for LLVM bail out here.
        if !build.config.llvm_enabled {
            return;
        }

        // If we're using a custom LLVM bail out here, but we can only use a
        // custom LLVM for the build triple.
        if let Some(config) = build.config.target_config.get(&target) {
            if let Some(ref s) = config.llvm_config {
                return check_llvm_version(build, s);
            }
        }

        let rebuild_trigger = build.src.join("src/rustllvm/llvm-rebuild-trigger");
        let mut rebuild_trigger_contents = String::new();
        t!(t!(File::open(&rebuild_trigger)).read_to_string(&mut rebuild_trigger_contents));

        let out_dir = build.llvm_out(target);
        let done_stamp = out_dir.join("llvm-finished-building");
        if done_stamp.exists() {
            let mut done_contents = String::new();
            t!(t!(File::open(&done_stamp)).read_to_string(&mut done_contents));

            // If LLVM was already built previously and contents of the rebuild-trigger file
            // didn't change from the previous build, then no action is required.
            if done_contents == rebuild_trigger_contents {
                return
            }
        }

        let _folder = build.fold_output(|| "llvm");
        println!("Building LLVM for {}", target);
        let _time = util::timeit();
        t!(fs::create_dir_all(&out_dir));

        // http://llvm.org/docs/CMake.html
        let mut cfg = cmake::Config::new(build.src.join("src/llvm"));
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
            None => "X86;ARM;AArch64;Mips;PowerPC;SystemZ;JSBackend;MSP430;Sparc;NVPTX;Hexagon",
        };

        let llvm_exp_targets = &build.config.llvm_experimental_targets;

        let assertions = if build.config.llvm_assertions {"ON"} else {"OFF"};

        cfg.target(&target)
           .host(&build.build)
           .out_dir(&out_dir)
           .profile(profile)
           .define("LLVM_ENABLE_ASSERTIONS", assertions)
           .define("LLVM_TARGETS_TO_BUILD", llvm_targets)
           .define("LLVM_EXPERIMENTAL_TARGETS_TO_BUILD", llvm_exp_targets)
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


        // This setting makes the LLVM tools link to the dynamic LLVM library,
        // which saves both memory during parallel links and overall disk space
        // for the tools.  We don't distribute any of those tools, so this is
        // just a local concern.  However, it doesn't work well everywhere.
        if target.contains("linux-gnu") || target.contains("apple-darwin") {
           cfg.define("LLVM_LINK_LLVM_DYLIB", "ON");
        }

        if target.contains("msvc") {
            cfg.define("LLVM_USE_CRT_DEBUG", "MT");
            cfg.define("LLVM_USE_CRT_RELEASE", "MT");
            cfg.define("LLVM_USE_CRT_RELWITHDEBINFO", "MT");
            cfg.static_crt(true);
        }

        if target.starts_with("i686") {
            cfg.define("LLVM_BUILD_32_BITS", "ON");
        }

        if let Some(num_linkers) = build.config.llvm_link_jobs {
            if num_linkers > 0 {
                cfg.define("LLVM_PARALLEL_LINK_JOBS", num_linkers.to_string());
            }
        }

        // http://llvm.org/docs/HowToCrossCompileLLVM.html
        if target != build.build {
            builder.ensure(Llvm { target: build.build });
            // FIXME: if the llvm root for the build triple is overridden then we
            //        should use llvm-tblgen from there, also should verify that it
            //        actually exists most of the time in normal installs of LLVM.
            let host = build.llvm_out(build.build).join("bin/llvm-tblgen");
            cfg.define("CMAKE_CROSSCOMPILING", "True")
               .define("LLVM_TABLEGEN", &host);

            if target.contains("netbsd") {
               cfg.define("CMAKE_SYSTEM_NAME", "NetBSD");
            } else if target.contains("freebsd") {
               cfg.define("CMAKE_SYSTEM_NAME", "FreeBSD");
            }

            cfg.define("LLVM_NATIVE_BUILD", build.llvm_out(build.build).join("build"));
        }

        let sanitize_cc = |cc: &Path| {
            if target.contains("msvc") {
                OsString::from(cc.to_str().unwrap().replace("\\", "/"))
            } else {
                cc.as_os_str().to_owned()
            }
        };

        let configure_compilers = |cfg: &mut cmake::Config| {
            // MSVC with CMake uses msbuild by default which doesn't respect these
            // vars that we'd otherwise configure. In that case we just skip this
            // entirely.
            if target.contains("msvc") && !build.config.ninja {
                return
            }

            let cc = build.cc(target);
            let cxx = build.cxx(target).unwrap();

            // Handle msvc + ninja + ccache specially (this is what the bots use)
            if target.contains("msvc") &&
               build.config.ninja &&
               build.config.ccache.is_some() {
                let mut cc = env::current_exe().expect("failed to get cwd");
                cc.set_file_name("sccache-plus-cl.exe");

               cfg.define("CMAKE_C_COMPILER", sanitize_cc(&cc))
                  .define("CMAKE_CXX_COMPILER", sanitize_cc(&cc));
               cfg.env("SCCACHE_PATH",
                       build.config.ccache.as_ref().unwrap())
                  .env("SCCACHE_TARGET", target);

            // If ccache is configured we inform the build a little differently hwo
            // to invoke ccache while also invoking our compilers.
            } else if let Some(ref ccache) = build.config.ccache {
               cfg.define("CMAKE_C_COMPILER", ccache)
                  .define("CMAKE_C_COMPILER_ARG1", sanitize_cc(cc))
                  .define("CMAKE_CXX_COMPILER", ccache)
                  .define("CMAKE_CXX_COMPILER_ARG1", sanitize_cc(cxx));
            } else {
               cfg.define("CMAKE_C_COMPILER", sanitize_cc(cc))
                  .define("CMAKE_CXX_COMPILER", sanitize_cc(cxx));
            }

            cfg.build_arg("-j").build_arg(build.jobs().to_string());
            cfg.define("CMAKE_C_FLAGS", build.cflags(target).join(" "));
            cfg.define("CMAKE_CXX_FLAGS", build.cflags(target).join(" "));
            if let Some(ar) = build.ar(target) {
                if ar.is_absolute() {
                    // LLVM build breaks if `CMAKE_AR` is a relative path, for some reason it
                    // tries to resolve this path in the LLVM build directory.
                    cfg.define("CMAKE_AR", sanitize_cc(ar));
                }
            }
        };

        configure_compilers(&mut cfg);

        if env::var_os("SCCACHE_ERROR_LOG").is_some() {
            cfg.env("RUST_LOG", "sccache=warn");
        }

        // FIXME: we don't actually need to build all LLVM tools and all LLVM
        //        libraries here, e.g. we just want a few components and a few
        //        tools. Figure out how to filter them down and only build the right
        //        tools and libs on all platforms.
        cfg.build();

        t!(t!(File::create(&done_stamp)).write_all(rebuild_trigger_contents.as_bytes()));
    }
}

fn check_llvm_version(build: &Build, llvm_config: &Path) {
    if !build.config.llvm_version_check {
        return
    }

    let mut cmd = Command::new(llvm_config);
    let version = output(cmd.arg("--version"));
    let mut parts = version.split('.').take(2)
        .filter_map(|s| s.parse::<u32>().ok());
    if let (Some(major), Some(minor)) = (parts.next(), parts.next()) {
        if major > 3 || (major == 3 && minor >= 9) {
            return
        }
    }
    panic!("\n\nbad LLVM version: {}, need >=3.9\n\n", version)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TestHelpers {
    pub target: Interned<String>,
}

impl Step for TestHelpers {
    type Output = ();

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/rt/rust_test_helpers.c")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(TestHelpers { target: run.target })
    }

    /// Compiles the `rust_test_helpers.c` library which we used in various
    /// `run-pass` test suites for ABI testing.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let dst = build.test_helpers_out(target);
        let src = build.src.join("src/rt/rust_test_helpers.c");
        if up_to_date(&src, &dst.join("librust_test_helpers.a")) {
            return
        }

        let _folder = build.fold_output(|| "build_test_helpers");
        println!("Building test helpers");
        t!(fs::create_dir_all(&dst));
        let mut cfg = cc::Build::new();

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
           .target(&target)
           .host(&build.build)
           .opt_level(0)
           .warnings(false)
           .debug(false)
           .file(build.src.join("src/rt/rust_test_helpers.c"))
           .compile("rust_test_helpers");
    }
}

const OPENSSL_VERS: &'static str = "1.0.2m";
const OPENSSL_SHA256: &'static str =
    "8c6ff15ec6b319b50788f42c7abc2890c08ba5a1cdcd3810eb9092deada37b0f";

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Openssl {
    pub target: Interned<String>,
}

impl Step for Openssl {
    type Output = ();

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let out = match build.openssl_dir(target) {
            Some(dir) => dir,
            None => return,
        };

        let stamp = out.join(".stamp");
        let mut contents = String::new();
        drop(File::open(&stamp).and_then(|mut f| f.read_to_string(&mut contents)));
        if contents == OPENSSL_VERS {
            return
        }
        t!(fs::create_dir_all(&out));

        let name = format!("openssl-{}.tar.gz", OPENSSL_VERS);
        let tarball = out.join(&name);
        if !tarball.exists() {
            let tmp = tarball.with_extension("tmp");
            // originally from https://www.openssl.org/source/...
            let url = format!("https://s3-us-west-1.amazonaws.com/rust-lang-ci2/rust-ci-mirror/{}",
                              name);
            let mut last_error = None;
            for _ in 0..3 {
                let status = Command::new("curl")
                                .arg("-o").arg(&tmp)
                                .arg("-f")  // make curl fail if the URL does not return HTTP 200
                                .arg(&url)
                                .status()
                                .expect("failed to spawn curl");

                // Retry if download failed.
                if !status.success() {
                    last_error = Some(status.to_string());
                    continue;
                }

                // Ensure the hash is correct.
                let mut shasum = if target.contains("apple") || build.build.contains("netbsd") {
                    let mut cmd = Command::new("shasum");
                    cmd.arg("-a").arg("256");
                    cmd
                } else {
                    Command::new("sha256sum")
                };
                let output = output(&mut shasum.arg(&tmp));
                let found = output.split_whitespace().next().unwrap();

                // If the hash is wrong, probably the download is incomplete or S3 served an error
                // page. In any case, retry.
                if found != OPENSSL_SHA256 {
                    last_error = Some(format!(
                        "downloaded openssl sha256 different\n\
                         expected: {}\n\
                         found:    {}\n",
                        OPENSSL_SHA256,
                        found
                    ));
                    continue;
                }

                // Everything is fine, so exit the retry loop.
                last_error = None;
                break;
            }
            if let Some(error) = last_error {
                panic!("failed to download openssl source: {}", error);
            }
            t!(fs::rename(&tmp, &tarball));
        }
        let obj = out.join(format!("openssl-{}", OPENSSL_VERS));
        let dst = build.openssl_install_dir(target).unwrap();
        drop(fs::remove_dir_all(&obj));
        drop(fs::remove_dir_all(&dst));
        build.run(Command::new("tar").arg("zxf").arg(&tarball).current_dir(&out));

        let mut configure = Command::new("perl");
        configure.arg(obj.join("Configure"));
        configure.arg(format!("--prefix={}", dst.display()));
        configure.arg("no-dso");
        configure.arg("no-ssl2");
        configure.arg("no-ssl3");

        let os = match &*target {
            "aarch64-linux-android" => "linux-aarch64",
            "aarch64-unknown-linux-gnu" => "linux-aarch64",
            "aarch64-unknown-linux-musl" => "linux-aarch64",
            "arm-linux-androideabi" => "android",
            "arm-unknown-linux-gnueabi" => "linux-armv4",
            "arm-unknown-linux-gnueabihf" => "linux-armv4",
            "armv7-linux-androideabi" => "android-armv7",
            "armv7-unknown-linux-gnueabihf" => "linux-armv4",
            "i586-unknown-linux-gnu" => "linux-elf",
            "i586-unknown-linux-musl" => "linux-elf",
            "i686-apple-darwin" => "darwin-i386-cc",
            "i686-linux-android" => "android-x86",
            "i686-unknown-freebsd" => "BSD-x86-elf",
            "i686-unknown-linux-gnu" => "linux-elf",
            "i686-unknown-linux-musl" => "linux-elf",
            "i686-unknown-netbsd" => "BSD-x86-elf",
            "mips-unknown-linux-gnu" => "linux-mips32",
            "mips64-unknown-linux-gnuabi64" => "linux64-mips64",
            "mips64el-unknown-linux-gnuabi64" => "linux64-mips64",
            "mipsel-unknown-linux-gnu" => "linux-mips32",
            "powerpc-unknown-linux-gnu" => "linux-ppc",
            "powerpc64-unknown-linux-gnu" => "linux-ppc64",
            "powerpc64le-unknown-linux-gnu" => "linux-ppc64le",
            "s390x-unknown-linux-gnu" => "linux64-s390x",
            "sparc64-unknown-linux-gnu" => "linux64-sparcv9",
            "sparc64-unknown-netbsd" => "BSD-sparc64",
            "x86_64-apple-darwin" => "darwin64-x86_64-cc",
            "x86_64-linux-android" => "linux-x86_64",
            "x86_64-unknown-freebsd" => "BSD-x86_64",
            "x86_64-unknown-dragonfly" => "BSD-x86_64",
            "x86_64-unknown-linux-gnu" => "linux-x86_64",
            "x86_64-unknown-linux-musl" => "linux-x86_64",
            "x86_64-unknown-netbsd" => "BSD-x86_64",
            _ => panic!("don't know how to configure OpenSSL for {}", target),
        };
        configure.arg(os);
        configure.env("CC", build.cc(target));
        for flag in build.cflags(target) {
            configure.arg(flag);
        }
        // There is no specific os target for android aarch64 or x86_64,
        // so we need to pass some extra cflags
        if target == "aarch64-linux-android" || target == "x86_64-linux-android" {
            configure.arg("-mandroid");
            configure.arg("-fomit-frame-pointer");
        }
        if target == "sparc64-unknown-netbsd" {
            // Need -m64 to get assembly generated correctly for sparc64.
            configure.arg("-m64");
            if build.build.contains("netbsd") {
                // Disable sparc64 asm on NetBSD builders, it uses
                // m4(1)'s -B flag, which NetBSD m4 does not support.
                configure.arg("no-asm");
            }
        }
        // Make PIE binaries
        // Non-PIE linker support was removed in Lollipop
        // https://source.android.com/security/enhancements/enhancements50
        if target == "i686-linux-android" {
            configure.arg("no-asm");
        }
        configure.current_dir(&obj);
        println!("Configuring openssl for {}", target);
        build.run_quiet(&mut configure);
        println!("Building openssl for {}", target);
        build.run_quiet(Command::new("make").arg("-j1").current_dir(&obj));
        println!("Installing openssl for {}", target);
        build.run_quiet(Command::new("make").arg("install").current_dir(&obj));

        let mut f = t!(File::create(&stamp));
        t!(f.write_all(OPENSSL_VERS.as_bytes()));
    }
}
