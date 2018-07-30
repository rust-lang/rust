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
use std::path::{Path, PathBuf};
use std::process::Command;

use build_helper::output;
use cmake;
use cc;

use util::{self, exe};
use build_helper::up_to_date;
use builder::{Builder, RunConfig, ShouldRun, Step};
use cache::Interned;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Llvm {
    pub target: Interned<String>,
    pub emscripten: bool,
}

impl Step for Llvm {
    type Output = PathBuf; // path to llvm-config

    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/llvm").path("src/llvm-emscripten")
    }

    fn make_run(run: RunConfig) {
        let emscripten = run.path.ends_with("llvm-emscripten");
        run.builder.ensure(Llvm {
            target: run.target,
            emscripten,
        });
    }

    /// Compile LLVM for `target`.
    fn run(self, builder: &Builder) -> PathBuf {
        let target = self.target;
        let emscripten = self.emscripten;

        // If we're using a custom LLVM bail out here, but we can only use a
        // custom LLVM for the build triple.
        if !self.emscripten {
            if let Some(config) = builder.config.target_config.get(&target) {
                if let Some(ref s) = config.llvm_config {
                    check_llvm_version(builder, s);
                    return s.to_path_buf()
                }
            }
        }

        let rebuild_trigger = builder.src.join("src/rustllvm/llvm-rebuild-trigger");
        let mut rebuild_trigger_contents = String::new();
        t!(t!(File::open(&rebuild_trigger)).read_to_string(&mut rebuild_trigger_contents));

        let (out_dir, llvm_config_ret_dir) = if emscripten {
            let dir = builder.emscripten_llvm_out(target);
            let config_dir = dir.join("bin");
            (dir, config_dir)
        } else {
            let mut dir = builder.llvm_out(builder.config.build);
            if !builder.config.build.contains("msvc") || builder.config.ninja {
                dir.push("build");
            }
            (builder.llvm_out(target), dir.join("bin"))
        };
        let done_stamp = out_dir.join("llvm-finished-building");
        let build_llvm_config = llvm_config_ret_dir
            .join(exe("llvm-config", &*builder.config.build));
        if done_stamp.exists() {
            let mut done_contents = String::new();
            t!(t!(File::open(&done_stamp)).read_to_string(&mut done_contents));

            // If LLVM was already built previously and contents of the rebuild-trigger file
            // didn't change from the previous build, then no action is required.
            if done_contents == rebuild_trigger_contents {
                return build_llvm_config
            }
        }

        let _folder = builder.fold_output(|| "llvm");
        let descriptor = if emscripten { "Emscripten " } else { "" };
        builder.info(&format!("Building {}LLVM for {}", descriptor, target));
        let _time = util::timeit(&builder);
        t!(fs::create_dir_all(&out_dir));

        // http://llvm.org/docs/CMake.html
        let root = if self.emscripten { "src/llvm-emscripten" } else { "src/llvm" };
        let mut cfg = cmake::Config::new(builder.src.join(root));

        let profile = match (builder.config.llvm_optimize, builder.config.llvm_release_debuginfo) {
            (false, _) => "Debug",
            (true, false) => "Release",
            (true, true) => "RelWithDebInfo",
        };

        // NOTE: remember to also update `config.toml.example` when changing the
        // defaults!
        let llvm_targets = if self.emscripten {
            "JSBackend"
        } else {
            match builder.config.llvm_targets {
                Some(ref s) => s,
                None => "X86;ARM;AArch64;Mips;PowerPC;SystemZ;MSP430;Sparc;NVPTX;Hexagon",
            }
        };

        let llvm_exp_targets = if self.emscripten {
            ""
        } else {
            &builder.config.llvm_experimental_targets[..]
        };

        let assertions = if builder.config.llvm_assertions {"ON"} else {"OFF"};

        cfg.out_dir(&out_dir)
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
           .define("LLVM_ENABLE_LIBXML2", "OFF")
           .define("LLVM_PARALLEL_COMPILE_JOBS", builder.jobs().to_string())
           .define("LLVM_TARGET_ARCH", target.split('-').next().unwrap())
           .define("LLVM_DEFAULT_TARGET_TRIPLE", target);

        // By default, LLVM will automatically find OCaml and, if it finds it,
        // install the LLVM bindings in LLVM_OCAML_INSTALL_PATH, which defaults
        // to /usr/bin/ocaml.
        // This causes problem for non-root builds of Rust. Side-step the issue
        // by setting LLVM_OCAML_INSTALL_PATH to a relative path, so it installs
        // in the prefix.
        cfg.define("LLVM_OCAML_INSTALL_PATH",
            env::var_os("LLVM_OCAML_INSTALL_PATH").unwrap_or_else(|| "usr/lib/ocaml".into()));

        // This setting makes the LLVM tools link to the dynamic LLVM library,
        // which saves both memory during parallel links and overall disk space
        // for the tools.  We don't distribute any of those tools, so this is
        // just a local concern.  However, it doesn't work well everywhere.
        //
        // If we are shipping llvm tools then we statically link them LLVM
        if (target.contains("linux-gnu") || target.contains("apple-darwin")) &&
            !builder.config.llvm_tools_enabled {
                cfg.define("LLVM_LINK_LLVM_DYLIB", "ON");
        }

        // For distribution we want the LLVM tools to be *statically* linked to libstdc++
        if builder.config.llvm_tools_enabled {
            if !target.contains("windows") {
                if target.contains("apple") {
                    cfg.define("CMAKE_EXE_LINKER_FLAGS", "-static-libstdc++");
                } else {
                    cfg.define("CMAKE_EXE_LINKER_FLAGS", "-Wl,-Bsymbolic -static-libstdc++");
                }
            }
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

        if let Some(num_linkers) = builder.config.llvm_link_jobs {
            if num_linkers > 0 {
                cfg.define("LLVM_PARALLEL_LINK_JOBS", num_linkers.to_string());
            }
        }

        // http://llvm.org/docs/HowToCrossCompileLLVM.html
        if target != builder.config.build && !emscripten {
            builder.ensure(Llvm {
                target: builder.config.build,
                emscripten: false,
            });
            // FIXME: if the llvm root for the build triple is overridden then we
            //        should use llvm-tblgen from there, also should verify that it
            //        actually exists most of the time in normal installs of LLVM.
            let host = builder.llvm_out(builder.config.build).join("bin/llvm-tblgen");
            cfg.define("CMAKE_CROSSCOMPILING", "True")
               .define("LLVM_TABLEGEN", &host);

            if target.contains("netbsd") {
               cfg.define("CMAKE_SYSTEM_NAME", "NetBSD");
            } else if target.contains("freebsd") {
               cfg.define("CMAKE_SYSTEM_NAME", "FreeBSD");
            }

            cfg.define("LLVM_NATIVE_BUILD", builder.llvm_out(builder.config.build).join("build"));
        }

        configure_cmake(builder, target, &mut cfg, false);

        // FIXME: we don't actually need to build all LLVM tools and all LLVM
        //        libraries here, e.g. we just want a few components and a few
        //        tools. Figure out how to filter them down and only build the right
        //        tools and libs on all platforms.

        if builder.config.dry_run {
            return build_llvm_config;
        }

        cfg.build();

        t!(t!(File::create(&done_stamp)).write_all(rebuild_trigger_contents.as_bytes()));

        build_llvm_config
    }
}

fn check_llvm_version(builder: &Builder, llvm_config: &Path) {
    if !builder.config.llvm_version_check {
        return
    }

    if builder.config.dry_run {
        return;
    }

    let mut cmd = Command::new(llvm_config);
    let version = output(cmd.arg("--version"));
    let mut parts = version.split('.').take(2)
        .filter_map(|s| s.parse::<u32>().ok());
    if let (Some(major), Some(_minor)) = (parts.next(), parts.next()) {
        if major >= 5 {
            return
        }
    }
    panic!("\n\nbad LLVM version: {}, need >=5.0\n\n", version)
}

fn configure_cmake(builder: &Builder,
                   target: Interned<String>,
                   cfg: &mut cmake::Config,
                   building_dist_binaries: bool) {
    if builder.config.ninja {
        cfg.generator("Ninja");
    }
    cfg.target(&target)
       .host(&builder.config.build);

    let sanitize_cc = |cc: &Path| {
        if target.contains("msvc") {
            OsString::from(cc.to_str().unwrap().replace("\\", "/"))
        } else {
            cc.as_os_str().to_owned()
        }
    };

    // MSVC with CMake uses msbuild by default which doesn't respect these
    // vars that we'd otherwise configure. In that case we just skip this
    // entirely.
    if target.contains("msvc") && !builder.config.ninja {
        return
    }

    let (cc, cxx) = match builder.config.llvm_clang_cl {
        Some(ref cl) => (cl.as_ref(), cl.as_ref()),
        None => (builder.cc(target), builder.cxx(target).unwrap()),
    };

    // Handle msvc + ninja + ccache specially (this is what the bots use)
    if target.contains("msvc") &&
       builder.config.ninja &&
       builder.config.ccache.is_some()
    {
       let mut wrap_cc = env::current_exe().expect("failed to get cwd");
       wrap_cc.set_file_name("sccache-plus-cl.exe");

       cfg.define("CMAKE_C_COMPILER", sanitize_cc(&wrap_cc))
          .define("CMAKE_CXX_COMPILER", sanitize_cc(&wrap_cc));
       cfg.env("SCCACHE_PATH",
               builder.config.ccache.as_ref().unwrap())
          .env("SCCACHE_TARGET", target)
          .env("SCCACHE_CC", &cc)
          .env("SCCACHE_CXX", &cxx);

       // Building LLVM on MSVC can be a little ludicrous at times. We're so far
       // off the beaten path here that I'm not really sure this is even half
       // supported any more. Here we're trying to:
       //
       // * Build LLVM on MSVC
       // * Build LLVM with `clang-cl` instead of `cl.exe`
       // * Build a project with `sccache`
       // * Build for 32-bit as well
       // * Build with Ninja
       //
       // For `cl.exe` there are different binaries to compile 32/64 bit which
       // we use but for `clang-cl` there's only one which internally
       // multiplexes via flags. As a result it appears that CMake's detection
       // of a compiler's architecture and such on MSVC **doesn't** pass any
       // custom flags we pass in CMAKE_CXX_FLAGS below. This means that if we
       // use `clang-cl.exe` it's always diagnosed as a 64-bit compiler which
       // definitely causes problems since all the env vars are pointing to
       // 32-bit libraries.
       //
       // To hack aroudn this... again... we pass an argument that's
       // unconditionally passed in the sccache shim. This'll get CMake to
       // correctly diagnose it's doing a 32-bit compilation and LLVM will
       // internally configure itself appropriately.
       if builder.config.llvm_clang_cl.is_some() && target.contains("i686") {
           cfg.env("SCCACHE_EXTRA_ARGS", "-m32");
       }

    // If ccache is configured we inform the build a little differently hwo
    // to invoke ccache while also invoking our compilers.
    } else if let Some(ref ccache) = builder.config.ccache {
       cfg.define("CMAKE_C_COMPILER", ccache)
          .define("CMAKE_C_COMPILER_ARG1", sanitize_cc(cc))
          .define("CMAKE_CXX_COMPILER", ccache)
          .define("CMAKE_CXX_COMPILER_ARG1", sanitize_cc(cxx));
    } else {
       cfg.define("CMAKE_C_COMPILER", sanitize_cc(cc))
          .define("CMAKE_CXX_COMPILER", sanitize_cc(cxx));
    }

    cfg.build_arg("-j").build_arg(builder.jobs().to_string());
    cfg.define("CMAKE_C_FLAGS", builder.cflags(target).join(" "));
    let mut cxxflags = builder.cflags(target).join(" ");
    if building_dist_binaries {
        if builder.config.llvm_static_stdcpp && !target.contains("windows") {
            cxxflags.push_str(" -static-libstdc++");
        }
    }
    cfg.define("CMAKE_CXX_FLAGS", cxxflags);
    if let Some(ar) = builder.ar(target) {
        if ar.is_absolute() {
            // LLVM build breaks if `CMAKE_AR` is a relative path, for some reason it
            // tries to resolve this path in the LLVM build directory.
            cfg.define("CMAKE_AR", sanitize_cc(ar));
        }
    }

    if env::var_os("SCCACHE_ERROR_LOG").is_some() {
        cfg.env("RUST_LOG", "sccache=warn");
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Lld {
    pub target: Interned<String>,
}

impl Step for Lld {
    type Output = PathBuf;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/lld")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Lld { target: run.target });
    }

    /// Compile LLVM for `target`.
    fn run(self, builder: &Builder) -> PathBuf {
        if builder.config.dry_run {
            return PathBuf::from("lld-out-dir-test-gen");
        }
        let target = self.target;

        let llvm_config = builder.ensure(Llvm {
            target: self.target,
            emscripten: false,
        });

        let out_dir = builder.lld_out(target);
        let done_stamp = out_dir.join("lld-finished-building");
        if done_stamp.exists() {
            return out_dir
        }

        let _folder = builder.fold_output(|| "lld");
        builder.info(&format!("Building LLD for {}", target));
        let _time = util::timeit(&builder);
        t!(fs::create_dir_all(&out_dir));

        let mut cfg = cmake::Config::new(builder.src.join("src/tools/lld"));
        configure_cmake(builder, target, &mut cfg, true);

        // This is an awful, awful hack. Discovered when we migrated to using
        // clang-cl to compile LLVM/LLD it turns out that LLD, when built out of
        // tree, will execute `llvm-config --cmakedir` and then tell CMake about
        // that directory for later processing. Unfortunately if this path has
        // forward slashes in it (which it basically always does on Windows)
        // then CMake will hit a syntax error later on as... something isn't
        // escaped it seems?
        //
        // Instead of attempting to fix this problem in upstream CMake and/or
        // LLVM/LLD we just hack around it here. This thin wrapper will take the
        // output from llvm-config and replace all instances of `\` with `/` to
        // ensure we don't hit the same bugs with escaping. It means that you
        // can't build on a system where your paths require `\` on Windows, but
        // there's probably a lot of reasons you can't do that other than this.
        let llvm_config_shim = env::current_exe()
            .unwrap()
            .with_file_name("llvm-config-wrapper");
        cfg.out_dir(&out_dir)
           .profile("Release")
           .env("LLVM_CONFIG_REAL", llvm_config)
           .define("LLVM_CONFIG_PATH", llvm_config_shim)
           .define("LLVM_INCLUDE_TESTS", "OFF");

        cfg.build();

        t!(File::create(&done_stamp));
        out_dir
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TestHelpers {
    pub target: Interned<String>,
}

impl Step for TestHelpers {
    type Output = ();

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/test/auxiliary/rust_test_helpers.c")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(TestHelpers { target: run.target })
    }

    /// Compiles the `rust_test_helpers.c` library which we used in various
    /// `run-pass` test suites for ABI testing.
    fn run(self, builder: &Builder) {
        if builder.config.dry_run {
            return;
        }
        let target = self.target;
        let dst = builder.test_helpers_out(target);
        let src = builder.src.join("src/test/auxiliary/rust_test_helpers.c");
        if up_to_date(&src, &dst.join("librust_test_helpers.a")) {
            return
        }

        let _folder = builder.fold_output(|| "build_test_helpers");
        builder.info("Building test helpers");
        t!(fs::create_dir_all(&dst));
        let mut cfg = cc::Build::new();

        // We may have found various cross-compilers a little differently due to our
        // extra configuration, so inform gcc of these compilers. Note, though, that
        // on MSVC we still need gcc's detection of env vars (ugh).
        if !target.contains("msvc") {
            if let Some(ar) = builder.ar(target) {
                cfg.archiver(ar);
            }
            cfg.compiler(builder.cc(target));
        }

        cfg.cargo_metadata(false)
           .out_dir(&dst)
           .target(&target)
           .host(&builder.config.build)
           .opt_level(0)
           .warnings(false)
           .debug(false)
           .file(builder.src.join("src/test/auxiliary/rust_test_helpers.c"))
           .compile("rust_test_helpers");
    }
}

const OPENSSL_VERS: &'static str = "1.0.2n";
const OPENSSL_SHA256: &'static str =
    "370babb75f278c39e0c50e8c4e7493bc0f18db6867478341a832a982fd15a8fe";

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
        if builder.config.dry_run {
            return;
        }
        let target = self.target;
        let out = match builder.openssl_dir(target) {
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
                let mut shasum = if target.contains("apple") ||
                    builder.config.build.contains("netbsd") {
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
        let dst = builder.openssl_install_dir(target).unwrap();
        drop(fs::remove_dir_all(&obj));
        drop(fs::remove_dir_all(&dst));
        builder.run(Command::new("tar").arg("zxf").arg(&tarball).current_dir(&out));

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
            "armv6-unknown-netbsd-eabihf" => "BSD-generic32",
            "armv7-linux-androideabi" => "android-armv7",
            "armv7-unknown-linux-gnueabihf" => "linux-armv4",
            "armv7-unknown-netbsd-eabihf" => "BSD-generic32",
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
            "powerpc-unknown-linux-gnuspe" => "linux-ppc",
            "powerpc-unknown-netbsd" => "BSD-generic32",
            "powerpc64-unknown-linux-gnu" => "linux-ppc64",
            "powerpc64le-unknown-linux-gnu" => "linux-ppc64le",
            "powerpc64le-unknown-linux-musl" => "linux-ppc64le",
            "s390x-unknown-linux-gnu" => "linux64-s390x",
            "sparc-unknown-linux-gnu" => "linux-sparcv9",
            "sparc64-unknown-linux-gnu" => "linux64-sparcv9",
            "sparc64-unknown-netbsd" => "BSD-sparc64",
            "x86_64-apple-darwin" => "darwin64-x86_64-cc",
            "x86_64-linux-android" => "linux-x86_64",
            "x86_64-unknown-freebsd" => "BSD-x86_64",
            "x86_64-unknown-dragonfly" => "BSD-x86_64",
            "x86_64-unknown-linux-gnu" => "linux-x86_64",
            "x86_64-unknown-linux-gnux32" => "linux-x32",
            "x86_64-unknown-linux-musl" => "linux-x86_64",
            "x86_64-unknown-netbsd" => "BSD-x86_64",
            _ => panic!("don't know how to configure OpenSSL for {}", target),
        };
        configure.arg(os);
        configure.env("CC", builder.cc(target));
        for flag in builder.cflags(target) {
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
            if builder.config.build.contains("netbsd") {
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
        builder.info(&format!("Configuring openssl for {}", target));
        builder.run_quiet(&mut configure);
        builder.info(&format!("Building openssl for {}", target));
        builder.run_quiet(Command::new("make").arg("-j1").current_dir(&obj));
        builder.info(&format!("Installing openssl for {}", target));
        builder.run_quiet(Command::new("make").arg("install").arg("-j1").current_dir(&obj));

        let mut f = t!(File::create(&stamp));
        t!(f.write_all(OPENSSL_VERS.as_bytes()));
    }
}
