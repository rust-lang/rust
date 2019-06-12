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
use std::path::{Path, PathBuf};
use std::process::Command;

use build_helper::{output, t};
use cmake;
use cc;

use crate::channel;
use crate::util::{self, exe};
use build_helper::up_to_date;
use crate::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::cache::Interned;
use crate::GitRepo;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Llvm {
    pub target: Interned<String>,
    pub emscripten: bool,
}

impl Step for Llvm {
    type Output = PathBuf; // path to llvm-config

    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project")
            .path("src/llvm-project/llvm")
            .path("src/llvm")
            .path("src/llvm-emscripten")
    }

    fn make_run(run: RunConfig<'_>) {
        let emscripten = run.path.ends_with("llvm-emscripten");
        run.builder.ensure(Llvm {
            target: run.target,
            emscripten,
        });
    }

    /// Compile LLVM for `target`.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
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

        let (llvm_info, root, out_dir, llvm_config_ret_dir) = if emscripten {
            let info = &builder.emscripten_llvm_info;
            let dir = builder.emscripten_llvm_out(target);
            let config_dir = dir.join("bin");
            (info, "src/llvm-emscripten", dir, config_dir)
        } else {
            let info = &builder.in_tree_llvm_info;
            let mut dir = builder.llvm_out(builder.config.build);
            if !builder.config.build.contains("msvc") || builder.config.ninja {
                dir.push("build");
            }
            (info, "src/llvm-project/llvm", builder.llvm_out(target), dir.join("bin"))
        };

        if !llvm_info.is_git() {
            println!(
                "git could not determine the LLVM submodule commit hash. \
                Assuming that an LLVM build is necessary.",
            );
        }

        let build_llvm_config = llvm_config_ret_dir
            .join(exe("llvm-config", &*builder.config.build));
        let done_stamp = out_dir.join("llvm-finished-building");

        if let Some(llvm_commit) = llvm_info.sha() {
            if done_stamp.exists() {
                let done_contents = t!(fs::read(&done_stamp));

                // If LLVM was already built previously and the submodule's commit didn't change
                // from the previous build, then no action is required.
                if done_contents == llvm_commit.as_bytes() {
                    return build_llvm_config
                }
            }
        }

        let _folder = builder.fold_output(|| "llvm");
        let descriptor = if emscripten { "Emscripten " } else { "" };
        builder.info(&format!("Building {}LLVM for {}", descriptor, target));
        let _time = util::timeit(&builder);
        t!(fs::create_dir_all(&out_dir));

        // http://llvm.org/docs/CMake.html
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
           .define("LLVM_INCLUDE_BENCHMARKS", "OFF")
           .define("LLVM_ENABLE_ZLIB", "OFF")
           .define("WITH_POLLY", "OFF")
           .define("LLVM_ENABLE_TERMINFO", "OFF")
           .define("LLVM_ENABLE_LIBEDIT", "OFF")
           .define("LLVM_PARALLEL_COMPILE_JOBS", builder.jobs().to_string())
           .define("LLVM_TARGET_ARCH", target.split('-').next().unwrap())
           .define("LLVM_DEFAULT_TARGET_TRIPLE", target);

        if builder.config.llvm_thin_lto && !emscripten {
            cfg.define("LLVM_ENABLE_LTO", "Thin");
            if !target.contains("apple") {
               cfg.define("LLVM_ENABLE_LLD", "ON");
            }
        }

        // By default, LLVM will automatically find OCaml and, if it finds it,
        // install the LLVM bindings in LLVM_OCAML_INSTALL_PATH, which defaults
        // to /usr/bin/ocaml.
        // This causes problem for non-root builds of Rust. Side-step the issue
        // by setting LLVM_OCAML_INSTALL_PATH to a relative path, so it installs
        // in the prefix.
        cfg.define("LLVM_OCAML_INSTALL_PATH",
            env::var_os("LLVM_OCAML_INSTALL_PATH").unwrap_or_else(|| "usr/lib/ocaml".into()));

        let want_lldb = builder.config.lldb_enabled && !self.emscripten;

        // This setting makes the LLVM tools link to the dynamic LLVM library,
        // which saves both memory during parallel links and overall disk space
        // for the tools. We don't do this on every platform as it doesn't work
        // equally well everywhere.
        if builder.llvm_link_tools_dynamically(target) && !emscripten {
            cfg.define("LLVM_LINK_LLVM_DYLIB", "ON");
        }

        // For distribution we want the LLVM tools to be *statically* linked to libstdc++
        if builder.config.llvm_tools_enabled || want_lldb {
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

        let mut enabled_llvm_projects = Vec::new();

        if util::forcing_clang_based_tests() {
            enabled_llvm_projects.push("clang");
            enabled_llvm_projects.push("compiler-rt");
        }

        if want_lldb {
            enabled_llvm_projects.push("clang");
            enabled_llvm_projects.push("lldb");
            // For the time being, disable code signing.
            cfg.define("LLDB_CODESIGN_IDENTITY", "");
            cfg.define("LLDB_NO_DEBUGSERVER", "ON");
        } else {
            // LLDB requires libxml2; but otherwise we want it to be disabled.
            // See https://github.com/rust-lang/rust/pull/50104
            cfg.define("LLVM_ENABLE_LIBXML2", "OFF");
        }

        if enabled_llvm_projects.len() > 0 {
            enabled_llvm_projects.sort();
            enabled_llvm_projects.dedup();
            cfg.define("LLVM_ENABLE_PROJECTS", enabled_llvm_projects.join(";"));
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

        if let Some(ref suffix) = builder.config.llvm_version_suffix {
            // Allow version-suffix="" to not define a version suffix at all.
            if !suffix.is_empty() {
                cfg.define("LLVM_VERSION_SUFFIX", suffix);
            }
        } else {
            let mut default_suffix = format!(
                "-rust-{}-{}",
                channel::CFG_RELEASE_NUM,
                builder.config.channel,
            );
            if let Some(sha) = llvm_info.sha_short() {
                default_suffix.push_str("-");
                default_suffix.push_str(sha);
            }
            cfg.define("LLVM_VERSION_SUFFIX", default_suffix);
        }

        if let Some(ref linker) = builder.config.llvm_use_linker {
            cfg.define("LLVM_USE_LINKER", linker);
        }

        if let Some(true) = builder.config.llvm_allow_old_toolchain {
            cfg.define("LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN", "YES");
        }

        if let Some(ref python) = builder.config.python {
            cfg.define("PYTHON_EXECUTABLE", python);
        }

        configure_cmake(builder, target, &mut cfg);

        // FIXME: we don't actually need to build all LLVM tools and all LLVM
        //        libraries here, e.g., we just want a few components and a few
        //        tools. Figure out how to filter them down and only build the right
        //        tools and libs on all platforms.

        if builder.config.dry_run {
            return build_llvm_config;
        }

        cfg.build();

        if let Some(llvm_commit) = llvm_info.sha() {
            t!(fs::write(&done_stamp, llvm_commit));
        }

        build_llvm_config
    }
}

fn check_llvm_version(builder: &Builder<'_>, llvm_config: &Path) {
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
        if major >= 6 {
            return
        }
    }
    panic!("\n\nbad LLVM version: {}, need >=6.0\n\n", version)
}

fn configure_cmake(builder: &Builder<'_>,
                   target: Interned<String>,
                   cfg: &mut cmake::Config) {
    // Do not print installation messages for up-to-date files.
    // LLVM and LLD builds can produce a lot of those and hit CI limits on log size.
    cfg.define("CMAKE_INSTALL_MESSAGE", "LAZY");

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
       // To hack around this... again... we pass an argument that's
       // unconditionally passed in the sccache shim. This'll get CMake to
       // correctly diagnose it's doing a 32-bit compilation and LLVM will
       // internally configure itself appropriately.
       if builder.config.llvm_clang_cl.is_some() && target.contains("i686") {
           cfg.env("SCCACHE_EXTRA_ARGS", "-m32");
       }
    } else {
       // If ccache is configured we inform the build a little differently how
       // to invoke ccache while also invoking our compilers.
       if let Some(ref ccache) = builder.config.ccache {
         cfg.define("CMAKE_C_COMPILER_LAUNCHER", ccache)
            .define("CMAKE_CXX_COMPILER_LAUNCHER", ccache);
       }
       cfg.define("CMAKE_C_COMPILER", sanitize_cc(cc))
          .define("CMAKE_CXX_COMPILER", sanitize_cc(cxx));
    }

    cfg.build_arg("-j").build_arg(builder.jobs().to_string());
    let mut cflags = builder.cflags(target, GitRepo::Llvm).join(" ");
    if let Some(ref s) = builder.config.llvm_cflags {
        cflags.push_str(&format!(" {}", s));
    }
    cfg.define("CMAKE_C_FLAGS", cflags);
    let mut cxxflags = builder.cflags(target, GitRepo::Llvm).join(" ");
    if builder.config.llvm_static_stdcpp &&
        !target.contains("windows") &&
        !target.contains("netbsd")
    {
        cxxflags.push_str(" -static-libstdc++");
    }
    if let Some(ref s) = builder.config.llvm_cxxflags {
        cxxflags.push_str(&format!(" {}", s));
    }
    cfg.define("CMAKE_CXX_FLAGS", cxxflags);
    if let Some(ar) = builder.ar(target) {
        if ar.is_absolute() {
            // LLVM build breaks if `CMAKE_AR` is a relative path, for some reason it
            // tries to resolve this path in the LLVM build directory.
            cfg.define("CMAKE_AR", sanitize_cc(ar));
        }
    }

    if let Some(ranlib) = builder.ranlib(target) {
        if ranlib.is_absolute() {
            // LLVM build breaks if `CMAKE_RANLIB` is a relative path, for some reason it
            // tries to resolve this path in the LLVM build directory.
            cfg.define("CMAKE_RANLIB", sanitize_cc(ranlib));
        }
    }

    if let Some(ref s) = builder.config.llvm_ldflags {
        cfg.define("CMAKE_SHARED_LINKER_FLAGS", s);
        cfg.define("CMAKE_MODULE_LINKER_FLAGS", s);
        cfg.define("CMAKE_EXE_LINKER_FLAGS", s);
    }

    if env::var_os("SCCACHE_ERROR_LOG").is_some() {
        cfg.env("RUSTC_LOG", "sccache=warn");
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Lld {
    pub target: Interned<String>,
}

impl Step for Lld {
    type Output = PathBuf;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project/lld").path("src/tools/lld")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Lld { target: run.target });
    }

    /// Compile LLVM for `target`.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
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

        let mut cfg = cmake::Config::new(builder.src.join("src/llvm-project/lld"));
        configure_cmake(builder, target, &mut cfg);

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

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/test/auxiliary/rust_test_helpers.c")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(TestHelpers { target: run.target })
    }

    /// Compiles the `rust_test_helpers.c` library which we used in various
    /// `run-pass` test suites for ABI testing.
    fn run(self, builder: &Builder<'_>) {
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
