use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::{env, fs};

use build_helper::fs::{ignore_not_found, recursive_remove};

use super::{ProcRes, TestCx, disable_error_reporting};
use crate::util::{copy_dir_all, dylib_env_var};

impl TestCx<'_> {
    pub(super) fn run_rmake_test(&self) {
        let test_dir = &self.testpaths.file;
        if test_dir.join("rmake.rs").exists() {
            self.run_rmake_v2_test();
        } else if test_dir.join("Makefile").exists() {
            self.run_rmake_legacy_test();
        } else {
            self.fatal("failed to find either `rmake.rs` or `Makefile`")
        }
    }

    fn run_rmake_legacy_test(&self) {
        let cwd = env::current_dir().unwrap();
        let src_root = self.config.src_base.parent().unwrap().parent().unwrap();
        let src_root = cwd.join(&src_root);

        // FIXME(Zalathar): This should probably be `output_base_dir` to avoid
        // an unnecessary extra subdirectory, but since legacy Makefile tests
        // are hopefully going away, it seems safer to leave this perilous code
        // as-is until it can all be deleted.
        let tmpdir = cwd.join(self.output_base_name());
        ignore_not_found(|| recursive_remove(&tmpdir)).unwrap();

        fs::create_dir_all(&tmpdir).unwrap();

        let host = &self.config.host;
        let make = if host.contains("dragonfly")
            || host.contains("freebsd")
            || host.contains("netbsd")
            || host.contains("openbsd")
            || host.contains("aix")
        {
            "gmake"
        } else {
            "make"
        };

        let mut cmd = Command::new(make);
        cmd.current_dir(&self.testpaths.file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("TARGET", &self.config.target)
            .env("PYTHON", &self.config.python)
            .env("S", src_root)
            .env("RUST_BUILD_STAGE", &self.config.stage_id)
            .env("RUSTC", cwd.join(&self.config.rustc_path))
            .env("TMPDIR", &tmpdir)
            .env("LD_LIB_PATH_ENVVAR", dylib_env_var())
            .env("HOST_RPATH_DIR", cwd.join(&self.config.compile_lib_path))
            .env("TARGET_RPATH_DIR", cwd.join(&self.config.run_lib_path))
            .env("LLVM_COMPONENTS", &self.config.llvm_components)
            // We for sure don't want these tests to run in parallel, so make
            // sure they don't have access to these vars if we run via `make`
            // at the top level
            .env_remove("MAKEFLAGS")
            .env_remove("MFLAGS")
            .env_remove("CARGO_MAKEFLAGS");

        if let Some(ref cargo) = self.config.cargo_path {
            cmd.env("CARGO", cwd.join(cargo));
        }

        if let Some(ref rustdoc) = self.config.rustdoc_path {
            cmd.env("RUSTDOC", cwd.join(rustdoc));
        }

        if let Some(ref node) = self.config.nodejs {
            cmd.env("NODE", node);
        }

        if let Some(ref linker) = self.config.target_linker {
            cmd.env("RUSTC_LINKER", linker);
        }

        if let Some(ref clang) = self.config.run_clang_based_tests_with {
            cmd.env("CLANG", clang);
        }

        if let Some(ref filecheck) = self.config.llvm_filecheck {
            cmd.env("LLVM_FILECHECK", filecheck);
        }

        if let Some(ref llvm_bin_dir) = self.config.llvm_bin_dir {
            cmd.env("LLVM_BIN_DIR", llvm_bin_dir);
        }

        if let Some(ref remote_test_client) = self.config.remote_test_client {
            cmd.env("REMOTE_TEST_CLIENT", remote_test_client);
        }

        // We don't want RUSTFLAGS set from the outside to interfere with
        // compiler flags set in the test cases:
        cmd.env_remove("RUSTFLAGS");

        // Use dynamic musl for tests because static doesn't allow creating dylibs
        if self.config.host.contains("musl") {
            cmd.env("RUSTFLAGS", "-Ctarget-feature=-crt-static").env("IS_MUSL_HOST", "1");
        }

        if self.config.bless {
            cmd.env("RUSTC_BLESS_TEST", "--bless");
            // Assume this option is active if the environment variable is "defined", with _any_ value.
            // As an example, a `Makefile` can use this option by:
            //
            //   ifdef RUSTC_BLESS_TEST
            //       cp "$(TMPDIR)"/actual_something.ext expected_something.ext
            //   else
            //       $(DIFF) expected_something.ext "$(TMPDIR)"/actual_something.ext
            //   endif
        }

        if self.config.target.contains("msvc") && !self.config.cc.is_empty() {
            // We need to pass a path to `lib.exe`, so assume that `cc` is `cl.exe`
            // and that `lib.exe` lives next to it.
            let lib = Path::new(&self.config.cc).parent().unwrap().join("lib.exe");

            // MSYS doesn't like passing flags of the form `/foo` as it thinks it's
            // a path and instead passes `C:\msys64\foo`, so convert all
            // `/`-arguments to MSVC here to `-` arguments.
            let cflags = self
                .config
                .cflags
                .split(' ')
                .map(|s| s.replace("/", "-"))
                .collect::<Vec<_>>()
                .join(" ");
            let cxxflags = self
                .config
                .cxxflags
                .split(' ')
                .map(|s| s.replace("/", "-"))
                .collect::<Vec<_>>()
                .join(" ");

            cmd.env("IS_MSVC", "1")
                .env("IS_WINDOWS", "1")
                .env("MSVC_LIB", format!("'{}' -nologo", lib.display()))
                .env("MSVC_LIB_PATH", format!("{}", lib.display()))
                .env("CC", format!("'{}' {}", self.config.cc, cflags))
                .env("CXX", format!("'{}' {}", &self.config.cxx, cxxflags));
        } else {
            cmd.env("CC", format!("{} {}", self.config.cc, self.config.cflags))
                .env("CXX", format!("{} {}", self.config.cxx, self.config.cxxflags))
                .env("AR", &self.config.ar);

            if self.config.target.contains("windows") {
                cmd.env("IS_WINDOWS", "1");
            }
        }

        let (output, truncated) =
            self.read2_abbreviated(cmd.spawn().expect("failed to spawn `make`"));
        if !output.status.success() {
            let res = ProcRes {
                status: output.status,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                truncated,
                cmdline: format!("{:?}", cmd),
            };
            self.fatal_proc_rec("make failed", &res);
        }
    }

    fn run_rmake_v2_test(&self) {
        // For `run-make` V2, we need to perform 2 steps to build and run a `run-make` V2 recipe
        // (`rmake.rs`) to run the actual tests. The support library is already built as a tool rust
        // library and is available under `build/$TARGET/stageN-tools-bin/librun_make_support.rlib`.
        //
        // 1. We need to build the recipe `rmake.rs` as a binary and link in the `run_make_support`
        //    library.
        // 2. We need to run the recipe binary.

        // So we assume the rust-lang/rust project setup looks like the following (our `.` is the
        // top-level directory, irrelevant entries to our purposes omitted):
        //
        // ```
        // .                               // <- `source_root`
        // ├── build/                      // <- `build_root`
        // ├── compiler/
        // ├── library/
        // ├── src/
        // │  └── tools/
        // │     └── run_make_support/
        // └── tests
        //    └── run-make/
        // ```

        // `source_root` is the top-level directory containing the rust-lang/rust checkout.
        let source_root =
            self.config.find_rust_src_root().expect("could not determine rust source root");
        // `self.config.build_base` is actually the build base folder + "test" + test suite name, it
        // looks like `build/<host_triple>/test/run-make`. But we want `build/<host_triple>/`. Note
        // that the `build` directory does not need to be called `build`, nor does it need to be
        // under `source_root`, so we must compute it based off of `self.config.build_base`.
        let build_root =
            self.config.build_base.parent().and_then(Path::parent).unwrap().to_path_buf();

        // We construct the following directory tree for each rmake.rs test:
        // ```
        // <base_dir>/
        //     rmake.exe
        //     rmake_out/
        // ```
        // having the recipe executable separate from the output artifacts directory allows the
        // recipes to `remove_dir_all($TMPDIR)` without running into issues related trying to remove
        // a currently running executable because the recipe executable is not under the
        // `rmake_out/` directory.
        //
        // This setup intentionally diverges from legacy Makefile run-make tests.
        let base_dir = self.output_base_dir();
        ignore_not_found(|| recursive_remove(&base_dir)).unwrap();

        let rmake_out_dir = base_dir.join("rmake_out");
        fs::create_dir_all(&rmake_out_dir).unwrap();

        // Copy all input files (apart from rmake.rs) to the temporary directory,
        // so that the input directory structure from `tests/run-make/<test>` is mirrored
        // to the `rmake_out` directory.
        for path in walkdir::WalkDir::new(&self.testpaths.file).min_depth(1) {
            let path = path.unwrap().path().to_path_buf();
            if path.file_name().is_some_and(|s| s != "rmake.rs") {
                let target = rmake_out_dir.join(path.strip_prefix(&self.testpaths.file).unwrap());
                if path.is_dir() {
                    copy_dir_all(&path, target).unwrap();
                } else {
                    fs::copy(&path, target).unwrap();
                }
            }
        }

        // In order to link in the support library as a rlib when compiling recipes, we need three
        // paths:
        // 1. Path of the built support library rlib itself.
        // 2. Path of the built support library's dependencies directory.
        // 3. Path of the built support library's dependencies' dependencies directory.
        //
        // The paths look like
        //
        // ```
        // build/<target_triple>/
        // ├── stageN-tools-bin/
        // │   └── librun_make_support.rlib       // <- support rlib itself
        // ├── stageN-tools/
        // │   ├── release/deps/                  // <- deps of deps
        // │   └── <host_triple>/release/deps/    // <- deps
        // ```
        //
        // FIXME(jieyouxu): there almost certainly is a better way to do this (specifically how the
        // support lib and its deps are organized, can't we copy them to the tools-bin dir as
        // well?), but this seems to work for now.

        let stage_number = self.config.stage;

        let stage_tools_bin = build_root.join(format!("stage{stage_number}-tools-bin"));
        let support_lib_path = stage_tools_bin.join("librun_make_support.rlib");

        let stage_tools = build_root.join(format!("stage{stage_number}-tools"));
        let support_lib_deps = stage_tools.join(&self.config.host).join("release").join("deps");
        let support_lib_deps_deps = stage_tools.join("release").join("deps");

        // To compile the recipe with rustc, we need to provide suitable dynamic library search
        // paths to rustc. This includes both:
        // 1. The "base" dylib search paths that was provided to compiletest, e.g. `LD_LIBRARY_PATH`
        //    on some linux distros.
        // 2. Specific library paths in `self.config.compile_lib_path` needed for running rustc.

        let base_dylib_search_paths =
            Vec::from_iter(env::split_paths(&env::var(dylib_env_var()).unwrap()));

        let host_dylib_search_paths = {
            let mut paths = vec![self.config.compile_lib_path.clone()];
            paths.extend(base_dylib_search_paths.iter().cloned());
            paths
        };

        // Calculate the paths of the recipe binary. As previously discussed, this is placed at
        // `<base_dir>/<bin_name>` with `bin_name` being `rmake` or `rmake.exe` depending on
        // platform.
        let recipe_bin = {
            let mut p = base_dir.join("rmake");
            p.set_extension(env::consts::EXE_EXTENSION);
            p
        };

        let mut rustc = Command::new(&self.config.rustc_path);
        rustc
            .arg("-o")
            .arg(&recipe_bin)
            // Specify library search paths for `run_make_support`.
            .arg(format!("-Ldependency={}", &support_lib_path.parent().unwrap().to_string_lossy()))
            .arg(format!("-Ldependency={}", &support_lib_deps.to_string_lossy()))
            .arg(format!("-Ldependency={}", &support_lib_deps_deps.to_string_lossy()))
            // Provide `run_make_support` as extern prelude, so test writers don't need to write
            // `extern run_make_support;`.
            .arg("--extern")
            .arg(format!("run_make_support={}", &support_lib_path.to_string_lossy()))
            .arg("--edition=2021")
            .arg(&self.testpaths.file.join("rmake.rs"))
            .arg("-Cprefer-dynamic")
            // Provide necessary library search paths for rustc.
            .env(dylib_env_var(), &env::join_paths(host_dylib_search_paths).unwrap());

        // In test code we want to be very pedantic about values being silently discarded that are
        // annotated with `#[must_use]`.
        rustc.arg("-Dunused_must_use");

        // > `cg_clif` uses `COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0` for running the rustc
        // > test suite. With the introduction of rmake.rs this broke. `librun_make_support.rlib` is
        // > compiled using the bootstrap rustc wrapper which sets `--sysroot
        // > build/aarch64-unknown-linux-gnu/stage0-sysroot`, but then compiletest will compile
        // > `rmake.rs` using the sysroot of the bootstrap compiler causing it to not find the
        // > `libstd.rlib` against which `librun_make_support.rlib` is compiled.
        //
        // The gist here is that we have to pass the proper stage0 sysroot if we want
        //
        // ```
        // $ COMPILETEST_FORCE_STAGE0=1 ./x test run-make --stage 0
        // ```
        //
        // to work correctly.
        //
        // See <https://github.com/rust-lang/rust/pull/122248> for more background.
        let stage0_sysroot = build_root.join("stage0-sysroot");
        if std::env::var_os("COMPILETEST_FORCE_STAGE0").is_some() {
            rustc.arg("--sysroot").arg(&stage0_sysroot);
        }

        // Now run rustc to build the recipe.
        let res = self.run_command_to_procres(&mut rustc);
        if !res.status.success() {
            self.fatal_proc_rec("run-make test failed: could not build `rmake.rs` recipe", &res);
        }

        // To actually run the recipe, we have to provide the recipe with a bunch of information
        // provided through env vars.

        // Compute stage-specific standard library paths.
        let stage_std_path = build_root.join(format!("stage{stage_number}")).join("lib");

        // Compute dynamic library search paths for recipes.
        let recipe_dylib_search_paths = {
            let mut paths = base_dylib_search_paths.clone();

            // For stage 0, we need to explicitly include the stage0-sysroot libstd dylib.
            // See <https://github.com/rust-lang/rust/issues/135373>.
            if std::env::var_os("COMPILETEST_FORCE_STAGE0").is_some() {
                paths.push(
                    stage0_sysroot.join("lib").join("rustlib").join(&self.config.host).join("lib"),
                );
            }

            paths.push(support_lib_path.parent().unwrap().to_path_buf());
            paths.push(stage_std_path.join("rustlib").join(&self.config.host).join("lib"));
            paths
        };

        // Compute runtime library search paths for recipes. This is target-specific.
        let target_runtime_dylib_search_paths = {
            let mut paths = vec![rmake_out_dir.clone()];
            paths.extend(base_dylib_search_paths.iter().cloned());
            paths
        };

        // FIXME(jieyouxu): please rename `TARGET_RPATH_ENV`, `HOST_RPATH_DIR` and
        // `TARGET_RPATH_DIR`, it is **extremely** confusing!
        let mut cmd = Command::new(&recipe_bin);
        cmd.current_dir(&rmake_out_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Provide the target-specific env var that is used to record dylib search paths. For
            // example, this could be `LD_LIBRARY_PATH` on some linux distros but `PATH` on Windows.
            .env("LD_LIB_PATH_ENVVAR", dylib_env_var())
            // Provide the dylib search paths.
            .env(dylib_env_var(), &env::join_paths(recipe_dylib_search_paths).unwrap())
            // Provide runtime dylib search paths.
            .env("TARGET_RPATH_ENV", &env::join_paths(target_runtime_dylib_search_paths).unwrap())
            // Provide the target.
            .env("TARGET", &self.config.target)
            // Some tests unfortunately still need Python, so provide path to a Python interpreter.
            .env("PYTHON", &self.config.python)
            // Provide path to checkout root. This is the top-level directory containing
            // rust-lang/rust checkout.
            .env("SOURCE_ROOT", &source_root)
            // Path to the build directory. This is usually the same as `source_root.join("build").join("host")`.
            .env("BUILD_ROOT", &build_root)
            // Provide path to stage-corresponding rustc.
            .env("RUSTC", &self.config.rustc_path)
            // Provide the directory to libraries that are needed to run the *compiler*. This is not
            // to be confused with `TARGET_RPATH_ENV` or `TARGET_RPATH_DIR`. This is needed if the
            // recipe wants to invoke rustc.
            .env("HOST_RPATH_DIR", &self.config.compile_lib_path)
            // Provide the directory to libraries that might be needed to run compiled binaries
            // (further compiled by the recipe!).
            .env("TARGET_RPATH_DIR", &self.config.run_lib_path)
            // Provide which LLVM components are available (e.g. which LLVM components are provided
            // through a specific CI runner).
            .env("LLVM_COMPONENTS", &self.config.llvm_components);

        if let Some(ref cargo) = self.config.cargo_path {
            cmd.env("CARGO", source_root.join(cargo));
        }

        if let Some(ref rustdoc) = self.config.rustdoc_path {
            cmd.env("RUSTDOC", source_root.join(rustdoc));
        }

        if let Some(ref node) = self.config.nodejs {
            cmd.env("NODE", node);
        }

        if let Some(ref linker) = self.config.target_linker {
            cmd.env("RUSTC_LINKER", linker);
        }

        if let Some(ref clang) = self.config.run_clang_based_tests_with {
            cmd.env("CLANG", clang);
        }

        if let Some(ref filecheck) = self.config.llvm_filecheck {
            cmd.env("LLVM_FILECHECK", filecheck);
        }

        if let Some(ref llvm_bin_dir) = self.config.llvm_bin_dir {
            cmd.env("LLVM_BIN_DIR", llvm_bin_dir);
        }

        if let Some(ref remote_test_client) = self.config.remote_test_client {
            cmd.env("REMOTE_TEST_CLIENT", remote_test_client);
        }

        // We don't want RUSTFLAGS set from the outside to interfere with
        // compiler flags set in the test cases:
        cmd.env_remove("RUSTFLAGS");

        // Use dynamic musl for tests because static doesn't allow creating dylibs
        if self.config.host.contains("musl") {
            cmd.env("RUSTFLAGS", "-Ctarget-feature=-crt-static").env("IS_MUSL_HOST", "1");
        }

        if self.config.bless {
            // If we're running in `--bless` mode, set an environment variable to tell
            // `run_make_support` to bless snapshot files instead of checking them.
            //
            // The value is this test's source directory, because the support code
            // will need that path in order to bless the _original_ snapshot files,
            // not the copies in `rmake_out`.
            // (See <https://github.com/rust-lang/rust/issues/129038>.)
            cmd.env("RUSTC_BLESS_TEST", &self.testpaths.file);
        }

        if self.config.target.contains("msvc") && !self.config.cc.is_empty() {
            // We need to pass a path to `lib.exe`, so assume that `cc` is `cl.exe`
            // and that `lib.exe` lives next to it.
            let lib = Path::new(&self.config.cc).parent().unwrap().join("lib.exe");

            // MSYS doesn't like passing flags of the form `/foo` as it thinks it's
            // a path and instead passes `C:\msys64\foo`, so convert all
            // `/`-arguments to MSVC here to `-` arguments.
            let cflags = self
                .config
                .cflags
                .split(' ')
                .map(|s| s.replace("/", "-"))
                .collect::<Vec<_>>()
                .join(" ");
            let cxxflags = self
                .config
                .cxxflags
                .split(' ')
                .map(|s| s.replace("/", "-"))
                .collect::<Vec<_>>()
                .join(" ");

            cmd.env("IS_MSVC", "1")
                .env("IS_WINDOWS", "1")
                .env("MSVC_LIB", format!("'{}' -nologo", lib.display()))
                .env("MSVC_LIB_PATH", format!("{}", lib.display()))
                // Note: we diverge from legacy run_make and don't lump `CC` the compiler and
                // default flags together.
                .env("CC_DEFAULT_FLAGS", &cflags)
                .env("CC", &self.config.cc)
                .env("CXX_DEFAULT_FLAGS", &cxxflags)
                .env("CXX", &self.config.cxx);
        } else {
            cmd.env("CC_DEFAULT_FLAGS", &self.config.cflags)
                .env("CC", &self.config.cc)
                .env("CXX_DEFAULT_FLAGS", &self.config.cxxflags)
                .env("CXX", &self.config.cxx)
                .env("AR", &self.config.ar);

            if self.config.target.contains("windows") {
                cmd.env("IS_WINDOWS", "1");
            }
        }

        let proc = disable_error_reporting(|| cmd.spawn().expect("failed to spawn `rmake`"));
        let (Output { stdout, stderr, status }, truncated) = self.read2_abbreviated(proc);
        let stdout = String::from_utf8_lossy(&stdout).into_owned();
        let stderr = String::from_utf8_lossy(&stderr).into_owned();
        // This conditions on `status.success()` so we don't print output twice on error.
        // NOTE: this code is called from a libtest thread, so it's hidden by default unless --nocapture is passed.
        self.dump_output(status.success(), &cmd.get_program().to_string_lossy(), &stdout, &stderr);
        if !status.success() {
            let res = ProcRes { status, stdout, stderr, truncated, cmdline: format!("{:?}", cmd) };
            self.fatal_proc_rec("rmake recipe failed to complete", &res);
        }
    }
}
