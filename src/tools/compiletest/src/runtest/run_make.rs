use std::process::{Command, Output, Stdio};
use std::{env, fs};

use build_helper::fs::{ignore_not_found, recursive_remove};
use camino::{Utf8Path, Utf8PathBuf};

use super::{ProcRes, TestCx, disable_error_reporting};
use crate::util::{copy_dir_all, dylib_env_var};

impl TestCx<'_> {
    pub(super) fn run_rmake_test(&self) {
        // For `run-make` V2, we need to perform 2 steps to build and run a `run-make` V2 recipe
        // (`rmake.rs`) to run the actual tests. The support library is already built as a tool rust
        // library and is available under
        // `build/$HOST/stage0-bootstrap-tools/$TARGET/release/librun_make_support.rlib`.
        //
        // 1. We need to build the recipe `rmake.rs` as a binary and link in the `run_make_support`
        //    library.
        // 2. We need to run the recipe binary.

        let host_build_root = self.config.build_root.join(&self.config.host);

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
        let base_dir = self.output_base_dir();
        ignore_not_found(|| recursive_remove(&base_dir)).unwrap();

        let rmake_out_dir = base_dir.join("rmake_out");
        fs::create_dir_all(&rmake_out_dir).unwrap();

        // Copy all input files (apart from rmake.rs) to the temporary directory,
        // so that the input directory structure from `tests/run-make/<test>` is mirrored
        // to the `rmake_out` directory.
        for entry in walkdir::WalkDir::new(&self.testpaths.file).min_depth(1) {
            let entry = entry.unwrap();
            let path = entry.path();
            let path = <&Utf8Path>::try_from(path).unwrap();
            if path.file_name().is_some_and(|s| s != "rmake.rs") {
                let target = rmake_out_dir.join(path.strip_prefix(&self.testpaths.file).unwrap());
                if path.is_dir() {
                    copy_dir_all(&path, &target).unwrap();
                } else {
                    fs::copy(path.as_std_path(), target).unwrap();
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
        // ├── stage0-bootstrap-tools/
        // │   ├── <host_triple>/release/librun_make_support.rlib   // <- support rlib itself
        // │   ├── <host_triple>/release/deps/                      // <- deps
        // │   └── release/deps/                                    // <- deps of deps
        // ```
        //
        // FIXME(jieyouxu): there almost certainly is a better way to do this (specifically how the
        // support lib and its deps are organized), but this seems to work for now.

        let tools_bin = host_build_root.join("stage0-bootstrap-tools");
        let support_host_path = tools_bin.join(&self.config.host).join("release");
        let support_lib_path = support_host_path.join("librun_make_support.rlib");

        let support_lib_deps = support_host_path.join("deps");
        let support_lib_deps_deps = tools_bin.join("release").join("deps");

        // To compile the recipe with rustc, we need to provide suitable dynamic library search
        // paths to rustc. This includes both:
        // 1. The "base" dylib search paths that was provided to compiletest, e.g. `LD_LIBRARY_PATH`
        //    on some linux distros.
        // 2. Specific library paths in `self.config.compile_lib_path` needed for running rustc.

        let base_dylib_search_paths = Vec::from_iter(
            env::split_paths(&env::var(dylib_env_var()).unwrap())
                .map(|p| Utf8PathBuf::try_from(p).expect("dylib env var contains non-UTF8 paths")),
        );

        // Calculate the paths of the recipe binary. As previously discussed, this is placed at
        // `<base_dir>/<bin_name>` with `bin_name` being `rmake` or `rmake.exe` depending on
        // platform.
        let recipe_bin = {
            let mut p = base_dir.join("rmake");
            p.set_extension(env::consts::EXE_EXTENSION);
            p
        };

        // run-make-support and run-make tests are compiled using the stage0 compiler
        // If the stage is 0, then the compiler that we test (either bootstrap or an explicitly
        // set compiler) is the one that actually compiled run-make-support.
        let stage0_rustc = self
            .config
            .stage0_rustc_path
            .as_ref()
            .expect("stage0 rustc is required to run run-make tests");
        let mut rustc = Command::new(&stage0_rustc);
        rustc
            // `rmake.rs` **must** be buildable by a stable compiler, it may not use *any* unstable
            // library or compiler features. Here, we force the stage 0 rustc to consider itself as
            // a stable-channel compiler via `RUSTC_BOOTSTRAP=-1` to prevent *any* unstable
            // library/compiler usages, even if stage 0 rustc is *actually* a nightly rustc.
            .env("RUSTC_BOOTSTRAP", "-1")
            .arg("-o")
            .arg(&recipe_bin)
            // Specify library search paths for `run_make_support`.
            .arg(format!("-Ldependency={}", &support_lib_path.parent().unwrap()))
            .arg(format!("-Ldependency={}", &support_lib_deps))
            .arg(format!("-Ldependency={}", &support_lib_deps_deps))
            // Provide `run_make_support` as extern prelude, so test writers don't need to write
            // `extern run_make_support;`.
            .arg("--extern")
            .arg(format!("run_make_support={}", &support_lib_path))
            .arg("--edition=2021")
            .arg(&self.testpaths.file.join("rmake.rs"))
            .arg("-Cprefer-dynamic");

        // In test code we want to be very pedantic about values being silently discarded that are
        // annotated with `#[must_use]`.
        rustc.arg("-Dunused_must_use");

        // Now run rustc to build the recipe.
        let res = self.run_command_to_procres(&mut rustc);
        if !res.status.success() {
            self.fatal_proc_rec("run-make test failed: could not build `rmake.rs` recipe", &res);
        }

        // To actually run the recipe, we have to provide the recipe with a bunch of information
        // provided through env vars.

        // Compute dynamic library search paths for recipes.
        // These dylib directories are needed to **execute the recipe**.
        let recipe_dylib_search_paths = {
            let mut paths = base_dylib_search_paths.clone();
            paths.push(
                stage0_rustc
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join("lib")
                    .join("rustlib")
                    .join(&self.config.host)
                    .join("lib"),
            );
            paths
        };

        let mut cmd = Command::new(&recipe_bin);
        cmd.current_dir(&rmake_out_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Provide the target-specific env var that is used to record dylib search paths. For
            // example, this could be `LD_LIBRARY_PATH` on some linux distros but `PATH` on Windows.
            .env("LD_LIB_PATH_ENVVAR", dylib_env_var())
            // Provide the dylib search paths.
            // This is required to run the **recipe** itself.
            .env(dylib_env_var(), &env::join_paths(recipe_dylib_search_paths).unwrap())
            // Provide the directory to libraries that are needed to run the *compiler* invoked
            // by the recipe.
            .env("HOST_RUSTC_DYLIB_PATH", &self.config.compile_lib_path)
            // Provide the directory to libraries that might be needed to run binaries created
            // by a compiler invoked by the recipe.
            .env("TARGET_EXE_DYLIB_PATH", &self.config.run_lib_path)
            // Provide the target.
            .env("TARGET", &self.config.target)
            // Some tests unfortunately still need Python, so provide path to a Python interpreter.
            .env("PYTHON", &self.config.python)
            // Provide path to sources root.
            .env("SOURCE_ROOT", &self.config.src_root)
            // Path to the host build directory.
            .env("BUILD_ROOT", &host_build_root)
            // Provide path to stage-corresponding rustc.
            .env("RUSTC", &self.config.rustc_path)
            // Provide which LLVM components are available (e.g. which LLVM components are provided
            // through a specific CI runner).
            .env("LLVM_COMPONENTS", &self.config.llvm_components);

        if let Some(ref cargo) = self.config.cargo_path {
            cmd.env("CARGO", cargo);
        }

        if let Some(ref rustdoc) = self.config.rustdoc_path {
            cmd.env("RUSTDOC", rustdoc);
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
            let lib = Utf8Path::new(&self.config.cc).parent().unwrap().join("lib.exe");

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
                .env("MSVC_LIB", format!("'{}' -nologo", lib))
                .env("MSVC_LIB_PATH", &lib)
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
