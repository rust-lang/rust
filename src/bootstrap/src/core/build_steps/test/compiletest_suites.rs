//! Test suites managed and ran by `src/tools/compiletest`.

use std::path::{Path, PathBuf};
use std::{env, fs};

use super::test_helpers::{RemoteCopyLibs, run_cargo_test};
use crate::core::build_steps::doc::DocumentationFormat;
use crate::core::build_steps::llvm::get_llvm_version;
use crate::core::build_steps::synthetic_targets::MirOptPanicAbortSyntheticTarget;
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::build_steps::{compile, dist, llvm};
use crate::core::builder::{Builder, Compiler, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::core::config::flags::Subcommand;
use crate::utils::build_stamp::{self};
use crate::utils::exec::command;
use crate::utils::helpers::{
    self, LldThreads, add_rustdoc_cargo_linker_args, linker_flags, t, up_to_date,
};
use crate::utils::render_tests::try_run_tests;
use crate::{CLang, DocTests, GitRepo, Mode, PathSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Compiletest {
    compiler: Compiler,
    target: TargetSelection,
    mode: &'static str,
    suite: &'static str,
    path: &'static str,
    compare_mode: Option<&'static str>,
}

impl Step for Compiletest {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Executes the `compiletest` tool to run a suite of tests.
    ///
    /// Compiles all tests with `compiler` for `target` with the specified
    /// compiletest `mode` and `suite` arguments. For example `mode` can be
    /// "run-pass" or `suite` can be something like `debuginfo`.
    fn run(self, builder: &Builder<'_>) {
        if builder.doc_tests == DocTests::Only {
            return;
        }

        if builder.top_stage == 0 && env::var("COMPILETEST_FORCE_STAGE0").is_err() {
            eprintln!("\
ERROR: `--stage 0` runs compiletest on the beta compiler, not your local changes, and will almost always cause tests to fail
HELP: to test the compiler, use `--stage 1` instead
HELP: to test the standard library, use `--stage 0 library/std` instead
NOTE: if you're sure you want to do this, please open an issue as to why. In the meantime, you can override this with `COMPILETEST_FORCE_STAGE0=1`."
            );
            crate::exit!(1);
        }

        let mut compiler = self.compiler;
        let target = self.target;
        let mode = self.mode;
        let suite = self.suite;

        // Path for test suite
        let suite_path = self.path;

        // Skip codegen tests if they aren't enabled in configuration.
        if !builder.config.codegen_tests && suite == "codegen" {
            return;
        }

        // Support stage 1 ui-fulldeps. This is somewhat complicated: ui-fulldeps tests for the most
        // part test the *API* of the compiler, not how it compiles a given file. As a result, we
        // can run them against the stage 1 sources as long as we build them with the stage 0
        // bootstrap compiler.
        // NOTE: Only stage 1 is special cased because we need the rustc_private artifacts to match the
        // running compiler in stage 2 when plugins run.
        let (stage, stage_id) = if suite == "ui-fulldeps" && compiler.stage == 1 {
            // At stage 0 (stage - 1) we are using the beta compiler. Using `self.target` can lead
            // finding an incorrect compiler path on cross-targets, as the stage 0 beta compiler is
            // always equal to `build.build` in the configuration.
            let build = builder.build.build;
            compiler = builder.compiler(compiler.stage - 1, build);
            let test_stage = compiler.stage + 1;
            (test_stage, format!("stage{}-{}", test_stage, build))
        } else {
            let stage = compiler.stage;
            (stage, format!("stage{}-{}", stage, target))
        };

        if suite.ends_with("fulldeps") {
            builder.ensure(compile::Rustc::new(compiler, target));
        }

        if suite == "debuginfo" {
            builder.ensure(dist::DebuggerScripts {
                sysroot: builder.sysroot(compiler).to_path_buf(),
                host: target,
            });
        }

        // Also provide `rust_test_helpers` for the host.
        builder.ensure(TestHelpers { target: compiler.host });

        // ensure that `libproc_macro` is available on the host.
        if suite == "mir-opt" {
            builder.ensure(compile::Std::new(compiler, compiler.host).is_for_mir_opt_tests(true));
        } else {
            builder.ensure(compile::Std::new(compiler, compiler.host));
        }

        // As well as the target
        if suite != "mir-opt" {
            builder.ensure(TestHelpers { target });
        }

        let mut cmd = builder.tool_cmd(Tool::Compiletest);

        if suite == "mir-opt" {
            builder.ensure(compile::Std::new(compiler, target).is_for_mir_opt_tests(true));
        } else {
            builder.ensure(compile::Std::new(compiler, target));
        }

        builder.ensure(RemoteCopyLibs { compiler, target });

        // compiletest currently has... a lot of arguments, so let's just pass all
        // of them!

        cmd.arg("--stage").arg(stage.to_string());
        cmd.arg("--stage-id").arg(stage_id);

        cmd.arg("--compile-lib-path").arg(builder.rustc_libdir(compiler));
        cmd.arg("--run-lib-path").arg(builder.sysroot_target_libdir(compiler, target));
        cmd.arg("--rustc-path").arg(builder.rustc(compiler));

        // Minicore auxiliary lib for `no_core` tests that need `core` stubs in cross-compilation
        // scenarios.
        cmd.arg("--minicore-path")
            .arg(builder.src.join("tests").join("auxiliary").join("minicore.rs"));

        let is_rustdoc = suite == "rustdoc-ui" || suite == "rustdoc-js";

        if mode == "run-make" {
            let cargo_path = if builder.top_stage == 0 {
                // If we're using `--stage 0`, we should provide the bootstrap cargo.
                builder.initial_cargo.clone()
            } else {
                // We need to properly build cargo using the suitable stage compiler.

                let compiler = builder.download_rustc().then_some(compiler).unwrap_or_else(||
                    // HACK: currently tool stages are off-by-one compared to compiler stages, i.e. if
                    // you give `tool::Cargo` a stage 1 rustc, it will cause stage 2 rustc to be built
                    // and produce a cargo built with stage 2 rustc. To fix this, we need to chop off
                    // the compiler stage by 1 to align with expected `./x test run-make --stage N`
                    // behavior, i.e. we need to pass `N - 1` compiler stage to cargo. See also Miri
                    // which does a similar hack.
                    builder.compiler(builder.top_stage - 1, compiler.host));

                builder.ensure(tool::Cargo { compiler, target: compiler.host })
            };

            cmd.arg("--cargo-path").arg(cargo_path);
        }

        // Avoid depending on rustdoc when we don't need it.
        if mode == "rustdoc"
            || mode == "run-make"
            || (mode == "ui" && is_rustdoc)
            || mode == "rustdoc-js"
            || mode == "rustdoc-json"
            || suite == "coverage-run-rustdoc"
        {
            cmd.arg("--rustdoc-path").arg(builder.rustdoc(compiler));
        }

        if mode == "rustdoc-json" {
            // Use the beta compiler for jsondocck
            let json_compiler = compiler.with_stage(0);
            cmd.arg("--jsondocck-path")
                .arg(builder.ensure(tool::JsonDocCk { compiler: json_compiler, target }));
            cmd.arg("--jsondoclint-path")
                .arg(builder.ensure(tool::JsonDocLint { compiler: json_compiler, target }));
        }

        if matches!(mode, "coverage-map" | "coverage-run") {
            let coverage_dump = builder.tool_exe(Tool::CoverageDump);
            cmd.arg("--coverage-dump-path").arg(coverage_dump);
        }

        cmd.arg("--src-base").arg(builder.src.join("tests").join(suite));
        cmd.arg("--build-base").arg(builder.test_out(compiler.host).join(suite));

        // When top stage is 0, that means that we're testing an externally provided compiler.
        // In that case we need to use its specific sysroot for tests to pass.
        let sysroot = if builder.top_stage == 0 {
            builder.initial_sysroot.clone()
        } else {
            builder.sysroot(compiler).to_path_buf()
        };

        cmd.arg("--sysroot-base").arg(sysroot);

        cmd.arg("--suite").arg(suite);
        cmd.arg("--mode").arg(mode);
        cmd.arg("--target").arg(target.rustc_target_arg());
        cmd.arg("--host").arg(&*compiler.host.triple);
        cmd.arg("--llvm-filecheck").arg(builder.llvm_filecheck(builder.config.build));

        if builder.build.config.llvm_enzyme {
            cmd.arg("--has-enzyme");
        }

        if builder.config.cmd.bless() {
            cmd.arg("--bless");
        }

        if builder.config.cmd.force_rerun() {
            cmd.arg("--force-rerun");
        }

        if builder.config.cmd.no_capture() {
            cmd.arg("--no-capture");
        }

        let compare_mode =
            builder.config.cmd.compare_mode().or_else(|| {
                if builder.config.test_compare_mode { self.compare_mode } else { None }
            });

        if let Some(ref pass) = builder.config.cmd.pass() {
            cmd.arg("--pass");
            cmd.arg(pass);
        }

        if let Some(ref run) = builder.config.cmd.run() {
            cmd.arg("--run");
            cmd.arg(run);
        }

        if let Some(ref nodejs) = builder.config.nodejs {
            cmd.arg("--nodejs").arg(nodejs);
        } else if mode == "rustdoc-js" {
            panic!("need nodejs to run rustdoc-js suite");
        }
        if let Some(ref npm) = builder.config.npm {
            cmd.arg("--npm").arg(npm);
        }
        if builder.config.rust_optimize_tests {
            cmd.arg("--optimize-tests");
        }
        if builder.config.rust_randomize_layout {
            cmd.arg("--rust-randomized-layout");
        }
        if builder.config.cmd.only_modified() {
            cmd.arg("--only-modified");
        }
        if let Some(compiletest_diff_tool) = &builder.config.compiletest_diff_tool {
            cmd.arg("--compiletest-diff-tool").arg(compiletest_diff_tool);
        }

        let mut flags = if is_rustdoc { Vec::new() } else { vec!["-Crpath".to_string()] };
        flags.push(format!("-Cdebuginfo={}", builder.config.rust_debuginfo_level_tests));
        flags.extend(builder.config.cmd.compiletest_rustc_args().iter().map(|s| s.to_string()));

        if suite != "mir-opt" {
            if let Some(linker) = builder.linker(target) {
                cmd.arg("--target-linker").arg(linker);
            }
            if let Some(linker) = builder.linker(compiler.host) {
                cmd.arg("--host-linker").arg(linker);
            }
        }

        // FIXME(136096): on macOS, we get linker warnings about duplicate `-lm` flags.
        // NOTE: `stage > 1` here because `test --stage 1 ui-fulldeps` is a hack that compiles
        // with stage 0, but links the tests against stage 1.
        // cfg(bootstrap) - remove only the `stage > 1` check, leave everything else.
        if suite == "ui-fulldeps" && compiler.stage > 1 && target.ends_with("darwin") {
            flags.push("-Alinker_messages".into());
        }

        let mut hostflags = flags.clone();
        hostflags.push(format!("-Lnative={}", builder.test_helpers_out(compiler.host).display()));
        hostflags.extend(linker_flags(builder, compiler.host, LldThreads::No));

        let mut targetflags = flags;
        targetflags.push(format!("-Lnative={}", builder.test_helpers_out(target).display()));

        for flag in hostflags {
            cmd.arg("--host-rustcflags").arg(flag);
        }
        for flag in targetflags {
            cmd.arg("--target-rustcflags").arg(flag);
        }

        cmd.arg("--python").arg(builder.python());

        if let Some(ref gdb) = builder.config.gdb {
            cmd.arg("--gdb").arg(gdb);
        }

        let lldb_exe = builder.config.lldb.clone().unwrap_or_else(|| PathBuf::from("lldb"));
        let lldb_version = command(&lldb_exe)
            .allow_failure()
            .arg("--version")
            .run_capture(builder)
            .stdout_if_ok()
            .and_then(|v| if v.trim().is_empty() { None } else { Some(v) });
        if let Some(ref vers) = lldb_version {
            cmd.arg("--lldb-version").arg(vers);
            let lldb_python_dir = command(&lldb_exe)
                .allow_failure()
                .arg("-P")
                .run_capture_stdout(builder)
                .stdout_if_ok()
                .map(|p| p.lines().next().expect("lldb Python dir not found").to_string());
            if let Some(ref dir) = lldb_python_dir {
                cmd.arg("--lldb-python-dir").arg(dir);
            }
        }

        if helpers::forcing_clang_based_tests() {
            let clang_exe = builder.llvm_out(target).join("bin").join("clang");
            cmd.arg("--run-clang-based-tests-with").arg(clang_exe);
        }

        for exclude in &builder.config.skip {
            cmd.arg("--skip");
            cmd.arg(exclude);
        }

        // Get paths from cmd args
        let paths = match &builder.config.cmd {
            Subcommand::Test { .. } => &builder.config.paths[..],
            _ => &[],
        };

        // Get test-args by striping suite path
        let mut test_args: Vec<&str> = paths
            .iter()
            .filter_map(|p| helpers::is_valid_test_suite_arg(p, suite_path, builder))
            .collect();

        test_args.append(&mut builder.config.test_args());

        // On Windows, replace forward slashes in test-args by backslashes
        // so the correct filters are passed to libtest
        if cfg!(windows) {
            let test_args_win: Vec<String> =
                test_args.iter().map(|s| s.replace('/', "\\")).collect();
            cmd.args(&test_args_win);
        } else {
            cmd.args(&test_args);
        }

        if builder.is_verbose() {
            cmd.arg("--verbose");
        }

        cmd.arg("--json");

        if builder.config.rustc_debug_assertions {
            cmd.arg("--with-rustc-debug-assertions");
        }

        if builder.config.std_debug_assertions {
            cmd.arg("--with-std-debug-assertions");
        }

        let mut llvm_components_passed = false;
        let mut copts_passed = false;
        if builder.config.llvm_enabled(compiler.host) {
            let llvm::LlvmResult { llvm_config, .. } =
                builder.ensure(llvm::Llvm { target: builder.config.build });
            if !builder.config.dry_run() {
                let llvm_version = get_llvm_version(builder, &llvm_config);
                let llvm_components =
                    command(&llvm_config).arg("--components").run_capture_stdout(builder).stdout();
                // Remove trailing newline from llvm-config output.
                cmd.arg("--llvm-version")
                    .arg(llvm_version.trim())
                    .arg("--llvm-components")
                    .arg(llvm_components.trim());
                llvm_components_passed = true;
            }
            if !builder.is_rust_llvm(target) {
                // FIXME: missing Rust patches is not the same as being system llvm; we should rename the flag at some point.
                // Inspecting the tests with `// no-system-llvm` in src/test *looks* like this is doing the right thing, though.
                cmd.arg("--system-llvm");
            }

            // Tests that use compiler libraries may inherit the `-lLLVM` link
            // requirement, but the `-L` library path is not propagated across
            // separate compilations. We can add LLVM's library path to the
            // rustc args as a workaround.
            if !builder.config.dry_run() && suite.ends_with("fulldeps") {
                let llvm_libdir =
                    command(&llvm_config).arg("--libdir").run_capture_stdout(builder).stdout();
                let link_llvm = if target.is_msvc() {
                    format!("-Clink-arg=-LIBPATH:{llvm_libdir}")
                } else {
                    format!("-Clink-arg=-L{llvm_libdir}")
                };
                cmd.arg("--host-rustcflags").arg(link_llvm);
            }

            if !builder.config.dry_run() && matches!(mode, "run-make" | "coverage-run") {
                // The llvm/bin directory contains many useful cross-platform
                // tools. Pass the path to run-make tests so they can use them.
                // (The coverage-run tests also need these tools to process
                // coverage reports.)
                let llvm_bin_path = llvm_config
                    .parent()
                    .expect("Expected llvm-config to be contained in directory");
                assert!(llvm_bin_path.is_dir());
                cmd.arg("--llvm-bin-dir").arg(llvm_bin_path);
            }

            if !builder.config.dry_run() && mode == "run-make" {
                // If LLD is available, add it to the PATH
                if builder.config.lld_enabled {
                    let lld_install_root =
                        builder.ensure(llvm::Lld { target: builder.config.build });

                    let lld_bin_path = lld_install_root.join("bin");

                    let old_path = env::var_os("PATH").unwrap_or_default();
                    let new_path = env::join_paths(
                        std::iter::once(lld_bin_path).chain(env::split_paths(&old_path)),
                    )
                    .expect("Could not add LLD bin path to PATH");
                    cmd.env("PATH", new_path);
                }
            }
        }

        // Only pass correct values for these flags for the `run-make` suite as it
        // requires that a C++ compiler was configured which isn't always the case.
        if !builder.config.dry_run() && mode == "run-make" {
            let mut cflags = builder.cc_handled_clags(target, CLang::C);
            cflags.extend(builder.cc_unhandled_cflags(target, GitRepo::Rustc, CLang::C));
            let mut cxxflags = builder.cc_handled_clags(target, CLang::Cxx);
            cxxflags.extend(builder.cc_unhandled_cflags(target, GitRepo::Rustc, CLang::Cxx));
            cmd.arg("--cc")
                .arg(builder.cc(target))
                .arg("--cxx")
                .arg(builder.cxx(target).unwrap())
                .arg("--cflags")
                .arg(cflags.join(" "))
                .arg("--cxxflags")
                .arg(cxxflags.join(" "));
            copts_passed = true;
            if let Some(ar) = builder.ar(target) {
                cmd.arg("--ar").arg(ar);
            }
        }

        if !llvm_components_passed {
            cmd.arg("--llvm-components").arg("");
        }
        if !copts_passed {
            cmd.arg("--cc")
                .arg("")
                .arg("--cxx")
                .arg("")
                .arg("--cflags")
                .arg("")
                .arg("--cxxflags")
                .arg("");
        }

        if builder.remote_tested(target) {
            cmd.arg("--remote-test-client").arg(builder.tool_exe(Tool::RemoteTestClient));
        } else if let Some(tool) = builder.runner(target) {
            cmd.arg("--runner").arg(tool);
        }

        if suite != "mir-opt" {
            // Running a C compiler on MSVC requires a few env vars to be set, to be
            // sure to set them here.
            //
            // Note that if we encounter `PATH` we make sure to append to our own `PATH`
            // rather than stomp over it.
            if !builder.config.dry_run() && target.is_msvc() {
                for (k, v) in builder.cc.borrow()[&target].env() {
                    if k != "PATH" {
                        cmd.env(k, v);
                    }
                }
            }
        }

        // Special setup to enable running with sanitizers on MSVC.
        if !builder.config.dry_run()
            && target.contains("msvc")
            && builder.config.sanitizers_enabled(target)
        {
            // Ignore interception failures: not all dlls in the process will have been built with
            // address sanitizer enabled (e.g., ntdll.dll).
            cmd.env("ASAN_WIN_CONTINUE_ON_INTERCEPTION_FAILURE", "1");
            // Add the address sanitizer runtime to the PATH - it is located next to cl.exe.
            let asan_runtime_path =
                builder.cc.borrow()[&target].path().parent().unwrap().to_path_buf();
            let old_path = cmd
                .get_envs()
                .find_map(|(k, v)| (k == "PATH").then_some(v))
                .flatten()
                .map_or_else(|| env::var_os("PATH").unwrap_or_default(), |v| v.to_owned());
            let new_path = env::join_paths(
                env::split_paths(&old_path).chain(std::iter::once(asan_runtime_path)),
            )
            .expect("Could not add ASAN runtime path to PATH");
            cmd.env("PATH", new_path);
        }

        // Some UI tests trigger behavior in rustc where it reads $CARGO and changes behavior if it exists.
        // To make the tests work that rely on it not being set, make sure it is not set.
        cmd.env_remove("CARGO");

        cmd.env("RUSTC_BOOTSTRAP", "1");
        // Override the rustc version used in symbol hashes to reduce the amount of normalization
        // needed when diffing test output.
        cmd.env("RUSTC_FORCE_RUSTC_VERSION", "compiletest");
        cmd.env("DOC_RUST_LANG_ORG_CHANNEL", builder.doc_rust_lang_org_channel());
        builder.add_rust_test_threads(&mut cmd);

        if builder.config.sanitizers_enabled(target) {
            cmd.env("RUSTC_SANITIZER_SUPPORT", "1");
        }

        if builder.config.profiler_enabled(target) {
            cmd.arg("--profiler-runtime");
        }

        cmd.env("RUST_TEST_TMPDIR", builder.tempdir());

        cmd.arg("--adb-path").arg("adb");

        const ADB_TEST_DIR: &str = "/data/local/tmp/work";
        cmd.arg("--adb-test-dir").arg(ADB_TEST_DIR);
        if target.contains("android") && !builder.config.dry_run() {
            // Assume that cc for this target comes from the android sysroot
            cmd.arg("--android-cross-path")
                .arg(builder.cc(target).parent().unwrap().parent().unwrap());
        } else {
            cmd.arg("--android-cross-path").arg("");
        }

        if builder.config.cmd.rustfix_coverage() {
            cmd.arg("--rustfix-coverage");
        }

        cmd.arg("--channel").arg(&builder.config.channel);

        if !builder.config.omit_git_hash {
            cmd.arg("--git-hash");
        }

        let git_config = builder.config.git_config();
        cmd.arg("--git-repository").arg(git_config.git_repository);
        cmd.arg("--nightly-branch").arg(git_config.nightly_branch);
        cmd.arg("--git-merge-commit-email").arg(git_config.git_merge_commit_email);
        cmd.force_coloring_in_ci();

        #[cfg(feature = "build-metrics")]
        builder.metrics.begin_test_suite(
            build_helper::metrics::TestSuiteMetadata::Compiletest {
                suite: suite.into(),
                mode: mode.into(),
                compare_mode: None,
                target: self.target.triple.to_string(),
                host: self.compiler.host.triple.to_string(),
                stage: self.compiler.stage,
            },
            builder,
        );

        let _group = builder.msg(
            Kind::Test,
            compiler.stage,
            format!("compiletest suite={suite} mode={mode}"),
            compiler.host,
            target,
        );
        try_run_tests(builder, &mut cmd, false);

        if let Some(compare_mode) = compare_mode {
            cmd.arg("--compare-mode").arg(compare_mode);

            #[cfg(feature = "build-metrics")]
            builder.metrics.begin_test_suite(
                build_helper::metrics::TestSuiteMetadata::Compiletest {
                    suite: suite.into(),
                    mode: mode.into(),
                    compare_mode: Some(compare_mode.into()),
                    target: self.target.triple.to_string(),
                    host: self.compiler.host.triple.to_string(),
                    stage: self.compiler.stage,
                },
                builder,
            );

            builder.info(&format!(
                "Check compiletest suite={} mode={} compare_mode={} ({} -> {})",
                suite, mode, compare_mode, &compiler.host, target
            ));
            let _time = helpers::timeit(builder);
            try_run_tests(builder, &mut cmd, false);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TestHelpers {
    pub target: TargetSelection,
}

impl Step for TestHelpers {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("tests/auxiliary/rust_test_helpers.c")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(TestHelpers { target: run.target })
    }

    /// Compiles the `rust_test_helpers.c` library which we used in various
    /// `run-pass` tests for ABI testing.
    fn run(self, builder: &Builder<'_>) {
        if builder.config.dry_run() {
            return;
        }
        // The x86_64-fortanix-unknown-sgx target doesn't have a working C
        // toolchain. However, some x86_64 ELF objects can be linked
        // without issues. Use this hack to compile the test helpers.
        let target = if self.target == "x86_64-fortanix-unknown-sgx" {
            TargetSelection::from_user("x86_64-unknown-linux-gnu")
        } else {
            self.target
        };
        let dst = builder.test_helpers_out(target);
        let src = builder.src.join("tests/auxiliary/rust_test_helpers.c");
        if up_to_date(&src, &dst.join("librust_test_helpers.a")) {
            return;
        }

        let _guard = builder.msg_unstaged(Kind::Build, "test helpers", target);
        t!(fs::create_dir_all(&dst));
        let mut cfg = cc::Build::new();

        // We may have found various cross-compilers a little differently due to our
        // extra configuration, so inform cc of these compilers. Note, though, that
        // on MSVC we still need cc's detection of env vars (ugh).
        if !target.is_msvc() {
            if let Some(ar) = builder.ar(target) {
                cfg.archiver(ar);
            }
            cfg.compiler(builder.cc(target));
        }
        cfg.cargo_metadata(false)
            .out_dir(&dst)
            .target(&target.triple)
            .host(&builder.config.build.triple)
            .opt_level(0)
            .warnings(false)
            .debug(false)
            .file(builder.src.join("tests/auxiliary/rust_test_helpers.c"))
            .compile("rust_test_helpers");
    }
}

/// Declares a test step that invokes compiletest on a particular test suite.
macro_rules! test {
    (
        $( #[$attr:meta] )* // allow docstrings and attributes
        $name:ident {
            path: $path:expr,
            mode: $mode:expr,
            suite: $suite:expr,
            default: $default:expr
            $( , only_hosts: $only_hosts:expr )? // default: false
            $( , compare_mode: $compare_mode:expr )? // default: None
            $( , )? // optional trailing comma
        }
    ) => {
        $( #[$attr] )*
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: TargetSelection,
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = $default;
            const ONLY_HOSTS: bool = (const {
                #[allow(unused_assignments, unused_mut)]
                let mut value = false;
                $( value = $only_hosts; )?
                value
            });

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.suite_path($path)
            }

            fn make_run(run: RunConfig<'_>) {
                let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());

                run.builder.ensure($name { compiler, target: run.target });
            }

            fn run(self, builder: &Builder<'_>) {
                builder.ensure(Compiletest {
                    compiler: self.compiler,
                    target: self.target,
                    mode: $mode,
                    suite: $suite,
                    path: $path,
                    compare_mode: (const {
                        #[allow(unused_assignments, unused_mut)]
                        let mut value = None;
                        $( value = $compare_mode; )?
                        value
                    }),
                })
            }
        }
    };
}

test!(Ui { path: "tests/ui", mode: "ui", suite: "ui", default: true });

test!(Crashes { path: "tests/crashes", mode: "crashes", suite: "crashes", default: true });

test!(Codegen { path: "tests/codegen", mode: "codegen", suite: "codegen", default: true });

test!(CodegenUnits {
    path: "tests/codegen-units",
    mode: "codegen-units",
    suite: "codegen-units",
    default: true,
});

test!(Incremental {
    path: "tests/incremental",
    mode: "incremental",
    suite: "incremental",
    default: true,
});

test!(Debuginfo {
    path: "tests/debuginfo",
    mode: "debuginfo",
    suite: "debuginfo",
    default: true,
    compare_mode: Some("split-dwarf"),
});

test!(UiFullDeps {
    path: "tests/ui-fulldeps",
    mode: "ui",
    suite: "ui-fulldeps",
    default: true,
    only_hosts: true,
});

test!(Rustdoc {
    path: "tests/rustdoc",
    mode: "rustdoc",
    suite: "rustdoc",
    default: true,
    only_hosts: true,
});
test!(RustdocUi {
    path: "tests/rustdoc-ui",
    mode: "ui",
    suite: "rustdoc-ui",
    default: true,
    only_hosts: true,
});

test!(RustdocJson {
    path: "tests/rustdoc-json",
    mode: "rustdoc-json",
    suite: "rustdoc-json",
    default: true,
    only_hosts: true,
});

test!(Pretty {
    path: "tests/pretty",
    mode: "pretty",
    suite: "pretty",
    default: true,
    only_hosts: true,
});

/// Special-handling is needed for `run-make`, so don't use `test!` for defining `RunMake`
/// tests.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct RunMake {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RunMake {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.suite_path("tests/run-make")
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(RunMakeSupport { compiler, target: run.build_triple() });
        run.builder.ensure(RunMake { compiler, target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(Compiletest {
            compiler: self.compiler,
            target: self.target,
            mode: "run-make",
            suite: "run-make",
            path: "tests/run-make",
            compare_mode: None,
        });
    }
}

test!(Assembly { path: "tests/assembly", mode: "assembly", suite: "assembly", default: true });

/// Runs the coverage test suite at `tests/coverage` in some or all of the
/// coverage test modes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Coverage {
    pub compiler: Compiler,
    pub target: TargetSelection,
    pub mode: &'static str,
}

impl Coverage {
    const PATH: &'static str = "tests/coverage";
    const SUITE: &'static str = "coverage";
    const ALL_MODES: &[&str] = &["coverage-map", "coverage-run"];
}

impl Step for Coverage {
    type Output = ();
    const DEFAULT: bool = true;
    /// Compiletest will automatically skip the "coverage-run" tests if necessary.
    const ONLY_HOSTS: bool = false;

    fn should_run(mut run: ShouldRun<'_>) -> ShouldRun<'_> {
        // Support various invocation styles, including:
        // - `./x test coverage`
        // - `./x test tests/coverage/trivial.rs`
        // - `./x test coverage-map`
        // - `./x test coverage-run -- tests/coverage/trivial.rs`
        run = run.suite_path(Self::PATH);
        for mode in Self::ALL_MODES {
            run = run.alias(mode);
        }
        run
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        let target = run.target;

        // List of (coverage) test modes that the coverage test suite will be
        // run in. It's OK for this to contain duplicates, because the call to
        // `Builder::ensure` below will take care of deduplication.
        let mut modes = vec![];

        // From the pathsets that were selected on the command-line (or by default),
        // determine which modes to run in.
        for path in &run.paths {
            match path {
                PathSet::Set(_) => {
                    for mode in Self::ALL_MODES {
                        if path.assert_single_path().path == Path::new(mode) {
                            modes.push(mode);
                            break;
                        }
                    }
                }
                PathSet::Suite(_) => {
                    modes.extend(Self::ALL_MODES);
                    break;
                }
            }
        }

        // Skip any modes that were explicitly skipped/excluded on the command-line.
        // FIXME(Zalathar): Integrate this into central skip handling somehow?
        modes.retain(|mode| !run.builder.config.skip.iter().any(|skip| skip == Path::new(mode)));

        // FIXME(Zalathar): Make these commands skip all coverage tests, as expected:
        // - `./x test --skip=tests`
        // - `./x test --skip=tests/coverage`
        // - `./x test --skip=coverage`
        // Skip handling currently doesn't have a way to know that skipping the coverage
        // suite should also skip the `coverage-map` and `coverage-run` aliases.

        for mode in modes {
            run.builder.ensure(Coverage { compiler, target, mode });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let Self { compiler, target, mode } = self;
        // Like other compiletest suite test steps, delegate to an internal
        // compiletest task to actually run the tests.
        builder.ensure(Compiletest {
            compiler,
            target,
            mode,
            suite: Self::SUITE,
            path: Self::PATH,
            compare_mode: None,
        });
    }
}

test!(CoverageRunRustdoc {
    path: "tests/coverage-run-rustdoc",
    mode: "coverage-run",
    suite: "coverage-run-rustdoc",
    default: true,
    only_hosts: true,
});

// For the mir-opt suite we do not use macros, as we need custom behavior when blessing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MirOpt {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for MirOpt {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.suite_path("tests/mir-opt")
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(MirOpt { compiler, target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let run = |target| {
            builder.ensure(Compiletest {
                compiler: self.compiler,
                target,
                mode: "mir-opt",
                suite: "mir-opt",
                path: "tests/mir-opt",
                compare_mode: None,
            })
        };

        run(self.target);

        // Run more targets with `--bless`. But we always run the host target first, since some
        // tests use very specific `only` clauses that are not covered by the target set below.
        if builder.config.cmd.bless() {
            // All that we really need to do is cover all combinations of 32/64-bit and unwind/abort,
            // but while we're at it we might as well flex our cross-compilation support. This
            // selection covers all our tier 1 operating systems and architectures using only tier
            // 1 targets.

            for target in ["aarch64-unknown-linux-gnu", "i686-pc-windows-msvc"] {
                run(TargetSelection::from_user(target));
            }

            for target in ["x86_64-apple-darwin", "i686-unknown-linux-musl"] {
                let target = TargetSelection::from_user(target);
                let panic_abort_target = builder.ensure(MirOptPanicAbortSyntheticTarget {
                    compiler: self.compiler,
                    base: target,
                });
                run(panic_abort_target);
            }
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJSStd {
    pub target: TargetSelection,
}

impl Step for RustdocJSStd {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = run.builder.config.nodejs.is_some();
        run.suite_path("tests/rustdoc-js-std").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustdocJSStd { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let nodejs =
            builder.config.nodejs.as_ref().expect("need nodejs to run rustdoc-js-std tests");
        let mut command = command(nodejs);
        command
            .arg(builder.src.join("src/tools/rustdoc-js/tester.js"))
            .arg("--crate-name")
            .arg("std")
            .arg("--resource-suffix")
            .arg(&builder.version)
            .arg("--doc-folder")
            .arg(builder.doc_out(self.target))
            .arg("--test-folder")
            .arg(builder.src.join("tests/rustdoc-js-std"));
        for path in &builder.paths {
            if let Some(p) = helpers::is_valid_test_suite_arg(path, "tests/rustdoc-js-std", builder)
            {
                if !p.ends_with(".js") {
                    eprintln!("A non-js file was given: `{}`", path.display());
                    panic!("Cannot run rustdoc-js-std tests");
                }
                command.arg("--test-file").arg(path);
            }
        }
        builder.ensure(crate::core::build_steps::doc::Std::new(
            builder.top_stage,
            self.target,
            DocumentationFormat::Html,
        ));
        let _guard = builder.msg(
            Kind::Test,
            builder.top_stage,
            "rustdoc-js-std",
            builder.config.build,
            self.target,
        );
        command.run(builder);
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustdocJSNotStd {
    pub target: TargetSelection,
    pub compiler: Compiler,
}

impl Step for RustdocJSNotStd {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = run.builder.config.nodejs.is_some();
        run.suite_path("tests/rustdoc-js").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(RustdocJSNotStd { target: run.target, compiler });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(Compiletest {
            compiler: self.compiler,
            target: self.target,
            mode: "rustdoc-js",
            suite: "rustdoc-js",
            path: "tests/rustdoc-js",
            compare_mode: None,
        });
    }
}

fn get_browser_ui_test_version_inner(
    builder: &Builder<'_>,
    npm: &Path,
    global: bool,
) -> Option<String> {
    let mut command = command(npm);
    command.arg("list").arg("--parseable").arg("--long").arg("--depth=0");
    if global {
        command.arg("--global");
    }
    let lines = command.allow_failure().run_capture(builder).stdout();
    lines
        .lines()
        .find_map(|l| l.split(':').nth(1)?.strip_prefix("browser-ui-test@"))
        .map(|v| v.to_owned())
}

fn get_browser_ui_test_version(builder: &Builder<'_>, npm: &Path) -> Option<String> {
    get_browser_ui_test_version_inner(builder, npm, false)
        .or_else(|| get_browser_ui_test_version_inner(builder, npm, true))
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustdocGUI {
    pub target: TargetSelection,
    pub compiler: Compiler,
}

impl Step for RustdocGUI {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        let run = run.suite_path("tests/rustdoc-gui");
        run.lazy_default_condition(Box::new(move || {
            builder.config.nodejs.is_some()
                && builder.doc_tests != DocTests::Only
                && builder
                    .config
                    .npm
                    .as_ref()
                    .map(|p| get_browser_ui_test_version(builder, p).is_some())
                    .unwrap_or(false)
        }))
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(RustdocGUI { target: run.target, compiler });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(compile::Std::new(self.compiler, self.target));

        let mut cmd = builder.tool_cmd(Tool::RustdocGUITest);

        let out_dir = builder.test_out(self.target).join("rustdoc-gui");
        build_stamp::clear_if_dirty(builder, &out_dir, &builder.rustdoc(self.compiler));

        if let Some(src) = builder.config.src.to_str() {
            cmd.arg("--rust-src").arg(src);
        }

        if let Some(out_dir) = out_dir.to_str() {
            cmd.arg("--out-dir").arg(out_dir);
        }

        if let Some(initial_cargo) = builder.config.initial_cargo.to_str() {
            cmd.arg("--initial-cargo").arg(initial_cargo);
        }

        cmd.arg("--jobs").arg(builder.jobs().to_string());

        cmd.env("RUSTDOC", builder.rustdoc(self.compiler))
            .env("RUSTC", builder.rustc(self.compiler));

        add_rustdoc_cargo_linker_args(&mut cmd, builder, self.compiler.host, LldThreads::No);

        for path in &builder.paths {
            if let Some(p) = helpers::is_valid_test_suite_arg(path, "tests/rustdoc-gui", builder) {
                if !p.ends_with(".goml") {
                    eprintln!("A non-goml file was given: `{}`", path.display());
                    panic!("Cannot run rustdoc-gui tests");
                }
                if let Some(name) = path.file_name().and_then(|f| f.to_str()) {
                    cmd.arg("--goml-file").arg(name);
                }
            }
        }

        for test_arg in builder.config.test_args() {
            cmd.arg("--test-arg").arg(test_arg);
        }

        if let Some(ref nodejs) = builder.config.nodejs {
            cmd.arg("--nodejs").arg(nodejs);
        }

        if let Some(ref npm) = builder.config.npm {
            cmd.arg("--npm").arg(npm);
        }

        let _time = helpers::timeit(builder);
        let _guard = builder.msg_sysroot_tool(
            Kind::Test,
            self.compiler.stage,
            "rustdoc-gui",
            self.compiler.host,
            self.target,
        );
        try_run_tests(builder, &mut cmd, true);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct RunMakeSupport {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RunMakeSupport {
    type Output = PathBuf;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn make_run(run: RunConfig<'_>) {
        let compiler = run.builder.compiler(run.builder.top_stage, run.build_triple());
        run.builder.ensure(RunMakeSupport { compiler, target: run.build_triple() });
    }

    /// Builds run-make-support and returns the path to the resulting rlib.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        builder.ensure(compile::Std::new(self.compiler, self.target));

        let cargo = tool::prepare_tool_cargo(
            builder,
            self.compiler,
            Mode::ToolStd,
            self.target,
            Kind::Build,
            "src/tools/run-make-support",
            SourceType::InTree,
            &[],
        );

        cargo.into_cmd().run(builder);

        let lib_name = "librun_make_support.rlib";
        let lib = builder.tools_dir(self.compiler).join(lib_name);

        let cargo_out = builder.cargo_out(self.compiler, Mode::ToolStd, self.target).join(lib_name);
        builder.copy_link(&cargo_out, &lib);
        lib
    }
}

/// Runs `cargo test` on the `src/tools/run-make-support` crate.
/// That crate is used by run-make tests.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateRunMakeSupport {
    host: TargetSelection,
}

impl Step for CrateRunMakeSupport {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/run-make-support")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CrateRunMakeSupport { host: run.target });
    }

    /// Runs `cargo test` for run-make-support.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let compiler = builder.compiler(0, host);

        let mut cargo = tool::prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolBootstrap,
            host,
            Kind::Test,
            "src/tools/run-make-support",
            SourceType::InTree,
            &[],
        );
        cargo.allow_features("test");
        run_cargo_test(
            cargo,
            &[],
            &[],
            "run-make-support",
            "run-make-support self test",
            host,
            builder,
        );
    }
}
