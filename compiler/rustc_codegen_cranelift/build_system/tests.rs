use super::build_sysroot::{self, SYSROOT_SRC};
use super::config;
use super::path::{Dirs, RelPath};
use super::prepare::GitRepo;
use super::rustc_info::get_host_triple;
use super::utils::{spawn_and_wait, spawn_and_wait_with_input, CargoProject, Compiler};
use super::SysrootKind;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::process::Command;

static BUILD_EXAMPLE_OUT_DIR: RelPath = RelPath::BUILD.join("example");

struct TestCase {
    config: &'static str,
    cmd: TestCaseCmd,
}

enum TestCaseCmd {
    Custom { func: &'static dyn Fn(&TestRunner) },
    BuildLib { source: &'static str, crate_types: &'static str },
    BuildBinAndRun { source: &'static str, args: &'static [&'static str] },
    JitBin { source: &'static str, args: &'static str },
}

impl TestCase {
    // FIXME reduce usage of custom test case commands
    const fn custom(config: &'static str, func: &'static dyn Fn(&TestRunner)) -> Self {
        Self { config, cmd: TestCaseCmd::Custom { func } }
    }

    const fn build_lib(
        config: &'static str,
        source: &'static str,
        crate_types: &'static str,
    ) -> Self {
        Self { config, cmd: TestCaseCmd::BuildLib { source, crate_types } }
    }

    const fn build_bin_and_run(
        config: &'static str,
        source: &'static str,
        args: &'static [&'static str],
    ) -> Self {
        Self { config, cmd: TestCaseCmd::BuildBinAndRun { source, args } }
    }

    const fn jit_bin(config: &'static str, source: &'static str, args: &'static str) -> Self {
        Self { config, cmd: TestCaseCmd::JitBin { source, args } }
    }
}

const NO_SYSROOT_SUITE: &[TestCase] = &[
    TestCase::build_lib("build.mini_core", "example/mini_core.rs", "lib,dylib"),
    TestCase::build_lib("build.example", "example/example.rs", "lib"),
    TestCase::jit_bin("jit.mini_core_hello_world", "example/mini_core_hello_world.rs", "abc bcd"),
    TestCase::build_bin_and_run(
        "aot.mini_core_hello_world",
        "example/mini_core_hello_world.rs",
        &["abc", "bcd"],
    ),
];

const BASE_SYSROOT_SUITE: &[TestCase] = &[
    TestCase::build_bin_and_run(
        "aot.arbitrary_self_types_pointers_and_wrappers",
        "example/arbitrary_self_types_pointers_and_wrappers.rs",
        &[],
    ),
    TestCase::build_bin_and_run(
        "aot.issue_91827_extern_types",
        "example/issue-91827-extern-types.rs",
        &[],
    ),
    TestCase::build_lib("build.alloc_system", "example/alloc_system.rs", "lib"),
    TestCase::build_bin_and_run("aot.alloc_example", "example/alloc_example.rs", &[]),
    TestCase::jit_bin("jit.std_example", "example/std_example.rs", ""),
    TestCase::build_bin_and_run("aot.std_example", "example/std_example.rs", &["arg"]),
    TestCase::build_bin_and_run("aot.dst_field_align", "example/dst-field-align.rs", &[]),
    TestCase::build_bin_and_run(
        "aot.subslice-patterns-const-eval",
        "example/subslice-patterns-const-eval.rs",
        &[],
    ),
    TestCase::build_bin_and_run(
        "aot.track-caller-attribute",
        "example/track-caller-attribute.rs",
        &[],
    ),
    TestCase::build_bin_and_run("aot.float-minmax-pass", "example/float-minmax-pass.rs", &[]),
    TestCase::build_bin_and_run("aot.mod_bench", "example/mod_bench.rs", &[]),
    TestCase::build_bin_and_run("aot.issue-72793", "example/issue-72793.rs", &[]),
];

pub(crate) static RAND_REPO: GitRepo =
    GitRepo::github("rust-random", "rand", "0f933f9c7176e53b2a3c7952ded484e1783f0bf1", "rand");

pub(crate) static RAND: CargoProject = CargoProject::new(&RAND_REPO.source_dir(), "rand");

pub(crate) static REGEX_REPO: GitRepo =
    GitRepo::github("rust-lang", "regex", "341f207c1071f7290e3f228c710817c280c8dca1", "regex");

pub(crate) static REGEX: CargoProject = CargoProject::new(&REGEX_REPO.source_dir(), "regex");

pub(crate) static PORTABLE_SIMD_REPO: GitRepo = GitRepo::github(
    "rust-lang",
    "portable-simd",
    "582239ac3b32007613df04d7ffa78dc30f4c5645",
    "portable-simd",
);

pub(crate) static PORTABLE_SIMD: CargoProject =
    CargoProject::new(&PORTABLE_SIMD_REPO.source_dir(), "portable_simd");

pub(crate) static LIBCORE_TESTS: CargoProject =
    CargoProject::new(&SYSROOT_SRC.join("library/core/tests"), "core_tests");

const EXTENDED_SYSROOT_SUITE: &[TestCase] = &[
    TestCase::custom("test.rust-random/rand", &|runner| {
        RAND.clean(&runner.dirs);

        if runner.is_native {
            eprintln!("[TEST] rust-random/rand");
            let mut test_cmd = RAND.test(&runner.target_compiler, &runner.dirs);
            test_cmd.arg("--workspace");
            spawn_and_wait(test_cmd);
        } else {
            eprintln!("[AOT] rust-random/rand");
            let mut build_cmd = RAND.build(&runner.target_compiler, &runner.dirs);
            build_cmd.arg("--workspace").arg("--tests");
            spawn_and_wait(build_cmd);
        }
    }),
    TestCase::custom("test.libcore", &|runner| {
        LIBCORE_TESTS.clean(&runner.dirs);

        if runner.is_native {
            spawn_and_wait(LIBCORE_TESTS.test(&runner.target_compiler, &runner.dirs));
        } else {
            eprintln!("Cross-Compiling: Not running tests");
            let mut build_cmd = LIBCORE_TESTS.build(&runner.target_compiler, &runner.dirs);
            build_cmd.arg("--tests");
            spawn_and_wait(build_cmd);
        }
    }),
    TestCase::custom("test.regex-shootout-regex-dna", &|runner| {
        REGEX.clean(&runner.dirs);

        // newer aho_corasick versions throw a deprecation warning
        let lint_rust_flags = format!("{} --cap-lints warn", runner.target_compiler.rustflags);

        let mut build_cmd = REGEX.build(&runner.target_compiler, &runner.dirs);
        build_cmd.arg("--example").arg("shootout-regex-dna");
        build_cmd.env("RUSTFLAGS", lint_rust_flags.clone());
        spawn_and_wait(build_cmd);

        if runner.is_native {
            let mut run_cmd = REGEX.run(&runner.target_compiler, &runner.dirs);
            run_cmd.arg("--example").arg("shootout-regex-dna");
            run_cmd.env("RUSTFLAGS", lint_rust_flags);

            let input = fs::read_to_string(
                REGEX.source_dir(&runner.dirs).join("examples").join("regexdna-input.txt"),
            )
            .unwrap();
            let expected = fs::read_to_string(
                REGEX.source_dir(&runner.dirs).join("examples").join("regexdna-output.txt"),
            )
            .unwrap();

            let output = spawn_and_wait_with_input(run_cmd, input);
            // Make sure `[codegen mono items] start` doesn't poison the diff
            let output = output
                .lines()
                .filter(|line| !line.contains("codegen mono items"))
                .chain(Some("")) // This just adds the trailing newline
                .collect::<Vec<&str>>()
                .join("\r\n");

            let output_matches = expected.lines().eq(output.lines());
            if !output_matches {
                println!("Output files don't match!");
                println!("Expected Output:\n{}", expected);
                println!("Actual Output:\n{}", output);

                std::process::exit(1);
            }
        }
    }),
    TestCase::custom("test.regex", &|runner| {
        REGEX.clean(&runner.dirs);

        // newer aho_corasick versions throw a deprecation warning
        let lint_rust_flags = format!("{} --cap-lints warn", runner.target_compiler.rustflags);

        if runner.is_native {
            let mut run_cmd = REGEX.test(&runner.target_compiler, &runner.dirs);
            run_cmd.args([
                "--tests",
                "--",
                "--exclude-should-panic",
                "--test-threads",
                "1",
                "-Zunstable-options",
                "-q",
            ]);
            run_cmd.env("RUSTFLAGS", lint_rust_flags);
            spawn_and_wait(run_cmd);
        } else {
            eprintln!("Cross-Compiling: Not running tests");
            let mut build_cmd = REGEX.build(&runner.target_compiler, &runner.dirs);
            build_cmd.arg("--tests");
            build_cmd.env("RUSTFLAGS", lint_rust_flags.clone());
            spawn_and_wait(build_cmd);
        }
    }),
    TestCase::custom("test.portable-simd", &|runner| {
        PORTABLE_SIMD.clean(&runner.dirs);

        let mut build_cmd = PORTABLE_SIMD.build(&runner.target_compiler, &runner.dirs);
        build_cmd.arg("--all-targets");
        spawn_and_wait(build_cmd);

        if runner.is_native {
            let mut test_cmd = PORTABLE_SIMD.test(&runner.target_compiler, &runner.dirs);
            test_cmd.arg("-q");
            spawn_and_wait(test_cmd);
        }
    }),
];

pub(crate) fn run_tests(
    dirs: &Dirs,
    channel: &str,
    sysroot_kind: SysrootKind,
    cg_clif_dylib: &Path,
    bootstrap_host_compiler: &Compiler,
    target_triple: String,
) {
    if config::get_bool("testsuite.no_sysroot") {
        let target_compiler = build_sysroot::build_sysroot(
            dirs,
            channel,
            SysrootKind::None,
            cg_clif_dylib,
            bootstrap_host_compiler,
            target_triple.clone(),
        );

        let runner =
            TestRunner::new(dirs.clone(), target_compiler, get_host_triple() == target_triple);

        BUILD_EXAMPLE_OUT_DIR.ensure_fresh(dirs);
        runner.run_testsuite(NO_SYSROOT_SUITE);
    } else {
        eprintln!("[SKIP] no_sysroot tests");
    }

    let run_base_sysroot = config::get_bool("testsuite.base_sysroot");
    let run_extended_sysroot = config::get_bool("testsuite.extended_sysroot");

    if run_base_sysroot || run_extended_sysroot {
        let target_compiler = build_sysroot::build_sysroot(
            dirs,
            channel,
            sysroot_kind,
            cg_clif_dylib,
            bootstrap_host_compiler,
            target_triple.clone(),
        );

        let runner =
            TestRunner::new(dirs.clone(), target_compiler, get_host_triple() == target_triple);

        if run_base_sysroot {
            runner.run_testsuite(BASE_SYSROOT_SUITE);
        } else {
            eprintln!("[SKIP] base_sysroot tests");
        }

        if run_extended_sysroot {
            runner.run_testsuite(EXTENDED_SYSROOT_SUITE);
        } else {
            eprintln!("[SKIP] extended_sysroot tests");
        }
    }
}

struct TestRunner {
    is_native: bool,
    jit_supported: bool,
    dirs: Dirs,
    target_compiler: Compiler,
}

impl TestRunner {
    pub fn new(dirs: Dirs, mut target_compiler: Compiler, is_native: bool) -> Self {
        if let Ok(rustflags) = env::var("RUSTFLAGS") {
            target_compiler.rustflags.push(' ');
            target_compiler.rustflags.push_str(&rustflags);
        }
        if let Ok(rustdocflags) = env::var("RUSTDOCFLAGS") {
            target_compiler.rustdocflags.push(' ');
            target_compiler.rustdocflags.push_str(&rustdocflags);
        }

        // FIXME fix `#[linkage = "extern_weak"]` without this
        if target_compiler.triple.contains("darwin") {
            target_compiler.rustflags.push_str(" -Clink-arg=-undefined -Clink-arg=dynamic_lookup");
        }

        let jit_supported = is_native
            && target_compiler.triple.contains("x86_64")
            && !target_compiler.triple.contains("windows");

        Self { is_native, jit_supported, dirs, target_compiler }
    }

    pub fn run_testsuite(&self, tests: &[TestCase]) {
        for TestCase { config, cmd } in tests {
            let (tag, testname) = config.split_once('.').unwrap();
            let tag = tag.to_uppercase();
            let is_jit_test = tag == "JIT";

            if !config::get_bool(config) || (is_jit_test && !self.jit_supported) {
                eprintln!("[{tag}] {testname} (skipped)");
                continue;
            } else {
                eprintln!("[{tag}] {testname}");
            }

            match *cmd {
                TestCaseCmd::Custom { func } => func(self),
                TestCaseCmd::BuildLib { source, crate_types } => {
                    self.run_rustc([source, "--crate-type", crate_types]);
                }
                TestCaseCmd::BuildBinAndRun { source, args } => {
                    self.run_rustc([source]);
                    self.run_out_command(
                        source.split('/').last().unwrap().split('.').next().unwrap(),
                        args,
                    );
                }
                TestCaseCmd::JitBin { source, args } => {
                    let mut jit_cmd = self.rustc_command([
                        "-Zunstable-options",
                        "-Cllvm-args=mode=jit",
                        "-Cprefer-dynamic",
                        source,
                        "--cfg",
                        "jit",
                    ]);
                    if !args.is_empty() {
                        jit_cmd.env("CG_CLIF_JIT_ARGS", args);
                    }
                    spawn_and_wait(jit_cmd);

                    eprintln!("[JIT-lazy] {testname}");
                    let mut jit_cmd = self.rustc_command([
                        "-Zunstable-options",
                        "-Cllvm-args=mode=jit-lazy",
                        "-Cprefer-dynamic",
                        source,
                        "--cfg",
                        "jit",
                    ]);
                    if !args.is_empty() {
                        jit_cmd.env("CG_CLIF_JIT_ARGS", args);
                    }
                    spawn_and_wait(jit_cmd);
                }
            }
        }
    }

    #[must_use]
    fn rustc_command<I, S>(&self, args: I) -> Command
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut cmd = Command::new(&self.target_compiler.rustc);
        cmd.args(self.target_compiler.rustflags.split_whitespace());
        cmd.arg("-L");
        cmd.arg(format!("crate={}", BUILD_EXAMPLE_OUT_DIR.to_path(&self.dirs).display()));
        cmd.arg("--out-dir");
        cmd.arg(format!("{}", BUILD_EXAMPLE_OUT_DIR.to_path(&self.dirs).display()));
        cmd.arg("-Cdebuginfo=2");
        cmd.arg("--target");
        cmd.arg(&self.target_compiler.triple);
        cmd.arg("-Cpanic=abort");
        cmd.args(args);
        cmd
    }

    fn run_rustc<I, S>(&self, args: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        spawn_and_wait(self.rustc_command(args));
    }

    fn run_out_command<'a>(&self, name: &str, args: &[&str]) {
        let mut full_cmd = vec![];

        // Prepend the RUN_WRAPPER's
        if !self.target_compiler.runner.is_empty() {
            full_cmd.extend(self.target_compiler.runner.iter().cloned());
        }

        full_cmd.push(
            BUILD_EXAMPLE_OUT_DIR.to_path(&self.dirs).join(name).to_str().unwrap().to_string(),
        );

        for arg in args {
            full_cmd.push(arg.to_string());
        }

        let mut cmd_iter = full_cmd.into_iter();
        let first = cmd_iter.next().unwrap();

        let mut cmd = Command::new(first);
        cmd.args(cmd_iter);

        spawn_and_wait(cmd);
    }
}
