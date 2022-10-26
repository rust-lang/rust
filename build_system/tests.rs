use crate::build_system::rustc_info::get_cargo_path;

use super::build_sysroot;
use super::config;
use super::prepare::GitRepo;
use super::rustc_info::get_wrapper_file_name;
use super::utils::{
    hyperfine_command, spawn_and_wait, spawn_and_wait_with_input, CargoProject, Compiler,
};
use super::SysrootKind;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct TestCase {
    config: &'static str,
    func: &'static dyn Fn(&TestRunner),
}

impl TestCase {
    const fn new(config: &'static str, func: &'static dyn Fn(&TestRunner)) -> Self {
        Self { config, func }
    }
}

const NO_SYSROOT_SUITE: &[TestCase] = &[
    TestCase::new("build.mini_core", &|runner| {
        runner.run_rustc([
            "example/mini_core.rs",
            "--crate-name",
            "mini_core",
            "--crate-type",
            "lib,dylib",
            "--target",
            &runner.target_compiler.triple,
        ]);
    }),
    TestCase::new("build.example", &|runner| {
        runner.run_rustc([
            "example/example.rs",
            "--crate-type",
            "lib",
            "--target",
            &runner.target_compiler.triple,
        ]);
    }),
    TestCase::new("jit.mini_core_hello_world", &|runner| {
        let mut jit_cmd = runner.rustc_command([
            "-Zunstable-options",
            "-Cllvm-args=mode=jit",
            "-Cprefer-dynamic",
            "example/mini_core_hello_world.rs",
            "--cfg",
            "jit",
            "--target",
            &runner.target_compiler.triple,
        ]);
        jit_cmd.env("CG_CLIF_JIT_ARGS", "abc bcd");
        spawn_and_wait(jit_cmd);

        eprintln!("[JIT-lazy] mini_core_hello_world");
        let mut jit_cmd = runner.rustc_command([
            "-Zunstable-options",
            "-Cllvm-args=mode=jit-lazy",
            "-Cprefer-dynamic",
            "example/mini_core_hello_world.rs",
            "--cfg",
            "jit",
            "--target",
            &runner.target_compiler.triple,
        ]);
        jit_cmd.env("CG_CLIF_JIT_ARGS", "abc bcd");
        spawn_and_wait(jit_cmd);
    }),
    TestCase::new("aot.mini_core_hello_world", &|runner| {
        runner.run_rustc([
            "example/mini_core_hello_world.rs",
            "--crate-name",
            "mini_core_hello_world",
            "--crate-type",
            "bin",
            "-g",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("mini_core_hello_world", ["abc", "bcd"]);
    }),
];

const BASE_SYSROOT_SUITE: &[TestCase] = &[
    TestCase::new("aot.arbitrary_self_types_pointers_and_wrappers", &|runner| {
        runner.run_rustc([
            "example/arbitrary_self_types_pointers_and_wrappers.rs",
            "--crate-name",
            "arbitrary_self_types_pointers_and_wrappers",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("arbitrary_self_types_pointers_and_wrappers", []);
    }),
    TestCase::new("aot.issue_91827_extern_types", &|runner| {
        runner.run_rustc([
            "example/issue-91827-extern-types.rs",
            "--crate-name",
            "issue_91827_extern_types",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("issue_91827_extern_types", []);
    }),
    TestCase::new("build.alloc_system", &|runner| {
        runner.run_rustc([
            "example/alloc_system.rs",
            "--crate-type",
            "lib",
            "--target",
            &runner.target_compiler.triple,
        ]);
    }),
    TestCase::new("aot.alloc_example", &|runner| {
        runner.run_rustc([
            "example/alloc_example.rs",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("alloc_example", []);
    }),
    TestCase::new("jit.std_example", &|runner| {
        runner.run_rustc([
            "-Zunstable-options",
            "-Cllvm-args=mode=jit",
            "-Cprefer-dynamic",
            "example/std_example.rs",
            "--target",
            &runner.target_compiler.triple,
        ]);

        eprintln!("[JIT-lazy] std_example");
        runner.run_rustc([
            "-Zunstable-options",
            "-Cllvm-args=mode=jit-lazy",
            "-Cprefer-dynamic",
            "example/std_example.rs",
            "--target",
            &runner.target_compiler.triple,
        ]);
    }),
    TestCase::new("aot.std_example", &|runner| {
        runner.run_rustc([
            "example/std_example.rs",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("std_example", ["arg"]);
    }),
    TestCase::new("aot.dst_field_align", &|runner| {
        runner.run_rustc([
            "example/dst-field-align.rs",
            "--crate-name",
            "dst_field_align",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("dst_field_align", []);
    }),
    TestCase::new("aot.subslice-patterns-const-eval", &|runner| {
        runner.run_rustc([
            "example/subslice-patterns-const-eval.rs",
            "--crate-type",
            "bin",
            "-Cpanic=abort",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("subslice-patterns-const-eval", []);
    }),
    TestCase::new("aot.track-caller-attribute", &|runner| {
        runner.run_rustc([
            "example/track-caller-attribute.rs",
            "--crate-type",
            "bin",
            "-Cpanic=abort",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("track-caller-attribute", []);
    }),
    TestCase::new("aot.float-minmax-pass", &|runner| {
        runner.run_rustc([
            "example/float-minmax-pass.rs",
            "--crate-type",
            "bin",
            "-Cpanic=abort",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("float-minmax-pass", []);
    }),
    TestCase::new("aot.mod_bench", &|runner| {
        runner.run_rustc([
            "example/mod_bench.rs",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_compiler.triple,
        ]);
        runner.run_out_command("mod_bench", []);
    }),
];

pub(crate) static RAND_REPO: GitRepo =
    GitRepo::github("rust-random", "rand", "0f933f9c7176e53b2a3c7952ded484e1783f0bf1", "rand");

static RAND: CargoProject = CargoProject::git(&RAND_REPO, ".");

pub(crate) static REGEX_REPO: GitRepo =
    GitRepo::github("rust-lang", "regex", "341f207c1071f7290e3f228c710817c280c8dca1", "regex");

static REGEX: CargoProject = CargoProject::git(&REGEX_REPO, ".");

pub(crate) static PORTABLE_SIMD_REPO: GitRepo = GitRepo::github(
    "rust-lang",
    "portable-simd",
    "d5cd4a8112d958bd3a252327e0d069a6363249bd",
    "portable-simd",
);

static PORTABLE_SIMD: CargoProject = CargoProject::git(&PORTABLE_SIMD_REPO, ".");

pub(crate) static SIMPLE_RAYTRACER_REPO: GitRepo = GitRepo::github(
    "ebobby",
    "simple-raytracer",
    "804a7a21b9e673a482797aa289a18ed480e4d813",
    "<none>",
);

pub(crate) static SIMPLE_RAYTRACER: CargoProject = CargoProject::git(&SIMPLE_RAYTRACER_REPO, ".");

static LIBCORE_TESTS: CargoProject =
    CargoProject::local("build_sysroot/sysroot_src/library/core/tests");

const EXTENDED_SYSROOT_SUITE: &[TestCase] = &[
    TestCase::new("test.rust-random/rand", &|runner| {
        spawn_and_wait(RAND.clean(&runner.target_compiler.cargo));

        if runner.is_native {
            eprintln!("[TEST] rust-random/rand");
            let mut test_cmd = RAND.test(&runner.target_compiler);
            test_cmd.arg("--workspace");
            spawn_and_wait(test_cmd);
        } else {
            eprintln!("[AOT] rust-random/rand");
            let mut build_cmd = RAND.build(&runner.target_compiler);
            build_cmd.arg("--workspace").arg("--tests");
            spawn_and_wait(build_cmd);
        }
    }),
    TestCase::new("bench.simple-raytracer", &|runner| {
        let run_runs = env::var("RUN_RUNS").unwrap_or("10".to_string()).parse().unwrap();

        if runner.is_native {
            eprintln!("[BENCH COMPILE] ebobby/simple-raytracer");
            let cargo_clif = env::current_dir()
                .unwrap()
                .join("build")
                .join(get_wrapper_file_name("cargo-clif", "bin"));
            let source_dir = SIMPLE_RAYTRACER.source_dir();
            let manifest_path = SIMPLE_RAYTRACER.manifest_path();
            let target_dir = SIMPLE_RAYTRACER.target_dir();

            let clean_cmd = format!(
                "cargo clean --manifest-path {manifest_path} --target-dir {target_dir}",
                manifest_path = manifest_path.display(),
                target_dir = target_dir.display(),
            );
            let llvm_build_cmd = format!(
                "cargo build --manifest-path {manifest_path} --target-dir {target_dir}",
                manifest_path = manifest_path.display(),
                target_dir = target_dir.display(),
            );
            let clif_build_cmd = format!(
                "{cargo_clif} build --manifest-path {manifest_path} --target-dir {target_dir}",
                cargo_clif = cargo_clif.display(),
                manifest_path = manifest_path.display(),
                target_dir = target_dir.display(),
            );

            let bench_compile =
                hyperfine_command(1, run_runs, Some(&clean_cmd), &llvm_build_cmd, &clif_build_cmd);

            spawn_and_wait(bench_compile);

            eprintln!("[BENCH RUN] ebobby/simple-raytracer");
            fs::copy(target_dir.join("debug").join("main"), source_dir.join("raytracer_cg_clif"))
                .unwrap();

            let mut bench_run = hyperfine_command(
                0,
                run_runs,
                None,
                &source_dir.join("raytracer_cg_llvm").display().to_string(),
                &source_dir.join("raytracer_cg_clif").display().to_string(),
            );
            bench_run.current_dir(SIMPLE_RAYTRACER.source_dir());
            spawn_and_wait(bench_run);
        } else {
            spawn_and_wait(SIMPLE_RAYTRACER.clean(&runner.target_compiler.cargo));
            eprintln!("[BENCH COMPILE] ebobby/simple-raytracer (skipped)");
            eprintln!("[COMPILE] ebobby/simple-raytracer");
            spawn_and_wait(SIMPLE_RAYTRACER.build(&runner.target_compiler));
            eprintln!("[BENCH RUN] ebobby/simple-raytracer (skipped)");
        }
    }),
    TestCase::new("test.libcore", &|runner| {
        spawn_and_wait(LIBCORE_TESTS.clean(&runner.host_compiler.cargo));

        if runner.is_native {
            spawn_and_wait(LIBCORE_TESTS.test(&runner.target_compiler));
        } else {
            eprintln!("Cross-Compiling: Not running tests");
            let mut build_cmd = LIBCORE_TESTS.build(&runner.target_compiler);
            build_cmd.arg("--tests");
            spawn_and_wait(build_cmd);
        }
    }),
    TestCase::new("test.regex-shootout-regex-dna", &|runner| {
        spawn_and_wait(REGEX.clean(&runner.target_compiler.cargo));

        // newer aho_corasick versions throw a deprecation warning
        let lint_rust_flags = format!("{} --cap-lints warn", runner.target_compiler.rustflags);

        let mut build_cmd = REGEX.build(&runner.target_compiler);
        build_cmd.arg("--example").arg("shootout-regex-dna");
        build_cmd.env("RUSTFLAGS", lint_rust_flags.clone());
        spawn_and_wait(build_cmd);

        if runner.is_native {
            let mut run_cmd = REGEX.run(&runner.target_compiler);
            run_cmd.arg("--example").arg("shootout-regex-dna");
            run_cmd.env("RUSTFLAGS", lint_rust_flags);

            let input =
                fs::read_to_string(REGEX.source_dir().join("examples").join("regexdna-input.txt"))
                    .unwrap();
            let expected_path = REGEX.source_dir().join("examples").join("regexdna-output.txt");
            let expected = fs::read_to_string(&expected_path).unwrap();

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
                let res_path = REGEX.source_dir().join("res.txt");
                fs::write(&res_path, &output).unwrap();

                if cfg!(windows) {
                    println!("Output files don't match!");
                    println!("Expected Output:\n{}", expected);
                    println!("Actual Output:\n{}", output);
                } else {
                    let mut diff = Command::new("diff");
                    diff.arg("-u");
                    diff.arg(res_path);
                    diff.arg(expected_path);
                    spawn_and_wait(diff);
                }

                std::process::exit(1);
            }
        }
    }),
    TestCase::new("test.regex", &|runner| {
        spawn_and_wait(REGEX.clean(&runner.host_compiler.cargo));

        // newer aho_corasick versions throw a deprecation warning
        let lint_rust_flags = format!("{} --cap-lints warn", runner.target_compiler.rustflags);

        if runner.is_native {
            let mut run_cmd = REGEX.test(&runner.target_compiler);
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
            let mut build_cmd = REGEX.build(&runner.target_compiler);
            build_cmd.arg("--tests");
            build_cmd.env("RUSTFLAGS", lint_rust_flags.clone());
            spawn_and_wait(build_cmd);
        }
    }),
    TestCase::new("test.portable-simd", &|runner| {
        spawn_and_wait(PORTABLE_SIMD.clean(&runner.host_compiler.cargo));

        let mut build_cmd = PORTABLE_SIMD.build(&runner.target_compiler);
        build_cmd.arg("--all-targets");
        spawn_and_wait(build_cmd);

        if runner.is_native {
            let mut test_cmd = PORTABLE_SIMD.test(&runner.target_compiler);
            test_cmd.arg("-q");
            spawn_and_wait(test_cmd);
        }
    }),
];

pub(crate) fn run_tests(
    channel: &str,
    sysroot_kind: SysrootKind,
    target_dir: &Path,
    cg_clif_dylib: &Path,
    host_triple: &str,
    target_triple: &str,
) {
    let runner = TestRunner::new(host_triple.to_string(), target_triple.to_string());

    if config::get_bool("testsuite.no_sysroot") {
        build_sysroot::build_sysroot(
            channel,
            SysrootKind::None,
            &target_dir,
            cg_clif_dylib,
            &host_triple,
            &target_triple,
        );

        let _ = fs::remove_dir_all(Path::new("target").join("out"));
        runner.run_testsuite(NO_SYSROOT_SUITE);
    } else {
        eprintln!("[SKIP] no_sysroot tests");
    }

    let run_base_sysroot = config::get_bool("testsuite.base_sysroot");
    let run_extended_sysroot = config::get_bool("testsuite.extended_sysroot");

    if run_base_sysroot || run_extended_sysroot {
        build_sysroot::build_sysroot(
            channel,
            sysroot_kind,
            &target_dir,
            cg_clif_dylib,
            &host_triple,
            &target_triple,
        );
    }

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

struct TestRunner {
    out_dir: PathBuf,
    is_native: bool,
    jit_supported: bool,
    host_compiler: Compiler,
    target_compiler: Compiler,
}

impl TestRunner {
    pub fn new(host_triple: String, target_triple: String) -> Self {
        let root_dir = env::current_dir().unwrap();

        let mut out_dir = root_dir.clone();
        out_dir.push("target");
        out_dir.push("out");

        let is_native = host_triple == target_triple;
        let jit_supported =
            target_triple.contains("x86_64") && is_native && !host_triple.contains("windows");

        let mut rustc_clif = root_dir.clone();
        rustc_clif.push("build");
        rustc_clif.push(get_wrapper_file_name("rustc-clif", "bin"));

        let mut rustdoc_clif = root_dir.clone();
        rustdoc_clif.push("build");
        rustdoc_clif.push(get_wrapper_file_name("rustdoc-clif", "bin"));

        let mut rustflags = env::var("RUSTFLAGS").ok().unwrap_or("".to_string());
        let mut runner = vec![];

        if !is_native {
            match target_triple.as_str() {
                "aarch64-unknown-linux-gnu" => {
                    // We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
                    rustflags = format!("-Clinker=aarch64-linux-gnu-gcc{}", rustflags);
                    runner = vec![
                        "qemu-aarch64".to_owned(),
                        "-L".to_owned(),
                        "/usr/aarch64-linux-gnu".to_owned(),
                    ];
                }
                "x86_64-pc-windows-gnu" => {
                    // We are cross-compiling for Windows. Run tests in wine.
                    runner = vec!["wine".to_owned()];
                }
                _ => {
                    println!("Unknown non-native platform");
                }
            }
        }

        // FIXME fix `#[linkage = "extern_weak"]` without this
        if target_triple.contains("darwin") {
            rustflags = format!("{} -Clink-arg=-undefined -Clink-arg=dynamic_lookup", rustflags);
        }

        let host_compiler = Compiler {
            cargo: get_cargo_path(),
            rustc: rustc_clif.clone(),
            rustdoc: rustdoc_clif.clone(),
            rustflags: String::new(),
            rustdocflags: String::new(),
            triple: host_triple,
            runner: vec![],
        };

        let target_compiler = Compiler {
            cargo: get_cargo_path(),
            rustc: rustc_clif.clone(),
            rustdoc: rustdoc_clif.clone(),
            rustflags: rustflags.clone(),
            rustdocflags: rustflags,
            triple: target_triple,
            runner,
        };

        Self { out_dir, is_native, jit_supported, host_compiler, target_compiler }
    }

    pub fn run_testsuite(&self, tests: &[TestCase]) {
        for &TestCase { config, func } in tests {
            let (tag, testname) = config.split_once('.').unwrap();
            let tag = tag.to_uppercase();
            let is_jit_test = tag == "JIT";

            if !config::get_bool(config) || (is_jit_test && !self.jit_supported) {
                eprintln!("[{tag}] {testname} (skipped)");
                continue;
            } else {
                eprintln!("[{tag}] {testname}");
            }

            func(self);
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
        cmd.arg(format!("crate={}", self.out_dir.display()));
        cmd.arg("--out-dir");
        cmd.arg(format!("{}", self.out_dir.display()));
        cmd.arg("-Cdebuginfo=2");
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

    fn run_out_command<'a, I>(&self, name: &str, args: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut full_cmd = vec![];

        // Prepend the RUN_WRAPPER's
        if !self.target_compiler.runner.is_empty() {
            full_cmd.extend(self.target_compiler.runner.iter().cloned());
        }

        full_cmd.push({
            let mut out_path = self.out_dir.clone();
            out_path.push(name);
            out_path.to_str().unwrap().to_string()
        });

        for arg in args.into_iter() {
            full_cmd.push(arg.to_string());
        }

        let mut cmd_iter = full_cmd.into_iter();
        let first = cmd_iter.next().unwrap();

        let mut cmd = Command::new(first);
        cmd.args(cmd_iter);

        spawn_and_wait(cmd);
    }
}
