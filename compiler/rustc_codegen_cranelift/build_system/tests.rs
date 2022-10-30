use super::build_sysroot;
use super::config;
use super::rustc_info::get_wrapper_file_name;
use super::utils::{spawn_and_wait, spawn_and_wait_with_input};
use build_system::SysrootKind;
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
            &runner.target_triple,
        ]);
    }),
    TestCase::new("build.example", &|runner| {
        runner.run_rustc([
            "example/example.rs",
            "--crate-type",
            "lib",
            "--target",
            &runner.target_triple,
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
            &runner.host_triple,
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
            &runner.host_triple,
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
            &runner.target_triple,
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
            &runner.target_triple,
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
            &runner.target_triple,
        ]);
        runner.run_out_command("issue_91827_extern_types", []);
    }),
    TestCase::new("build.alloc_system", &|runner| {
        runner.run_rustc([
            "example/alloc_system.rs",
            "--crate-type",
            "lib",
            "--target",
            &runner.target_triple,
        ]);
    }),
    TestCase::new("aot.alloc_example", &|runner| {
        runner.run_rustc([
            "example/alloc_example.rs",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_triple,
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
            &runner.host_triple,
        ]);

        eprintln!("[JIT-lazy] std_example");
        runner.run_rustc([
            "-Zunstable-options",
            "-Cllvm-args=mode=jit-lazy",
            "-Cprefer-dynamic",
            "example/std_example.rs",
            "--target",
            &runner.host_triple,
        ]);
    }),
    TestCase::new("aot.std_example", &|runner| {
        runner.run_rustc([
            "example/std_example.rs",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_triple,
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
            &runner.target_triple,
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
            &runner.target_triple,
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
            &runner.target_triple,
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
            &runner.target_triple,
        ]);
        runner.run_out_command("float-minmax-pass", []);
    }),
    TestCase::new("aot.mod_bench", &|runner| {
        runner.run_rustc([
            "example/mod_bench.rs",
            "--crate-type",
            "bin",
            "--target",
            &runner.target_triple,
        ]);
        runner.run_out_command("mod_bench", []);
    }),
];

const EXTENDED_SYSROOT_SUITE: &[TestCase] = &[
    TestCase::new("test.rust-random/rand", &|runner| {
        runner.in_dir(["rand"], |runner| {
            runner.run_cargo(["clean"]);

            if runner.host_triple == runner.target_triple {
                eprintln!("[TEST] rust-random/rand");
                runner.run_cargo(["test", "--workspace"]);
            } else {
                eprintln!("[AOT] rust-random/rand");
                runner.run_cargo([
                    "build",
                    "--workspace",
                    "--target",
                    &runner.target_triple,
                    "--tests",
                ]);
            }
        });
    }),
    TestCase::new("bench.simple-raytracer", &|runner| {
        runner.in_dir(["simple-raytracer"], |runner| {
            let run_runs = env::var("RUN_RUNS").unwrap_or("10".to_string());

            if runner.host_triple == runner.target_triple {
                eprintln!("[BENCH COMPILE] ebobby/simple-raytracer");
                let mut bench_compile = Command::new("hyperfine");
                bench_compile.arg("--runs");
                bench_compile.arg(&run_runs);
                bench_compile.arg("--warmup");
                bench_compile.arg("1");
                bench_compile.arg("--prepare");
                bench_compile.arg(format!("{:?}", runner.cargo_command(["clean"])));

                if cfg!(windows) {
                    bench_compile.arg("cmd /C \"set RUSTFLAGS= && cargo build\"");
                } else {
                    bench_compile.arg("RUSTFLAGS='' cargo build");
                }

                bench_compile.arg(format!("{:?}", runner.cargo_command(["build"])));
                spawn_and_wait(bench_compile);

                eprintln!("[BENCH RUN] ebobby/simple-raytracer");
                fs::copy(PathBuf::from("./target/debug/main"), PathBuf::from("raytracer_cg_clif"))
                    .unwrap();

                let mut bench_run = Command::new("hyperfine");
                bench_run.arg("--runs");
                bench_run.arg(&run_runs);
                bench_run.arg(PathBuf::from("./raytracer_cg_llvm"));
                bench_run.arg(PathBuf::from("./raytracer_cg_clif"));
                spawn_and_wait(bench_run);
            } else {
                runner.run_cargo(["clean"]);
                eprintln!("[BENCH COMPILE] ebobby/simple-raytracer (skipped)");
                eprintln!("[COMPILE] ebobby/simple-raytracer");
                runner.run_cargo(["build", "--target", &runner.target_triple]);
                eprintln!("[BENCH RUN] ebobby/simple-raytracer (skipped)");
            }
        });
    }),
    TestCase::new("test.libcore", &|runner| {
        runner.in_dir(["build_sysroot", "sysroot_src", "library", "core", "tests"], |runner| {
            runner.run_cargo(["clean"]);

            if runner.host_triple == runner.target_triple {
                runner.run_cargo(["test"]);
            } else {
                eprintln!("Cross-Compiling: Not running tests");
                runner.run_cargo(["build", "--target", &runner.target_triple, "--tests"]);
            }
        });
    }),
    TestCase::new("test.regex-shootout-regex-dna", &|runner| {
        runner.in_dir(["regex"], |runner| {
            runner.run_cargo(["clean"]);

            // newer aho_corasick versions throw a deprecation warning
            let lint_rust_flags = format!("{} --cap-lints warn", runner.rust_flags);

            let mut build_cmd = runner.cargo_command([
                "build",
                "--example",
                "shootout-regex-dna",
                "--target",
                &runner.target_triple,
            ]);
            build_cmd.env("RUSTFLAGS", lint_rust_flags.clone());
            spawn_and_wait(build_cmd);

            if runner.host_triple == runner.target_triple {
                let mut run_cmd = runner.cargo_command([
                    "run",
                    "--example",
                    "shootout-regex-dna",
                    "--target",
                    &runner.target_triple,
                ]);
                run_cmd.env("RUSTFLAGS", lint_rust_flags);

                let input =
                    fs::read_to_string(PathBuf::from("examples/regexdna-input.txt")).unwrap();
                let expected_path = PathBuf::from("examples/regexdna-output.txt");
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
                    let res_path = PathBuf::from("res.txt");
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
        });
    }),
    TestCase::new("test.regex", &|runner| {
        runner.in_dir(["regex"], |runner| {
            runner.run_cargo(["clean"]);

            // newer aho_corasick versions throw a deprecation warning
            let lint_rust_flags = format!("{} --cap-lints warn", runner.rust_flags);

            if runner.host_triple == runner.target_triple {
                let mut run_cmd = runner.cargo_command([
                    "test",
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
                let mut build_cmd =
                    runner.cargo_command(["build", "--tests", "--target", &runner.target_triple]);
                build_cmd.env("RUSTFLAGS", lint_rust_flags.clone());
                spawn_and_wait(build_cmd);
            }
        });
    }),
    TestCase::new("test.portable-simd", &|runner| {
        runner.in_dir(["portable-simd"], |runner| {
            runner.run_cargo(["clean"]);
            runner.run_cargo(["build", "--all-targets", "--target", &runner.target_triple]);

            if runner.host_triple == runner.target_triple {
                runner.run_cargo(["test", "-q"]);
            }
        });
    }),
];

pub(crate) fn run_tests(
    channel: &str,
    sysroot_kind: SysrootKind,
    target_dir: &Path,
    cg_clif_build_dir: &Path,
    host_triple: &str,
    target_triple: &str,
) {
    let runner = TestRunner::new(host_triple.to_string(), target_triple.to_string());

    if config::get_bool("testsuite.no_sysroot") {
        build_sysroot::build_sysroot(
            channel,
            SysrootKind::None,
            &target_dir,
            cg_clif_build_dir,
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
            cg_clif_build_dir,
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
    root_dir: PathBuf,
    out_dir: PathBuf,
    jit_supported: bool,
    rust_flags: String,
    run_wrapper: Vec<String>,
    host_triple: String,
    target_triple: String,
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

        let mut rust_flags = env::var("RUSTFLAGS").ok().unwrap_or("".to_string());
        let mut run_wrapper = Vec::new();

        if !is_native {
            match target_triple.as_str() {
                "aarch64-unknown-linux-gnu" => {
                    // We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
                    rust_flags = format!("-Clinker=aarch64-linux-gnu-gcc{}", rust_flags);
                    run_wrapper = vec!["qemu-aarch64", "-L", "/usr/aarch64-linux-gnu"];
                }
                "x86_64-pc-windows-gnu" => {
                    // We are cross-compiling for Windows. Run tests in wine.
                    run_wrapper = vec!["wine"];
                }
                _ => {
                    println!("Unknown non-native platform");
                }
            }
        }

        // FIXME fix `#[linkage = "extern_weak"]` without this
        if host_triple.contains("darwin") {
            rust_flags = format!("{} -Clink-arg=-undefined -Clink-arg=dynamic_lookup", rust_flags);
        }

        Self {
            root_dir,
            out_dir,
            jit_supported,
            rust_flags,
            run_wrapper: run_wrapper.iter().map(|s| s.to_string()).collect(),
            host_triple,
            target_triple,
        }
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

    fn in_dir<'a, I, F>(&self, dir: I, callback: F)
    where
        I: IntoIterator<Item = &'a str>,
        F: FnOnce(&TestRunner),
    {
        let current = env::current_dir().unwrap();
        let mut new = current.clone();
        for d in dir {
            new.push(d);
        }

        env::set_current_dir(new).unwrap();
        callback(self);
        env::set_current_dir(current).unwrap();
    }

    fn rustc_command<I, S>(&self, args: I) -> Command
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut rustc_clif = self.root_dir.clone();
        rustc_clif.push("build");
        rustc_clif.push(get_wrapper_file_name("rustc-clif", "bin"));

        let mut cmd = Command::new(rustc_clif);
        cmd.args(self.rust_flags.split_whitespace());
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
        if !self.run_wrapper.is_empty() {
            full_cmd.extend(self.run_wrapper.iter().cloned());
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

    fn cargo_command<I, S>(&self, args: I) -> Command
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut cargo_clif = self.root_dir.clone();
        cargo_clif.push("build");
        cargo_clif.push(get_wrapper_file_name("cargo-clif", "bin"));

        let mut cmd = Command::new(cargo_clif);
        cmd.args(args);
        cmd.env("RUSTFLAGS", &self.rust_flags);
        cmd
    }

    fn run_cargo<'a, I>(&self, args: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        spawn_and_wait(self.cargo_command(args));
    }
}
