#![allow(clippy::uninlined_format_args)]

use std::env::current_dir;
use std::path::{Path, PathBuf};
use std::process::Command;

use lang_tester::LangTester;
use tempfile::TempDir;

fn compile_and_run_cmds(
    compiler_args: Vec<String>,
    test_target: &Option<String>,
    exe: &Path,
    test_mode: TestMode,
) -> Vec<(&'static str, Command)> {
    let mut compiler = Command::new("rustc");
    compiler.args(compiler_args);

    // Test command 2: run `tempdir/x`.
    if test_target.is_some() {
        let mut env_path = std::env::var("PATH").unwrap_or_default();
        // TODO(antoyo): find a better way to add the PATH necessary locally.
        env_path = format!("/opt/m68k-unknown-linux-gnu/bin:{}", env_path);
        compiler.env("PATH", env_path);

        let mut commands = vec![("Compiler", compiler)];
        if test_mode.should_run() {
            let vm_parent_dir = std::env::var("CG_GCC_VM_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| std::env::current_dir().unwrap());
            let vm_dir = "vm";
            let exe_filename = exe.file_name().unwrap();
            let vm_home_dir = vm_parent_dir.join(vm_dir).join("home");
            let vm_exe_path = vm_home_dir.join(exe_filename);
            // FIXME(antoyo): panicking here makes the test pass.
            let inside_vm_exe_path = PathBuf::from("/home").join(exe_filename);

            let mut copy = Command::new("sudo");
            copy.arg("cp");
            copy.args([exe, &vm_exe_path]);

            let mut runtime = Command::new("sudo");
            runtime.args(["chroot", vm_dir, "qemu-m68k-static"]);
            runtime.arg(inside_vm_exe_path);
            runtime.current_dir(vm_parent_dir);

            commands.push(("Copy", copy));
            commands.push(("Run-time", runtime));
        }
        commands
    } else {
        let mut commands = vec![("Compiler", compiler)];
        if test_mode.should_run() {
            let runtime = Command::new(exe);
            commands.push(("Run-time", runtime));
        }
        commands
    }
}

#[derive(Clone, Copy)]
enum BuildMode {
    Debug,
    Release,
}

impl BuildMode {
    fn is_debug(self) -> bool {
        matches!(self, Self::Debug)
    }
}

#[derive(Clone, Copy)]
enum TestMode {
    Compile,
    CompileAndRun,
}

impl TestMode {
    fn should_run(self) -> bool {
        matches!(self, Self::CompileAndRun)
    }
}

fn build_test_runner(
    tempdir: PathBuf,
    current_dir: String,
    build_mode: BuildMode,
    test_kind: &str,
    test_dir: &str,
    test_mode: TestMode,
    files_to_ignore_on_m68k: &'static [&'static str],
) {
    fn rust_filter(path: &Path) -> bool {
        path.is_file() && path.extension().expect("extension").to_str().expect("to_str") == "rs"
    }

    #[cfg(feature = "master")]
    fn filter(filename: &Path) -> bool {
        rust_filter(filename)
    }

    #[cfg(not(feature = "master"))]
    fn filter(filename: &Path) -> bool {
        if let Some(filename) = filename.to_str()
            && filename.ends_with("gep.rs")
        {
            return false;
        }
        rust_filter(filename)
    }

    println!("=== {test_kind} tests ===");

    // TODO(antoyo): find a way to send this via a cli argument.
    let test_target = std::env::var("CG_GCC_TEST_TARGET").ok();
    let test_target_filter = test_target.clone();

    LangTester::new()
        .test_dir(test_dir)
        .test_path_filter(move |filename| {
            if !filter(filename) {
                return false;
            }
            if test_target_filter.is_some()
                && let Some(filename) = filename.file_name()
                && let Some(filename) = filename.to_str()
                && files_to_ignore_on_m68k.contains(&filename)
            {
                return false;
            }
            true
        })
        .test_extract(|path| {
            std::fs::read_to_string(path)
                .expect("read file")
                .lines()
                .skip_while(|l| !l.starts_with("//"))
                .take_while(|l| l.starts_with("//"))
                .map(|l| &l[2..])
                .collect::<Vec<_>>()
                .join("\n")
        })
        .test_cmds(move |path| {
            // Test command 1: Compile `x.rs` into `tempdir/x`.
            let mut exe = PathBuf::new();
            exe.push(&tempdir);
            exe.push(path.file_stem().expect("file_stem"));
            let mut compiler_args = vec![
                format!("-Zcodegen-backend={}/target/debug/librustc_codegen_gcc.so", current_dir),
                "--sysroot".into(),
                format!("{}/build/build_sysroot/sysroot/", current_dir),
                "-C".into(),
                "link-arg=-lc".into(),
                "--extern".into(),
                "mini_core=target/out/libmini_core.rlib".into(),
                "-o".into(),
                exe.to_str().expect("to_str").into(),
                path.to_str().expect("to_str").into(),
            ];

            if let Some(ref target) = test_target {
                compiler_args.extend_from_slice(&["--target".into(), target.into()]);

                let linker = format!("{}-gcc", target);
                compiler_args.push(format!("-Clinker={}", linker));
            }

            if let Some(flags) = option_env!("TEST_FLAGS") {
                for flag in flags.split_whitespace() {
                    compiler_args.push(flag.into());
                }
            }

            if build_mode.is_debug() {
                compiler_args
                    .extend_from_slice(&["-C".to_string(), "llvm-args=sanitize-undefined".into()]);
                if test_target.is_none() {
                    // m68k doesn't have lubsan for now
                    compiler_args.extend_from_slice(&["-C".into(), "link-args=-lubsan".into()]);
                }
            } else {
                compiler_args.extend_from_slice(&[
                    "-C".into(),
                    "opt-level=3".into(),
                    "-C".into(),
                    "lto=no".into(),
                ]);
            }

            compile_and_run_cmds(compiler_args, &test_target, &exe, test_mode)
        })
        .run();
}

fn compile_tests(tempdir: PathBuf, current_dir: String) {
    build_test_runner(
        tempdir,
        current_dir,
        BuildMode::Debug,
        "lang compile",
        "tests/compile",
        TestMode::Compile,
        &["simd-ffi.rs", "asm_nul_byte.rs", "global_asm_nul_byte.rs", "naked_asm_nul_byte.rs"],
    );
}

fn run_tests(tempdir: PathBuf, current_dir: String) {
    build_test_runner(
        tempdir.clone(),
        current_dir.clone(),
        BuildMode::Debug,
        "[DEBUG] lang run",
        "tests/run",
        TestMode::CompileAndRun,
        &[],
    );
    build_test_runner(
        tempdir,
        current_dir.to_string(),
        BuildMode::Release,
        "[RELEASE] lang run",
        "tests/run",
        TestMode::CompileAndRun,
        &[],
    );
}

fn main() {
    let tempdir = TempDir::new().expect("temp dir");
    let current_dir = current_dir().expect("current dir");
    let current_dir = current_dir.to_str().expect("current dir").to_string();

    let tempdir_path: PathBuf = tempdir.as_ref().into();
    compile_tests(tempdir_path.clone(), current_dir.clone());
    run_tests(tempdir_path, current_dir);
}
