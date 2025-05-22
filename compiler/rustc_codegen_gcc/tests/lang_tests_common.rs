//! The common code for `tests/lang_tests_*.rs`

#![allow(clippy::uninlined_format_args)]

use std::env::{self, current_dir};
use std::path::{Path, PathBuf};
use std::process::Command;

use boml::Toml;
use lang_tester::LangTester;
use tempfile::TempDir;

/// Controls the compile options (e.g., optimization level) used to compile
/// test code.
#[allow(dead_code)] // Each test crate picks one variant
pub enum Profile {
    Debug,
    Release,
}

pub fn main_inner(profile: Profile) {
    let tempdir = TempDir::new().expect("temp dir");
    let current_dir = current_dir().expect("current dir");
    let current_dir = current_dir.to_str().expect("current dir").to_string();

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    let gcc_path = std::fs::read_to_string(manifest_dir.join("config.toml"))
        .ok()
        .and_then(|v| {
            let toml = Toml::parse(&v).expect("Failed to parse `config.toml`");
            toml.get_string("gcc-path").map(PathBuf::from).ok()
        })
        .unwrap_or_else(|| {
            // then we try to retrieve it from the `target` folder.
            let commit = include_str!("../libgccjit.version").trim();
            Path::new("build/libgccjit").join(commit)
        });

    let gcc_path = Path::new(&gcc_path)
        .canonicalize()
        .expect("failed to get absolute path of `gcc-path`")
        .display()
        .to_string();
    unsafe {
        env::set_var("LD_LIBRARY_PATH", gcc_path);
    }

    fn rust_filter(path: &Path) -> bool {
        path.is_file() && path.extension().expect("extension").to_str().expect("to_str") == "rs"
    }

    #[cfg(feature = "master")]
    fn filter(filename: &Path) -> bool {
        rust_filter(filename)
    }

    #[cfg(not(feature = "master"))]
    fn filter(filename: &Path) -> bool {
        if let Some(filename) = filename.to_str() {
            if filename.ends_with("gep.rs") {
                return false;
            }
        }
        rust_filter(filename)
    }

    LangTester::new()
        .test_dir("tests/run")
        .test_path_filter(filter)
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
            let mut compiler = Command::new("rustc");
            compiler.args([
                &format!("-Zcodegen-backend={}/target/debug/librustc_codegen_gcc.so", current_dir),
                "--sysroot",
                &format!("{}/build/build_sysroot/sysroot/", current_dir),
                "-C",
                "link-arg=-lc",
                "--extern",
                "mini_core=target/out/libmini_core.rlib",
                "-o",
                exe.to_str().expect("to_str"),
                path.to_str().expect("to_str"),
            ]);

            // TODO(antoyo): find a way to send this via a cli argument.
            let test_target = std::env::var("CG_GCC_TEST_TARGET");
            if let Ok(ref target) = test_target {
                compiler.args(["--target", target]);
                let linker = format!("{}-gcc", target);
                compiler.args(&[format!("-Clinker={}", linker)]);
                let mut env_path = std::env::var("PATH").unwrap_or_default();
                // TODO(antoyo): find a better way to add the PATH necessary locally.
                env_path = format!("/opt/m68k-unknown-linux-gnu/bin:{}", env_path);
                compiler.env("PATH", env_path);
            }

            if let Some(flags) = option_env!("TEST_FLAGS") {
                for flag in flags.split_whitespace() {
                    compiler.arg(flag);
                }
            }
            match profile {
                Profile::Debug => {}
                Profile::Release => {
                    compiler.args(["-C", "opt-level=3", "-C", "lto=no"]);
                }
            }
            // Test command 2: run `tempdir/x`.
            if test_target.is_ok() {
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
                copy.args([&exe, &vm_exe_path]);

                let mut runtime = Command::new("sudo");
                runtime.args(["chroot", vm_dir, "qemu-m68k-static"]);
                runtime.arg(inside_vm_exe_path);
                runtime.current_dir(vm_parent_dir);
                vec![("Compiler", compiler), ("Copy", copy), ("Run-time", runtime)]
            } else {
                let runtime = Command::new(exe);
                vec![("Compiler", compiler), ("Run-time", runtime)]
            }
        })
        .run();
}
