use std::{
    env::{self, current_dir},
    path::PathBuf,
    process::Command,
};

use lang_tester::LangTester;
use tempfile::TempDir;

fn main() {
    let tempdir = TempDir::new().expect("temp dir");
    let current_dir = current_dir().expect("current dir");
    let current_dir = current_dir.to_str().expect("current dir").to_string();
    let gcc_path = include_str!("../gcc_path");
    let gcc_path = gcc_path.trim();
    env::set_var("LD_LIBRARY_PATH", gcc_path);
    LangTester::new()
        .test_dir("tests/run")
        .test_file_filter(|path| path.extension().expect("extension").to_str().expect("to_str") == "rs")
        .test_extract(|source| {
            let lines =
                source.lines()
                    .skip_while(|l| !l.starts_with("//"))
                    .take_while(|l| l.starts_with("//"))
                    .map(|l| &l[2..])
                    .collect::<Vec<_>>()
                    .join("\n");
            Some(lines)
        })
        .test_cmds(move |path| {
            // Test command 1: Compile `x.rs` into `tempdir/x`.
            let mut exe = PathBuf::new();
            exe.push(&tempdir);
            exe.push(path.file_stem().expect("file_stem"));
            let mut compiler = Command::new("rustc");
            compiler.args(&[
                &format!("-Zcodegen-backend={}/target/debug/librustc_codegen_gcc.so", current_dir),
                "--sysroot", &format!("{}/build_sysroot/sysroot/", current_dir),
                "-Zno-parallel-llvm",
                "-C", "panic=abort",
                "-C", "link-arg=-lc",
                "-o", exe.to_str().expect("to_str"),
                path.to_str().expect("to_str"),
            ]);
            // Test command 2: run `tempdir/x`.
            let runtime = Command::new(exe);
            vec![("Compiler", compiler), ("Run-time", runtime)]
        })
        .run();
}
