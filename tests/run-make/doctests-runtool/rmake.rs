// Tests behavior of rustdoc `--runtool`.

use run_make_support::{rustc, rustdoc, tmp_dir};
use std::env::current_dir;
use std::fs::{create_dir, remove_dir_all};
use std::path::PathBuf;

fn mkdir(name: &str) -> PathBuf {
    let dir = tmp_dir().join(name);
    create_dir(&dir).expect("failed to create doctests folder");
    dir
}

// Behavior with --runtool with relative paths and --test-run-directory.
fn main() {
    let run_dir_name = "rundir";
    let run_dir = mkdir(run_dir_name);
    let run_tool = mkdir("runtool");
    let run_tool_binary = run_tool.join("runtool");

    rustc().input("t.rs").crate_type("rlib").run();
    rustc().input("runtool.rs").output(&run_tool_binary).run();

    rustdoc()
        .input(current_dir().unwrap().join("t.rs"))
        .arg("-Zunstable-options")
        .arg("--test")
        .arg("--test-run-directory")
        .arg(run_dir_name)
        .arg("--runtool")
        .arg(&run_tool_binary)
        .extern_("t", "libt.rlib")
        .current_dir(tmp_dir())
        .run();

    remove_dir_all(run_dir);
    remove_dir_all(run_tool);
}
