//@ ignore-cross-compile (needs to run host tool binary)

// Tests behavior of rustdoc `--test-runtool`.

use std::path::PathBuf;

use run_make_support::rfs::{create_dir, remove_dir_all};
use run_make_support::{rustc, rustdoc};

fn mkdir(name: &str) -> PathBuf {
    let dir = PathBuf::from(name);
    create_dir(&dir);
    dir
}

// Behavior with --test-runtool with relative paths and --test-run-directory.
fn main() {
    let run_dir_name = "rundir";
    let run_dir = mkdir(run_dir_name);
    let run_tool = mkdir("runtool");
    let run_tool_binary = run_tool.join("runtool");

    rustc().input("t.rs").crate_type("rlib").run();
    rustc().input("runtool.rs").output(&run_tool_binary).run();

    rustdoc()
        .input("t.rs")
        .arg("-Zunstable-options")
        .arg("--test")
        .arg("--test-run-directory")
        .arg(run_dir_name)
        .arg("--test-runtool")
        .arg(&run_tool_binary)
        .extern_("t", "libt.rlib")
        .run();

    remove_dir_all(run_dir);
    remove_dir_all(run_tool);
}
