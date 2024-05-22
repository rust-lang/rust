// Check that valid binaries are persisted by running them, regardless of whether the
// --run or --no-run option is used.

use run_make_support::{run, rustc, rustdoc, tmp_dir};
use std::fs::{create_dir, remove_dir_all};
use std::path::Path;

fn setup_test_env<F: FnOnce(&Path, &Path)>(callback: F) {
    let out_dir = tmp_dir().join("doctests");
    create_dir(&out_dir).expect("failed to create doctests folder");
    rustc().input("t.rs").crate_type("rlib").run();
    callback(&out_dir, &tmp_dir().join("libt.rlib"));
    remove_dir_all(out_dir);
}

fn check_generated_binaries() {
    run("doctests/t_rs_2_0/rust_out");
    run("doctests/t_rs_8_0/rust_out");
}

fn main() {
    setup_test_env(|out_dir, extern_path| {
        rustdoc()
            .input("t.rs")
            .arg("-Zunstable-options")
            .arg("--test")
            .arg("--persist-doctests")
            .arg(out_dir)
            .extern_("t", extern_path)
            .run();
        check_generated_binaries();
    });
    setup_test_env(|out_dir, extern_path| {
        rustdoc()
            .input("t.rs")
            .arg("-Zunstable-options")
            .arg("--test")
            .arg("--persist-doctests")
            .arg(out_dir)
            .extern_("t", extern_path)
            .arg("--no-run")
            .run();
        check_generated_binaries();
    });
    // Behavior with --test-run-directory with relative paths.
    setup_test_env(|_out_dir, extern_path| {
        let run_dir = "rundir";
        let run_dir_path = tmp_dir().join("rundir");
        create_dir(&run_dir_path).expect("failed to create rundir folder");

        rustdoc()
            .current_dir(tmp_dir())
            .input(std::env::current_dir().unwrap().join("t.rs"))
            .arg("-Zunstable-options")
            .arg("--test")
            .arg("--persist-doctests")
            .arg("doctests")
            .arg("--test-run-directory")
            .arg(run_dir)
            .extern_("t", "libt.rlib")
            .run();

        remove_dir_all(run_dir_path);
    });
}
