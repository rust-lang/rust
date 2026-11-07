//@ ignore-cross-compile attempts to run the doctests

// Check that valid binaries are persisted by running them, regardless of whether the
// --run or --no-run option is used.

use std::path::Path;

use run_make_support::rfs::{create_dir, remove_dir_all};
use run_make_support::{run, rustc, rustdoc};

fn setup_test_env<F: FnOnce(&Path, &Path)>(callback: F) {
    let out_dir = Path::new("doctests");
    create_dir(&out_dir);
    rustc().input("t.rs").crate_type("rlib").run();
    callback(&out_dir, Path::new("libt.rlib"));
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
    setup_test_env(|_out_dir, _extern_path| {
        let run_dir_path = Path::new("rundir");
        create_dir(&run_dir_path);

        rustdoc()
            .input("t.rs")
            .arg("-Zunstable-options")
            .arg("--test")
            .arg("--persist-doctests")
            .arg("doctests")
            .arg("--test-run-directory")
            .arg(run_dir_path)
            .extern_("t", "libt.rlib")
            .run();

        remove_dir_all(run_dir_path);
    });
}
