// Check that valid binaries are persisted by running them, regardless of whether the
// --run or --no-run option is used.

//@ ignore-cross-compile

use std::path::Path;

use run_make_support::{rfs, run, rustc, rustdoc};

fn setup_test_env<F: FnOnce(&Path, &Path)>(callback: F) {
    let out_dir = Path::new("doctests");
    rfs::create_dir(&out_dir);
    rustc().input("t.rs").crate_type("rlib").run();
    callback(&out_dir, Path::new("libt.rlib"));
    rfs::remove_dir_all(out_dir);
}

fn check_generated_binaries() {
    let mut found_merged_doctest = false;
    rfs::read_dir_entries("doctests/", |path| {
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("merged_doctest_2024"))
        {
            found_merged_doctest = true;
            let rust_out = path.join("rust_out");
            let rust_out = rust_out.to_string_lossy();
            run(&*rust_out);
        }
    });
    if !found_merged_doctest {
        panic!("no directory starting with `merged_doctest_2024` found under `doctests/`");
    }
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
            .edition("2024")
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
            .edition("2024")
            .run();
        check_generated_binaries();
    });
    // Behavior with --test-run-directory with relative paths.
    setup_test_env(|_, _| {
        let run_dir_path = Path::new("rundir");
        rfs::create_dir(&run_dir_path);

        rustdoc()
            .input("t.rs")
            .arg("-Zunstable-options")
            .arg("--test")
            .arg("--persist-doctests")
            .arg("doctests")
            .arg("--test-run-directory")
            .arg(run_dir_path)
            .extern_("t", "libt.rlib")
            .edition("2024")
            .run();

        rfs::remove_dir_all(run_dir_path);
    });
}
