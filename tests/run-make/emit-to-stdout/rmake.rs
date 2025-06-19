//@ needs-target-std
//! If `-o -` or `--emit KIND=-` is provided, output should be written to stdout
//! instead. Binary output (`obj`, `llvm-bc`, `link` and `metadata`)
//! being written this way will result in an error if stdout is a tty.
//! Multiple output types going to stdout will trigger an error too,
//! as they would all be mixed together.
//!
//! See <https://github.com/rust-lang/rust/pull/111626>.

use std::fs::File;

use run_make_support::{diff, run_in_tmpdir, rustc};

// Test emitting text outputs to stdout works correctly
fn run_diff(name: &str, file_args: &[&str]) {
    rustc().emit(format!("{name}={name}")).input("test.rs").args(file_args).run();
    let out = rustc().emit(format!("{name}=-")).input("test.rs").run().stdout_utf8();
    diff().expected_file(name).actual_text("stdout", &out).run();
}

// Test that emitting binary formats to a terminal gives the correct error
fn run_terminal_err_diff(name: &str) {
    #[cfg(not(windows))]
    let terminal = File::create("/dev/ptmx").unwrap();
    // FIXME: If this test fails and the compiler does print to the console,
    // then this will produce a lot of output.
    // We should spawn a new console instead to print stdout.
    #[cfg(windows)]
    let terminal = File::options().read(true).write(true).open(r"\\.\CONOUT$").unwrap();

    let err = File::create(name).unwrap();
    rustc().emit(format!("{name}=-")).input("test.rs").stdout(terminal).stderr(err).run_fail();
    diff().expected_file(format!("emit-{name}.stderr")).actual_file(name).run();
}

fn main() {
    run_in_tmpdir(|| {
        run_diff("asm", &[]);
        run_diff("llvm-ir", &[]);
        run_diff("dep-info", &["-Zdep-info-omit-d-target=yes"]);
        run_diff("mir", &[]);

        run_terminal_err_diff("llvm-bc");
        run_terminal_err_diff("obj");
        run_terminal_err_diff("metadata");
        run_terminal_err_diff("link");

        // Test error for emitting multiple types to stdout
        rustc()
            .input("test.rs")
            .emit("asm=-")
            .emit("llvm-ir=-")
            .emit("dep-info=-")
            .emit("mir=-")
            .stderr(File::create("multiple-types").unwrap())
            .run_fail();
        diff().expected_file("emit-multiple-types.stderr").actual_file("multiple-types").run();

        // Same as above, but using `-o`
        rustc()
            .input("test.rs")
            .output("-")
            .emit("asm,llvm-ir,dep-info,mir")
            .stderr(File::create("multiple-types-option-o").unwrap())
            .run_fail();
        diff()
            .expected_file("emit-multiple-types.stderr")
            .actual_file("multiple-types-option-o")
            .run();

        // Test that `-o -` redirected to a file works correctly (#26719)
        rustc().input("test.rs").output("-").stdout(File::create("out-stdout").unwrap()).run();
    });
}
