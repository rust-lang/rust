// This ensures that a cdylib that uses std and panic=unwind but does not
// have any panics itself will not have *any* panic-related code in the final
// binary, at least when using fat LTO
// (since all the necessary nounwind propagation requires fat LTO).
//
// This code used to be pulled in via a landing pad in the personality function,
// (since that is `extern "C"` and therefore panics if something unwinds), so
// if this failed because you modified the personality function, ensure it contains
// no potentially unwinding calls.

use run_make_support::{cargo, dynamic_lib_name, llvm_nm, path, rustc, target};

fn main() {
    let target_dir = path("target");

    // We use build-std to ensure that the sysroot does not have debug assertions,
    // as this doesn't work with debug assertions.
    cargo()
        .args(&[
            "build",
            "--manifest-path",
            "Cargo.toml",
            "--release",
            "-Zbuild-std=std",
            "--target",
            &target(),
        ])
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("RUSTC_BOOTSTRAP", "1")
        .run();

    let output_path = target_dir.join(target()).join("release").join(dynamic_lib_name("add"));

    llvm_nm()
        .input(output_path)
        .run()
        // a collection of panic-related strings. if this appears in the output
        // for other reasons than having panic symbols, I am sorry.
        .assert_stdout_not_contains("panic")
        .assert_stdout_not_contains("addr2line")
        .assert_stdout_not_contains("backtrace")
        .assert_stdout_not_contains("gimli");
}
