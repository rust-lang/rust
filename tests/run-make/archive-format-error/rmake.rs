// Regression test for https://github.com/rust-lang/rust/issues/148217
// Two-layer defense: a warning for incompatible archive format, and an error
// (not ICE) for corrupt member offsets.

//@ ignore-cross-compile

use run_make_support::{cc, llvm_ar, path, rfs, rustc, static_lib_name, target};

fn main() {
    rfs::create_dir("archive");

    // Test 1 (first defense): BSD format archive on a GNU/Linux target should
    // emit a format mismatch warning.
    if target().contains("linux") {
        cc().arg("-c").input("native.c").output("archive/native.o").run();
        let bsd_archive = path("archive").join(static_lib_name("native_bsd"));
        llvm_ar()
            .arg("rcus")
            .arg("--format=bsd")
            .output_input(&bsd_archive, "archive/native.o")
            .run();
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .library_search_path("archive")
            .arg("-lstatic=native_bsd")
            .run()
            .assert_stderr_contains("BSD")
            .assert_stderr_contains("GNU");
    }

    // Test 2 (second defense): corrupt archive with member offset exceeding
    // file boundary should produce an error, not an ICE.
    let corrupt_archive = path("archive").join(static_lib_name("corrupt"));
    create_corrupt_archive(&corrupt_archive);
    rustc()
        .input("lib.rs")
        .crate_type("rlib")
        .library_search_path("archive")
        .arg("-lstatic=corrupt")
        .run_fail()
        .assert_stderr_not_contains("panicked")
        .assert_stderr_not_contains("unexpectedly panicked")
        .assert_stderr_contains("archive");
}

fn create_corrupt_archive(output_path: &std::path::Path) {
    use std::fs;
    use std::io::Write;

    let mut archive = b"!<arch>\n".to_vec();

    let member_name = "corrupt.o/        ";
    let mtime = "0           ";
    let uid = "0     ";
    let gid = "0     ";
    let mode = "100644  ";
    let size = "10000       ";
    let fmag = "`\n";

    archive.extend_from_slice(member_name.as_bytes());
    archive.extend_from_slice(mtime.as_bytes());
    archive.extend_from_slice(uid.as_bytes());
    archive.extend_from_slice(gid.as_bytes());
    archive.extend_from_slice(mode.as_bytes());
    archive.extend_from_slice(size.as_bytes());
    archive.extend_from_slice(fmag.as_bytes());

    archive.extend_from_slice(b"small_data");

    archive.push(b'\n');

    fs::write(output_path, archive).unwrap();
}
