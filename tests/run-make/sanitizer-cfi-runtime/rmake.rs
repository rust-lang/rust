//@ needs-sanitizer-support
//@ needs-sanitizer-cfi

use run_make_support::{run, run_fail, rustc};

fn main() {
    // 1. Check link args for default CFI (no diag/recover, no UBSan runtime)
    let link_args_default = rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .print("link-args")
        .input("program.rs")
        .run()
        .stdout_utf8();
    assert!(
        !link_args_default.contains("ubsan"),
        "did not expect any ubsan runtime in link args, got: {link_args_default}"
    );

    // 2. Check link args for full UBSan runtime with CFI diag mode
    let link_args_full_diag = rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .arg("-Zsanitizer-cfi-diag=true")
        .print("link-args")
        .input("program.rs")
        .run()
        .stdout_utf8();
    assert!(
        link_args_full_diag.contains("rt.ubsan.") || link_args_full_diag.contains("rt.ubsan\""),
        "expected ubsan runtime in link args, got: {link_args_full_diag}"
    );
    assert!(
        !link_args_full_diag.contains("ubsan_minimal"),
        "did not expect ubsan_minimal in link args, got: {link_args_full_diag}"
    );

    // 3. Check link args for full UBSan runtime with CFI recover mode
    let link_args_full_recover = rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .arg("-Zsanitizer-cfi-recover=true")
        .print("link-args")
        .input("program.rs")
        .run()
        .stdout_utf8();
    assert!(
        link_args_full_recover.contains("rt.ubsan.")
            || link_args_full_recover.contains("rt.ubsan\""),
        "expected ubsan runtime in link args, got: {link_args_full_recover}"
    );
    assert!(
        !link_args_full_recover.contains("ubsan_minimal"),
        "did not expect ubsan_minimal in link args, got: {link_args_full_recover}"
    );

    // 4. Check link args for minimal UBSan runtime with CFI diag mode
    let link_args_min_diag = rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .arg("-Zsanitizer-cfi-diag=true")
        .arg("-Zsanitizer-cfi-minimal-runtime=true")
        .print("link-args")
        .input("program.rs")
        .run()
        .stdout_utf8();
    assert!(
        link_args_min_diag.contains("rt.ubsan_minimal.")
            || link_args_min_diag.contains("rt.ubsan_minimal\""),
        "expected ubsan_minimal runtime in link args, got: {link_args_min_diag}"
    );

    // 5. Check link args for minimal UBSan runtime with CFI recover mode
    let link_args_min_recover = rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .arg("-Zsanitizer-cfi-recover=true")
        .arg("-Zsanitizer-cfi-minimal-runtime=true")
        .print("link-args")
        .input("program.rs")
        .run()
        .stdout_utf8();
    assert!(
        link_args_min_recover.contains("rt.ubsan_minimal.")
            || link_args_min_recover.contains("rt.ubsan_minimal\""),
        "expected ubsan_minimal runtime in link args, got: {link_args_min_recover}"
    );

    // 6. Build and run binary with default CFI (trap mode, no runtime)
    rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .output("program_default")
        .input("program.rs")
        .run();
    run_fail("program_default");

    // 7. Build and run binary with full runtime in diag (abort) mode
    rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .arg("-Zsanitizer-cfi-diag=true")
        .output("program_full_diag")
        .input("program.rs")
        .run();
    run_fail("program_full_diag")
        .assert_stderr_contains("runtime error: control flow integrity check for type");

    // 8. Build and run binary with full runtime in recover mode
    rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .arg("-Zsanitizer-cfi-recover=true")
        .output("program_full_recover")
        .input("program.rs")
        .run();
    run("program_full_recover")
        .assert_stderr_contains("runtime error: control flow integrity check for type");

    // 9. Build and run binary with minimal runtime in diag (abort) mode
    rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .arg("-Zsanitizer-cfi-diag=true")
        .arg("-Zsanitizer-cfi-minimal-runtime=true")
        .output("program_min_diag")
        .input("program.rs")
        .run();
    run_fail("program_min_diag").assert_stderr_contains("ubsan: cfi-check-fail");

    // 10. Build and run binary with minimal runtime in recover mode
    rustc()
        .arg("-Clto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Zsanitizer=cfi")
        .arg("-Zsanitizer-cfi-recover=true")
        .arg("-Zsanitizer-cfi-minimal-runtime=true")
        .output("program_min_recover")
        .input("program.rs")
        .run();
    run("program_min_recover").assert_stderr_contains("ubsan: cfi-check-fail");
}
