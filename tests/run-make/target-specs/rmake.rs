// Target-specific compilation in rustc used to have case-by-case peculiarities in 2014,
// with the compiler having redundant target types and unspecific names. An overarching rework
// in #16156 changed the way the target flag functions, and this test attempts compilation
// with the target flag's bundle of new features to check that compilation either succeeds while
// using them correctly, or fails with the right error message when using them improperly.
// See https://github.com/rust-lang/rust/pull/16156

use run_make_support::{diff, rfs, rustc};

fn main() {
    rustc().input("foo.rs").target("my-awesome-platform.json").crate_type("lib").emit("asm").run();
    assert!(!rfs::read_to_string("foo.s").contains("morestack"));
    rustc()
        .input("foo.rs")
        .target("my-invalid-platform.json")
        .run_fail()
        .assert_stderr_contains("error loading target specification");
    rustc()
        .input("foo.rs")
        .target("my-incomplete-platform.json")
        .run_fail()
        .assert_stderr_contains("Field llvm-target");
    rustc()
        .env("RUST_TARGET_PATH", ".")
        .input("foo.rs")
        .target("my-awesome-platform")
        .crate_type("lib")
        .emit("asm")
        .run();
    rustc()
        .env("RUST_TARGET_PATH", ".")
        .input("foo.rs")
        .target("my-x86_64-unknown-linux-gnu-platform")
        .crate_type("lib")
        .emit("asm")
        .run();
    let test_platform = rustc()
        .arg("-Zunstable-options")
        .target("my-awesome-platform.json")
        .print("target-spec-json")
        .run()
        .stdout_utf8();
    rfs::create_file("test-platform.json");
    rfs::write("test-platform.json", test_platform.as_bytes());
    let test_platform_2 = rustc()
        .arg("-Zunstable-options")
        .target("test-platform.json")
        .print("target-spec-json")
        .run()
        .stdout_utf8();
    diff()
        .expected_file("test-platform.json")
        .actual_text("test-platform-2", test_platform_2)
        .run();
    rustc()
        .input("foo.rs")
        .target("endianness-mismatch")
        .run_fail()
        .assert_stderr_contains(r#""data-layout" claims architecture is little-endian"#);
    rustc()
        .input("foo.rs")
        .target("mismatching-data-layout")
        .crate_type("lib")
        .run_fail()
        .assert_stderr_contains("data-layout for target");
    rustc()
        .input("foo.rs")
        .target("require-explicit-cpu")
        .crate_type("lib")
        .run_fail()
        .assert_stderr_contains("target requires explicitly specifying a cpu");
    rustc()
        .input("foo.rs")
        .target("require-explicit-cpu")
        .crate_type("lib")
        .arg("-Ctarget-cpu=generic")
        .run();
    rustc().target("require-explicit-cpu").arg("--print=target-cpus").run();
}
