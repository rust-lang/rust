// C/Rust interop under function pointer type discrimination, checked on the IR level (no need for
// linker/libc).
//
// Rust signs a function pointer with the discriminator of the function it points to and never
// resigns implicitly, while Clang encodes a code address according to the C type of the slot
// holding it. Interop is correct when, at every boundary, both compilers use the same
// discriminator. The test reads the discriminators Clang uses for `int (*)(int, int)` and
// `void (*)(void)` from its IR, and checks that Rust uses exactly those:
// * for correctly typed callbacks, in both directions, without any resign;
// * through `ptrauth_c_cast`, for code addresses crossing a C slot that erased their type
//   (`void *`, a generic callback type), in both directions.
//
// The test assumes that a pointer-authentication-enabled `clang` is available on the path.

//@ only-pauthtest

use run_make_support::path_helpers::source_root;
use run_make_support::{cc, llvm_filecheck, rfs, rustc};

/// Returns the body of the function `name` in the LLVM IR `ir`.
fn function_body<'a>(ir: &'a str, name: &str) -> &'a str {
    let define = ir
        .lines()
        .find(|l| l.starts_with("define") && l.contains(&format!("@{name}(")))
        .unwrap_or_else(|| panic!("no definition of `{name}` in:\n{ir}"));
    let start = ir.find(define).unwrap();
    let end = start + ir[start..].find("\n}").unwrap();
    &ir[start..end]
}

/// Returns the number following the first occurrence of `prefix` in `text`.
fn number_after(text: &str, prefix: &str) -> u64 {
    let start = text.find(prefix).unwrap_or_else(|| panic!("`{prefix}` not found in:\n{text}"))
        + prefix.len();
    let digits: String = text[start..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().unwrap()
}

/// Returns the (authenticated, signed) discriminators of the first `llvm.ptrauth.resign` call in
/// `body`. The resigned value may itself be a constant signed pointer, so the discriminators are
/// the last two of the call.
fn resign_discriminators(body: &str) -> (u64, u64) {
    let call = body
        .lines()
        .find(|l| l.contains("@llvm.ptrauth.resign("))
        .unwrap_or_else(|| panic!("no `llvm.ptrauth.resign` call in:\n{body}"));
    let prefix = "i32 0, i64 ";
    let discriminators: Vec<u64> =
        call.match_indices(prefix).map(|(i, _)| number_after(&call[i..], prefix)).collect();
    let &[.., from, to] = discriminators.as_slice() else {
        panic!("expected two discriminators in `{call}`");
    };
    (from, to)
}

fn main() {
    let sibling = source_root().join("tests/run-make/pauth-fn-ptr-type-discrimination-interop-ir");

    cc().input("interop.c")
        .args(&[
            "-target",
            "aarch64-unknown-linux-pauthtest",
            "-march=armv8.3-a+pauth",
            "-fptrauth-function-pointer-type-discrimination",
            "-O0",
            "-S",
            "-emit-llvm",
            "-o",
            "c.ll",
        ])
        .run();
    let c_ir = rfs::read_to_string("c.ll");

    // `apply` calls through `int (*)(int, int)`. `invoke_callback` casts `void (*)(void)` back to
    // `int (*)(int, int)`, resigning between the two.
    let binop = number_after(function_body(&c_ir, "apply"), "\"ptrauth\"(i32 0, i64 ");
    let (generic, to) = resign_discriminators(function_body(&c_ir, "invoke_callback"));
    assert_eq!(to, binop);
    assert_ne!(binop, 0);
    assert_ne!(generic, 0);
    assert_ne!(binop, generic);
    let defines = [format!("-DBINOP={binop}"), format!("-DGENERIC={generic}")];

    llvm_filecheck().patterns(sibling.join("c.check")).args(&defines).stdin_buf(&c_ir).run();

    rustc()
        .target("aarch64-unknown-linux-pauthtest")
        .input("interop.rs")
        .crate_type("lib")
        .edition("2021")
        .emit("llvm-ir")
        .opt_level("0")
        .arg("-Zpointer-authentication=+function-pointer-type-discrimination")
        .arg("-Cunsafe-allow-abi-mismatch=pointer-authentication")
        .run();

    // Functions are not emitted in source order: check each one with its own prefix.
    let rust_ir = rfs::read("interop.ll");
    for prefix in
        ["TO_C", "FROM_C", "FROM_VOID", "TO_GENERIC", "FROM_GENERIC", "TO_VOID", "ROUND_TRIP"]
    {
        llvm_filecheck()
            .patterns(sibling.join("rust.check"))
            .check_prefix(prefix)
            .args(&defines)
            .stdin_buf(&rust_ir)
            .run();
    }
}
