// ignore-tidy-linelength
// Reason: intel.com link

// This security test checks that the disassembled form of certain symbols
// is "hardened" - that means, the assembly instructions match a pattern that
// mitigate potential Load Value Injection vulnerabilities.
// To do so, a test crate is compiled, and certain symbols are found, disassembled
// and checked one by one.
// See https://github.com/rust-lang/rust/pull/77008

// On load value injection:
// https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/technical-documentation/load-value-injection.html

//@ only-x86_64-fortanix-unknown-sgx

use run_make_support::{cmd, cwd, llvm_filecheck, llvm_objdump, regex, set_current_dir, target};

fn main() {
    let main_dir = cwd();
    set_current_dir("enclave");
    // HACK(eddyb) sets `RUSTC_BOOTSTRAP=1` so Cargo can accept nightly features.
    // These come from the top-level Rust workspace, that this crate is not a
    // member of, but Cargo tries to load the workspace `Cargo.toml` anyway.
    cmd("cargo")
        .env("RUSTC_BOOTSTRAP", "1")
        .arg("-v")
        .arg("run")
        .arg("--target")
        .arg(target())
        .run();
    set_current_dir(&main_dir);
    // Rust has various ways of adding code to a binary:
    // - Rust code
    // - Inline assembly
    // - Global assembly
    // - C/C++ code compiled as part of Rust crates
    // For those different kinds, we do have very small code examples that should be
    // mitigated in some way. Mostly we check that ret instructions should no longer be present.
    check("unw_getcontext", "unw_getcontext.checks");
    check("__libunwind_Registers_x86_64_jumpto", "jumpto.checks");

    check("std::io::stdio::_print::[[:alnum:]]+", "print.with_frame_pointers.checks");

    check("rust_plus_one_global_asm", "rust_plus_one_global_asm.checks");

    check("cc_plus_one_c", "cc_plus_one_c.checks");
    check("cc_plus_one_c_asm", "cc_plus_one_c_asm.checks");
    check("cc_plus_one_cxx", "cc_plus_one_cxx.checks");
    check("cc_plus_one_cxx_asm", "cc_plus_one_cxx_asm.checks");
    check("cc_plus_one_asm", "cc_plus_one_asm.checks");

    check("cmake_plus_one_c", "cmake_plus_one_c.checks");
    check("cmake_plus_one_c_asm", "cmake_plus_one_c_asm.checks");
    check("cmake_plus_one_c_global_asm", "cmake_plus_one_c_global_asm.checks");
    check("cmake_plus_one_cxx", "cmake_plus_one_cxx.checks");
    check("cmake_plus_one_cxx_asm", "cmake_plus_one_cxx_asm.checks");
    check("cmake_plus_one_cxx_global_asm", "cmake_plus_one_cxx_global_asm.checks");
    check("cmake_plus_one_asm", "cmake_plus_one_asm.checks");
}

fn check(func_re: &str, mut checks: &str) {
    let dump = llvm_objdump()
        .input("enclave/target/x86_64-fortanix-unknown-sgx/debug/enclave")
        .args(&["--syms", "--demangle"])
        .run()
        .stdout_utf8();
    let re = regex::Regex::new(&format!("[[:blank:]]+{func_re}")).unwrap();
    let func = re.find_iter(&dump).map(|m| m.as_str().trim()).collect::<Vec<&str>>().join(",");
    assert!(!func.is_empty());
    let dump = llvm_objdump()
        .input("enclave/target/x86_64-fortanix-unknown-sgx/debug/enclave")
        .args(&["--demangle", &format!("--disassemble-symbols={func}")])
        .run()
        .stdout_utf8();
    let dump = dump.as_bytes();

    // Unique case, must succeed at one of two possible tests.
    // This is because frame pointers are optional, and them being enabled requires
    // an additional `popq` in the pattern checking file.
    if func_re == "std::io::stdio::_print::[[:alnum:]]+" {
        let output = llvm_filecheck().stdin_buf(&dump).patterns(checks).run_unchecked();
        if !output.status().success() {
            checks = "print.without_frame_pointers.checks";
            llvm_filecheck().stdin_buf(&dump).patterns(checks).run();
        }
    } else {
        llvm_filecheck().stdin_buf(&dump).patterns(checks).run();
    }
    if !["rust_plus_one_global_asm", "cmake_plus_one_c_global_asm", "cmake_plus_one_cxx_global_asm"]
        .contains(&func_re)
    {
        // The assembler cannot avoid explicit `ret` instructions. Sequences
        // of `shlq $0x0, (%rsp); lfence; retq` are used instead.
        llvm_filecheck()
            .args(&["--implicit-check-not", "ret"])
            .stdin_buf(dump)
            .patterns(checks)
            .run();
    }
}
