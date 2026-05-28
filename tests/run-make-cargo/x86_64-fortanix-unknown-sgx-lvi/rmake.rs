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

use run_make_support::{
    cargo, cwd, llvm_filecheck, llvm_objdump, regex, run, set_current_dir, target,
};

fn main() {
    cargo()
        .arg("-v")
        .arg("build")
        .arg("--target")
        .arg(target())
        .current_dir("enclave")
        .env("CC_x86_64_fortanix_unknown_sgx", "clang")
        .env(
            "CFLAGS_x86_64_fortanix_unknown_sgx",
            "-D__ELF__ -isystem/usr/include/x86_64-linux-gnu -mlvi-hardening -mllvm -x86-experimental-lvi-inline-asm-hardening",
        )
        .env("CXX_x86_64_fortanix_unknown_sgx", "clang++")
        .env(
            "CXXFLAGS_x86_64_fortanix_unknown_sgx",
            "-D__ELF__ -isystem/usr/include/x86_64-linux-gnu -mlvi-hardening -mllvm -x86-experimental-lvi-inline-asm-hardening",
        )
        .run();

    // Rust has several ways of including machine code into a binary:
    //
    // - Rust code
    // - Inline assembly
    // - Global assembly
    // - C/C++ code compiled as part of Rust crates
    //
    // For each of those, check that the mitigations are applied. Mostly we check
    // that ret instructions are no longer present.

    // Check that normal rust code has the right mitigations.
    check("unw_getcontext", "unw_getcontext.checks");
    check("__libunwind_Registers_x86_64_jumpto", "jumpto.checks");

    check("std::io::stdio::_print::[[:alnum:]]+", "print.with_frame_pointers.checks");

    // Check that rust global assembly has the right mitigations.
    check("rust_plus_one_global_asm", "rust_plus_one_global_asm.checks");

    // Check that C code compiled using the `cc` crate has the right mitigations.
    check("cc_plus_one_c", "cc_plus_one_c.checks");
    check("cc_plus_one_c_asm", "cc_plus_one_c_asm.checks");
    check("cc_plus_one_cxx", "cc_plus_one_cxx.checks");
    check("cc_plus_one_cxx_asm", "cc_plus_one_cxx_asm.checks");
    check("cc_plus_one_asm", "cc_plus_one_asm.checks");

    // Check that C++ code compiled using the `cc` crate has the right mitigations.
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
        .stdout();

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
