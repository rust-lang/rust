//! Regression test for <https://github.com/rust-lang/rust/issues/135332>.
//!
//! We can't simply drop debuginfo location spans when LLVM's location discriminator value limit is
//! reached. Otherwise, with `-Z verify-llvm-ir` and fat LTO, LLVM will report a broken module for
//!
//! ```text
//! inlinable function call in a function with debug info must have a !dbg location
//! ```

//@ ignore-cross-compile
//@ needs-dynamic-linking
//@ only-nightly (requires unstable rustc flag)

// This test trips a check in the MSVC linker for an outdated processor:
// "LNK1322: cannot avoid potential ARM hazard (Cortex-A53 MPCore processor bug #843419)"
// Until MSVC removes this check:
// https://developercommunity.microsoft.com/t/Remove-checking-for-and-fixing-Cortex-A/10905134
// we'll need to disable this test on Arm64 Windows.
//@ ignore-aarch64-pc-windows-msvc

#![deny(warnings)]

use run_make_support::{dynamic_lib_name, rfs, rust_lib_name, rustc};

// Synthesize a function that will have a large (`n`) number of functions
// MIR-inlined into it. When combined with a proc-macro, all of these inline
// callsites will have the same span, forcing rustc to use the DWARF
// discriminator to distinguish between them. LLVM's capacity to store that
// discriminator is not infinite (currently it allocates 12 bits for a
// maximum value of 4096) so if this function gets big enough rustc's error
// handling path will be exercised.
fn generate_program(n: u32) -> String {
    let mut program = String::from("pub type BigType = Vec<Vec<String>>;\n\n");
    program.push_str("pub fn big_function() -> BigType {\n");
    program.push_str("    vec![\n");
    for i in 1..=n {
        program.push_str(&format!("vec![\"string{}\".to_owned()],\n", i));
    }
    program.push_str("    ]\n");
    program.push_str("}\n");
    program
}

fn main() {
    // The reported threshold is around 1366 (4096/3), but let's bump it to
    // around 1500 to be less sensitive.
    rfs::write("generated.rs", generate_program(1500));

    rustc()
        .input("proc.rs")
        .crate_type("proc-macro")
        .edition("2021")
        .arg("-Cdebuginfo=line-tables-only")
        .run();
    rustc()
        .extern_("proc", dynamic_lib_name("proc"))
        .input("other.rs")
        .crate_type("rlib")
        .edition("2021")
        .opt_level("3")
        .arg("-Cdebuginfo=line-tables-only")
        .run();
    rustc()
        .extern_("other", rust_lib_name("other"))
        .input("main.rs")
        .edition("2021")
        .opt_level("3")
        .arg("-Cdebuginfo=line-tables-only")
        .arg("-Clto=fat")
        .arg("-Zverify-llvm-ir")
        .run();
}
