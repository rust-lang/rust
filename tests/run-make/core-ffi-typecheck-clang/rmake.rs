// This test checks that the clang defines for each target allign with the core ffi types defined in
// mod.rs. Therefore each rust target is queried and the clang defines for each target are read and
// compared to the core sizes to verify all types and sizes allign at buildtime.
//
// If this test fails because Rust adds a target that Clang does not support, this target should be
// added to SKIPPED_TARGETS.

use run_make_support::{clang, regex, rfs, rustc};

// It is not possible to run the Rust test-suite on these targets.
const SKIPPED_TARGETS: &[&str] = &[
    "riscv32gc-unknown-linux-gnu",
    "riscv32gc-unknown-linux-musl",
    "riscv32im-risc0-zkvm-elf",
    "riscv32imac-esp-espidf",
    "riscv32imafc-esp-espidf",
    "riscv32imafc-unknown-nuttx-elf",
    "riscv32imc-esp-espidf",
    "riscv32imac-unknown-nuttx-elf",
    "riscv32imac-unknown-xous-elf",
    "riscv32imc-unknown-nuttx-elf",
    "riscv32e-unknown-none-elf",
    "riscv32em-unknown-none-elf",
    "riscv32emc-unknown-none-elf",
    "riscv32i-unknown-none-elf",
    "riscv32im-unknown-none-elf",
    "riscv32imc-unknown-none-elf",
    "riscv32ima-unknown-none-elf",
    "riscv32imac-unknown-none-elf",
    "riscv32imafc-unknown-none-elf",
    "riscv64gc-unknown-freebsd",
    "riscv64gc-unknown-fuchsia",
    "riscv64gc-unknown-hermit",
    "riscv64gc-unknown-linux-gnu",
    "riscv64gc-unknown-linux-musl",
    "riscv64gc-unknown-netbsd",
    "riscv64gc-unknown-none-elf",
    "riscv64gc-unknown-nuttx-elf",
    "riscv64gc-unknown-openbsd",
    "riscv64imac-unknown-none-elf",
    "riscv64imac-unknown-nuttx-elf",
    "wasm32v1-none",
    "xtensa-esp32-espidf",
    "xtensa-esp32-none-elf",
    "xtensa-esp32s2-espidf",
    "xtensa-esp32s2-none-elf",
    "xtensa-esp32s3-espidf",
    "xtensa-esp32s3-none-elf",
];

fn main() {
    let targets = get_target_list();

    let minicore_path = run_make_support::source_root().join("tests/auxiliary/minicore.rs");

    regex_mod();

    for target in targets.lines() {
        if SKIPPED_TARGETS.iter().any(|&to_skip_target| target == to_skip_target) {
            continue;
        }

        // Run Clang's preprocessor for the relevant target, printing default macro definitions.
        let clang_output =
            clang().args(&["-E", "-dM", "-x", "c", "/dev/null", "-target", target]).run();

        if !clang_output.status().success() {
            continue;
        }

        let defines = String::from_utf8(clang_output.stdout()).expect("Invalid UTF-8");

        let minicore_content = rfs::read_to_string(&minicore_path);
        let mut rmake_content = format!(
            r#"
            #![no_std]
            #![no_core]
            #![feature(link_cfg)]
            #![allow(unused)]
            #![crate_type = "rlib"]

            /* begin minicore content */
            {minicore_content}
            /* end minicore content */

            #[path = "processed_mod.rs"]
            mod ffi;
            #[path = "tests.rs"]
            mod tests;
            "#
        );

        rmake_content.push_str(&format!(
            "
            const CLANG_C_CHAR_SIZE: usize = {};
            const CLANG_C_CHAR_SIGNED: bool = {};
            const CLANG_C_SHORT_SIZE: usize = {};
            const CLANG_C_INT_SIZE: usize = {};
            const CLANG_C_LONG_SIZE: usize = {};
            const CLANG_C_LONGLONG_SIZE: usize = {};
            const CLANG_C_FLOAT_SIZE: usize = {};
            const CLANG_C_DOUBLE_SIZE: usize = {};
            const CLANG_C_SIZE_T_SIZE: usize = {};
            const CLANG_C_PTRDIFF_T_SIZE: usize = {};
            ",
            parse_size(&defines, "CHAR"),
            char_is_signed(&defines),
            parse_size(&defines, "SHORT"),
            parse_size(&defines, "INT"),
            parse_size(&defines, "LONG"),
            parse_size(&defines, "LONG_LONG"),
            parse_size(&defines, "FLOAT"),
            parse_size(&defines, "DOUBLE"),
            parse_size(&defines, "SIZE_T"),
            parse_size(&defines, "PTRDIFF_T"),
        ));

        // Generate a target-specific rmake file.
        // If type misalignments occur,
        // generated rmake file name used to identify the failing target.
        let file_name = format!("{}_rmake.rs", target.replace("-", "_").replace(".", "_"));

        // Attempt to build the test file for the relevant target.
        // Tests use constant evaluation, so running is not necessary.
        rfs::write(&file_name, rmake_content);
        let rustc_output = rustc()
            .arg("-Zunstable-options")
            .arg("--emit=metadata")
            .arg("--target")
            .arg(target)
            .arg("-o-")
            .arg(&file_name)
            .run();
        rfs::remove_file(&file_name);
        if !rustc_output.status().success() {
            panic!("Failed for target {}", target);
        }
    }

    // Cleanup
    rfs::remove_file("processed_mod.rs");
}

/// Get a list of available targets for 'rustc'.
fn get_target_list() -> String {
    let completed_process = rustc().arg("--print").arg("target-list").run();
    String::from_utf8(completed_process.stdout()).expect("error not a string")
}

// Helper to parse size from clang defines
fn parse_size(defines: &str, type_name: &str) -> usize {
    let search_pattern = format!("__SIZEOF_{}__ ", type_name.to_uppercase());
    for line in defines.lines() {
        if line.contains(&search_pattern) {
            if let Some(size_str) = line.split_whitespace().last() {
                return size_str.parse().unwrap_or(0);
            }
        }
    }

    // Only allow CHAR to default to 1
    if type_name.to_uppercase() == "CHAR" {
        return 1;
    }

    panic!("Could not find size definition for type: {}", type_name);
}

fn char_is_signed(defines: &str) -> bool {
    !defines.lines().any(|line| line.contains("__CHAR_UNSIGNED__"))
}

/// Parse core/ffi/mod.rs to retrieve only necessary macros and type defines
fn regex_mod() {
    let mod_path = run_make_support::source_root().join("library/core/src/ffi/mod.rs");
    let mut content = rfs::read_to_string(&mod_path);

    //remove stability features #![unstable]
    let mut re = regex::Regex::new(r"#!?\[(un)?stable[^]]*?\]").unwrap();
    content = re.replace_all(&content, "").to_string();

    //remove doc features #[doc...]
    re = regex::Regex::new(r"#\[doc[^]]*?\]").unwrap();
    content = re.replace_all(&content, "").to_string();

    //remove lang feature #[lang...]
    re = regex::Regex::new(r"#\[lang[^]]*?\]").unwrap();
    content = re.replace_all(&content, "").to_string();

    //remove non inline modules
    re = regex::Regex::new(r".*mod.*;").unwrap();
    content = re.replace_all(&content, "").to_string();

    //remove use
    re = regex::Regex::new(r".*use.*;").unwrap();
    content = re.replace_all(&content, "").to_string();

    //remove fn fmt {...}
    re = regex::Regex::new(r"(?s)fn fmt.*?\{.*?\}").unwrap();
    content = re.replace_all(&content, "").to_string();

    //rmv impl fmt {...}
    re = regex::Regex::new(r"(?s)impl fmt::Debug for.*?\{.*?\}").unwrap();
    content = re.replace_all(&content, "").to_string();

    let file_name = "processed_mod.rs";

    rfs::create_file(&file_name);
    rfs::write(&file_name, content);
}
