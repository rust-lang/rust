//@ needs-force-clang-based-tests
// This test checks that the clang defines for each target allign with the core ffi types defined in
// mod.rs. Therefore each rust target is queried and the clang defines for each target are read and
// compared to the core sizes to verify all types and sizes allign at buildtime.
//
// If this test fails because Rust adds a target that Clang does not support, this target should be
// added to SKIPPED_TARGETS.

use run_make_support::{clang, llvm_components_contain, regex, rfs, rustc};

// It is not possible to run the Rust test-suite on these targets.
const SKIPPED_TARGETS: &[&str] = &[
    "wasm32v1-none",
    "xtensa-esp32-espidf",
    "xtensa-esp32-none-elf",
    "xtensa-esp32s2-espidf",
    "xtensa-esp32s2-none-elf",
    "xtensa-esp32s3-espidf",
    "xtensa-esp32s3-none-elf",
    "csky-unknown-linux-gnuabiv2",
    "csky-unknown-linux-gnuabiv2hf",
];

/// Map from a Rust target to the Clang target if they are not the same.
const MAPPED_TARGETS: &[(&str, &str)] = &[
    ("aarch64-apple-ios-sim", "aarch64-apple-ios"),
    ("aarch64-apple-tvos-sim", "aarch64-apple-tvos"),
    ("aarch64-apple-visionos-sim", "aarch64-apple-visionos"),
    ("aarch64-apple-watchos-sim", "aarch64-apple-watchos"),
    ("x86_64-apple-watchos-sim", "x86_64-apple-watchos"),
    ("aarch64-pc-windows-gnullvm", "aarch64-pc-windows-gnu"),
    ("aarch64-unknown-linux-gnu_ilp32", "aarch64-unknown-linux-gnu"),
    ("aarch64-unknown-none-softfloat", "aarch64-unknown-none"),
    ("aarch64-unknown-nto-qnx700", "aarch64-unknown-nto-700"),
    ("aarch64-unknown-nto-qnx710", "aarch64-unknown-nto-710"),
    ("aarch64-unknown-uefi", "aarch64-unknown"),
    ("aarch64_be-unknown-linux-gnu_ilp32", "aarch64_be-unknown-linux-gnu"),
    ("armv5te-unknown-linux-uclibceabi", "armv5te-unknown-linux"),
    ("armv7-sony-vita-newlibeabihf", "armv7-sony-vita"),
    ("armv7-unknown-linux-uclibceabi", "armv7-unknown-linux"),
    ("armv7-unknown-linux-uclibceabihf", "armv7-unknown-linux"),
    ("avr-unknown-gnu-atmega328", "avr-unknown-gnu"),
    ("csky-unknown-linux-gnuabiv2", "csky-unknown-linux-gnu"),
    ("i586-pc-nto-qnx700", "i586-pc-nto-700"),
    ("i686-pc-windows-gnullvm", "i686-pc-windows-gnu"),
    ("i686-unknown-uefi", "i686-unknown"),
    ("loongarch64-unknown-none-softfloat", "loongarch64-unknown-none"),
    ("mips-unknown-linux-uclibc", "mips-unknown-linux"),
    ("mipsel-unknown-linux-uclibc", "mipsel-unknown-linux"),
    ("powerpc-unknown-linux-gnuspe", "powerpc-unknown-linux-gnu"),
    ("powerpc-unknown-linux-muslspe", "powerpc-unknown-linux-musl"),
    ("powerpc-wrs-vxworks-spe", "powerpc-wrs-vxworks"),
    ("x86_64-fortanix-unknown-sgx", "x86_64-fortanix-unknown"),
    ("x86_64-pc-nto-qnx710", "x86_64-pc-nto-710"),
    ("x86_64-pc-windows-gnullvm", "x86_64-pc-windows-gnu"),
    ("x86_64-unknown-l4re-uclibc", "x86_64-unknown-l4re"),
];

fn main() {
    let targets = get_target_list();

    let minicore_path = run_make_support::source_root().join("tests/auxiliary/minicore.rs");

    preprocess_core_ffi();

    for target in targets.lines() {
        if SKIPPED_TARGETS.iter().any(|&to_skip_target| target == to_skip_target) {
            continue;
        }

        // Map the Rust target string to a Clang target string if needed
        // Also replace riscv with necessary replacements to match clang
        // If neither just use target string
        let ctarget = MAPPED_TARGETS
            .iter()
            .find(|(rtarget, _)| *rtarget == target)
            .map(|(_, ctarget)| ctarget.to_string())
            .unwrap_or_else(|| {
                if target.starts_with("riscv") {
                    target
                        .replace("imac-", "-")
                        .replace("gc-", "-")
                        .replace("imafc-", "-")
                        .replace("imc-", "-")
                        .replace("ima-", "-")
                        .replace("im-", "-")
                        .replace("emc-", "-")
                        .replace("em-", "-")
                        .replace("e-", "-")
                        .replace("i-", "-")
                } else {
                    target.to_string()
                }
            });

        // Run Clang's preprocessor for the relevant target, printing default macro definitions.
        let clang_output =
            clang().args(&["-E", "-dM", "-x", "c", "/dev/null", "-target", &ctarget]).run();

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
fn preprocess_core_ffi() {
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

    rfs::write(&file_name, content);
}
