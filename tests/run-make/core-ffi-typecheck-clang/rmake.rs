//@ needs-force-clang-based-tests

use run_make_support::{clang, regex, rfs, rustc};

const SKIPPED_TARGETS: &[&str] = &[
    "riscv",  //error: unknown target triple 'riscv32e-unknown-none-elf'
    "wasm",   //error: unknown target triple 'wasm32v1-none'
    "xtensa", //error: unknown target triple 'xtensa-esp32-espidf'
];

fn main() {
    let targets = get_target_list();

    let minicore_path = run_make_support::source_root().join("tests/auxiliary/minicore.rs");

    regex_mod();

    for target in targets.lines() {
        if SKIPPED_TARGETS.iter().any(|prefix| target.starts_with(prefix)) {
            continue;
        }

        let clang_output =
            clang().args(&["-E", "-dM", "-x", "c", "/dev/null", "-target", target]).run();

        let defines = String::from_utf8(clang_output.stdout()).expect("Invalid UTF-8");

        let minicore_content = rfs::read_to_string(&minicore_path);
        let mut rmake_content = format!(
            r#"
            #![no_std]
            #![no_core]
            #![feature(link_cfg)]
            #![allow(unused)]
            #![crate_type = "rlib"]
            {}
            #[path = "processed_mod.rs"]
            mod ffi;
            #[path = "tests.rs"]
            mod tests;
            "#,
            minicore_content
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
            ",
            parse_size(&defines, "CHAR"),
            parse_signed(&defines, "CHAR"),
            parse_size(&defines, "SHORT"),
            parse_size(&defines, "INT"),
            parse_size(&defines, "LONG"),
            parse_size(&defines, "LONG_LONG"),
            parse_size(&defines, "FLOAT"),
            parse_size(&defines, "DOUBLE"),
        ));

        // Write to target-specific rmake file
        let mut file_name = format!("{}_rmake.rs", target.replace("-", "_"));

        if target.starts_with("thumbv8m") {
            file_name = String::from("thumbv8m_rmake.rs");
        }

        rfs::create_file(&file_name);
        rfs::write(&file_name, rmake_content);
        let rustc_output = rustc()
            .arg("-Zunstable-options")
            .arg("--emit=metadata")
            .arg("--target")
            .arg(target)
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

// Helper to parse signedness from clang defines
fn parse_signed(defines: &str, type_name: &str) -> bool {
    match type_name.to_uppercase().as_str() {
        "CHAR" => {
            // Check if char is explicitly unsigned
            !defines.lines().any(|line| line.contains("__CHAR_UNSIGNED__"))
        }
        _ => true,
    }
}

// Parse core/ffi/mod.rs to retrieve only necessary macros and type defines
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

    let file_name = format!("processed_mod.rs");

    rfs::create_file(&file_name);
    rfs::write(&file_name, content);
}
