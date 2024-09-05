//! Verify for each supported target that the data layout `rustc` is
//! configured to give to LLVM matches what Clang does.

// Only run once, on the host machine, because we're testing all targets here.
//@ force-host

// TODO: Only run this if Clang is available, and configured as a cross compiler.

use std::process::ExitCode;

use run_make_support::{clang, regex, rustc, serde_json};

/// Walk both layouts, potentially skipping some on the rustc layout if it
/// contains more information.
///
/// See <https://llvm.org/docs/LangRef.html#data-layout>
fn compare_layout(clang: &str, rustc: &str) -> Result<(), String> {
    let mut clang = clang.split("-").peekable();
    let mut rustc = rustc.split("-").peekable();
    while let Some(rustc_piece) = rustc.next() {
        // If the pieces match, keep going
        if clang.peek() == Some(&rustc_piece) {
            clang.next();
            continue;
        }

        // Function pointer alignment does not seem to be sent from Clang to LLVM IR?
        if rustc_piece.starts_with('F') {
            continue;
        }

        // Rust's i128 has a different alignment compared to older versions of Clang
        if rustc_piece == "i128:128" {
            continue;
        }

        // TODO: Rustc sets 32 bits as native layout in a lot of places that
        // Clang doesn't?
        if rustc_piece == "n32:64" && clang.peek() == Some(&"n64") {
            clang.next();
            continue;
        }

        return Err(format!("mismatch at {rustc_piece}"));
    }

    Ok(())
}

fn main() -> ExitCode {
    let all_target_specs =
        rustc().arg("-Zunstable-options").print("all-target-specs-json").run().stdout_utf8();

    let all_target_specs: serde_json::Value =
        serde_json::from_str(&all_target_specs).expect("parse rustc all-target-specs-json");
    let all_target_specs = all_target_specs.as_object().expect("targets to be an object");

    println!("checking {} targets", all_target_specs.len());

    let data_layout_regex = regex::Regex::new(r#"target datalayout = "(.*)""#).unwrap();

    let mut exit_code = ExitCode::SUCCESS;

    for (target_name, spec) in all_target_specs {
        // // Clang OOMs for some reason?
        // if target_name == "aarch64-unknown-illumos" {
        //     continue;
        // }

        // // Passing wrong LLVM target
        // if target_name.starts_with("aarch64-apple-visionos") {
        //     continue;
        // }

        // // Incorrect data layout detected by Clang
        // if target_name.ends_with("gnu_ilp32") {
        //     continue;
        // }

        // // Bare metal, not supported by all versions of Clang
        // if target_name.starts_with("xtensa") {
        //     continue;
        // }

        let llvm_target = spec["llvm-target"].as_str().expect("llvm target to be string");
        let rustc_data_layout = spec["data-layout"].as_str().expect("data layout to be string");

        let output = clang()
            .target(llvm_target)
            .no_stdlib()
            .arg("-S")
            .arg("-emit-llvm")
            .input("empty.c")
            .arg("-o")
            .arg("-")
            .run_unchecked();

        if !output.status().success() {
            exit_code = ExitCode::FAILURE;
            eprintln!("Clang failed compiling {target_name}:");
            eprintln!("{}\n\n", output.stderr_utf8());
            continue;
        }

        let llvm_ir = output.stdout_utf8();

        let caps = data_layout_regex
            .captures(&llvm_ir)
            .expect("could not find data layout in Clang's LLVM IR");
        let clang_data_layout = &caps[1];

        if let Err(err) = compare_layout(clang_data_layout, rustc_data_layout) {
            exit_code = ExitCode::FAILURE;
            eprintln!("mismatch on {target_name}:");
            eprintln!("  clang: {clang_data_layout}");
            eprintln!("  rustc: {rustc_data_layout}");
            eprintln!("  {err}");
            eprintln!("{}", output.stderr_utf8());
            eprintln!();
            continue;
        }
    }

    exit_code
}
