use itertools::Itertools;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::process::Command;

pub fn generate_rust_program(
    notices: &str,
    configurations: &str,
    arch_definition: &str,
    arglists: &str,
    passes: &str,
) -> String {
    format!(
        r#"{notices}#![feature(simd_ffi)]
#![feature(link_llvm_intrinsics)]
#![feature(f16)]
{configurations}
#![allow(non_upper_case_globals)]
use core_arch::arch::{arch_definition}::*;

fn main() {{
{arglists}
{passes}
}}
"#,
    )
}

pub fn compile_rust(
    binaries: &[&str],
    toolchain: Option<&str>,
    target: &str,
    linker: Option<&str>,
) -> bool {
    let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
    cargo
        .write_all(
            format!(
                r#"[package]
name = "intrinsic-test-programs"
version = "{version}"
authors = [{authors}]
license = "{license}"
edition = "2018"
[workspace]
[dependencies]
core_arch = {{ path = "../crates/core_arch" }}
{binaries}"#,
                version = env!("CARGO_PKG_VERSION"),
                authors = env!("CARGO_PKG_AUTHORS")
                    .split(":")
                    .format_with(", ", |author, fmt| fmt(&format_args!("\"{author}\""))),
                license = env!("CARGO_PKG_LICENSE"),
                binaries = binaries
                    .iter()
                    .map(|binary| {
                        format!(
                            r#"[[bin]]
name = "{binary}"
path = "{binary}/main.rs""#,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )
            .into_bytes()
            .as_slice(),
        )
        .unwrap();

    let toolchain = match toolchain {
        None => return true,
        Some(t) => t,
    };

    /* If there has been a linker explicitly set from the command line then
     * we want to set it via setting it in the RUSTFLAGS*/

    let cargo_command = format!(
        "cargo {toolchain} build --target {target} --release",
        toolchain = toolchain,
        target = target
    );

    let mut command = Command::new("sh");
    command
        .current_dir("rust_programs")
        .arg("-c")
        .arg(cargo_command);

    let mut rust_flags = "-Cdebuginfo=0".to_string();
    if let Some(linker) = linker {
        rust_flags.push_str(" -C linker=");
        rust_flags.push_str(linker);
        rust_flags.push_str(" -C link-args=-static");

        command.env("CPPFLAGS", "-fuse-ld=lld");
    }

    command.env("RUSTFLAGS", rust_flags);
    let output = command.output();

    if let Ok(output) = output {
        if output.status.success() {
            true
        } else {
            error!(
                "Failed to compile code for rust intrinsics\n\nstdout:\n{}\n\nstderr:\n{}",
                std::str::from_utf8(&output.stdout).unwrap_or(""),
                std::str::from_utf8(&output.stderr).unwrap_or("")
            );
            false
        }
    } else {
        error!("Command failed: {:#?}", output);
        false
    }
}

pub fn create_rust_files(identifiers: &Vec<String>) -> BTreeMap<&String, File> {
    identifiers
        .par_iter()
        .map(|identifier| {
            let rust_dir = format!(r#"rust_programs/{}"#, identifier);
            let _ = std::fs::create_dir_all(&rust_dir);
            let rust_filename = format!(r#"{rust_dir}/main.rs"#);
            let file = File::create(&rust_filename).unwrap();

            (identifier, file)
        })
        .collect::<BTreeMap<&String, File>>()
}
