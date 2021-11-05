#![feature(slice_partition_dedup)]
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;

use std::fs::File;
use std::io::Write;
use std::process::Command;

use clap::{App, Arg};
use intrinsic::Intrinsic;
use itertools::Itertools;
use rayon::prelude::*;
use types::TypeKind;

use crate::acle_csv_parser::get_acle_intrinsics;
use crate::argument::Argument;

mod acle_csv_parser;
mod argument;
mod intrinsic;
mod types;
mod values;

#[derive(Debug, PartialEq)]
pub enum Language {
    Rust,
    C,
}

fn gen_code_c(intrinsic: &Intrinsic, constraints: &[&Argument], name: String) -> String {
    if let Some((current, constraints)) = constraints.split_last() {
        let range = current
            .constraints
            .iter()
            .map(|c| c.to_range())
            .flat_map(|r| r.into_iter());

        range
            .map(|i| {
                format!(
                    r#"  {{
  {ty} {name} = {val};
{pass}
  }}"#,
                    name = current.name,
                    ty = current.ty.c_type(),
                    val = i,
                    pass = gen_code_c(intrinsic, constraints, format!("{}-{}", name, i))
                )
            })
            .collect()
    } else {
        (1..20)
            .map(|idx| intrinsic.generate_pass_c(idx, &name))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn generate_c_program(header_files: &[&str], intrinsic: &Intrinsic) -> String {
    let constraints = intrinsic
        .arguments
        .iter()
        .filter(|i| i.has_constraint())
        .collect_vec();

    format!(
        r#"{header_files}
#include <iostream>
#include <cstring>
#include <iomanip>
#include <sstream>
template<typename T1, typename T2> T1 cast(T2 x) {{
  static_assert(sizeof(T1) == sizeof(T2), "sizeof T1 and T2 must be the same");
  T1 ret = 0;
  memcpy(&ret, &x, sizeof(T1));
  return ret;
}}
std::ostream& operator<<(std::ostream& os, poly128_t value) {{
  std::stringstream temp;
  do {{
    int n = value % 10;
    value /= 10;
    temp << n;
  }} while (value != 0);
  std::string tempstr(temp.str());
  std::string res(tempstr.rbegin(), tempstr.rend());
  os << res;
  return os;
}}
int main(int argc, char **argv) {{
{passes}
    return 0;
}}"#,
        header_files = header_files
            .iter()
            .map(|header| format!("#include <{}>", header))
            .collect::<Vec<_>>()
            .join("\n"),
        passes = gen_code_c(intrinsic, constraints.as_slice(), Default::default()),
    )
}

fn gen_code_rust(intrinsic: &Intrinsic, constraints: &[&Argument], name: String) -> String {
    if let Some((current, constraints)) = constraints.split_last() {
        let range = current
            .constraints
            .iter()
            .map(|c| c.to_range())
            .flat_map(|r| r.into_iter());

        range
            .map(|i| {
                format!(
                    r#"  {{
    const {name}: {ty} = {val};
{pass}
  }}"#,
                    name = current.name,
                    ty = current.ty.rust_type(),
                    val = i,
                    pass = gen_code_rust(intrinsic, constraints, format!("{}-{}", name, i))
                )
            })
            .collect()
    } else {
        (1..20)
            .map(|idx| intrinsic.generate_pass_rust(idx, &name))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn generate_rust_program(intrinsic: &Intrinsic) -> String {
    let constraints = intrinsic
        .arguments
        .iter()
        .filter(|i| i.has_constraint())
        .collect_vec();

    format!(
        r#"#![feature(simd_ffi)]
#![feature(link_llvm_intrinsics)]
#![feature(stdsimd)]
#![allow(overflowing_literals)]
#![allow(non_upper_case_globals)]
use core_arch::arch::aarch64::*;

fn main() {{
{passes}
}}
"#,
        passes = gen_code_rust(intrinsic, &constraints, Default::default())
    )
}

fn compile_c(c_filename: &str, intrinsic: &Intrinsic, compiler: &str) -> bool {
    let flags = std::env::var("CPPFLAGS").unwrap_or("".into());

    let output = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "{cpp} {cppflags} {arch_flags} -Wno-narrowing -O2 -target {target} -o c_programs/{intrinsic} {filename}",
            target = "aarch64-unknown-linux-gnu",
            arch_flags = "-march=armv8.6-a+crypto+sha3+crc+dotprod",
            filename = c_filename,
            intrinsic = intrinsic.name,
            cpp = compiler,
            cppflags = flags,
        ))
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            true
        } else {
            let stderr = std::str::from_utf8(&output.stderr).unwrap_or("");
            if stderr.contains("error: use of undeclared identifier") {
                warn!("Skipping intrinsic due to no support: {}", intrinsic.name);
                true
            } else {
                error!(
                    "Failed to compile code for intrinsic: {}\n\nstdout:\n{}\n\nstderr:\n{}",
                    intrinsic.name,
                    std::str::from_utf8(&output.stdout).unwrap_or(""),
                    std::str::from_utf8(&output.stderr).unwrap_or("")
                );
                false
            }
        }
    } else {
        error!("Command failed: {:#?}", output);
        false
    }
}

fn build_c(intrinsics: &Vec<Intrinsic>, compiler: &str) -> bool {
    let _ = std::fs::create_dir("c_programs");
    intrinsics
        .par_iter()
        .map(|i| {
            let c_filename = format!(r#"c_programs/{}.cpp"#, i.name);
            let mut file = File::create(&c_filename).unwrap();

            let c_code = generate_c_program(&["arm_neon.h", "arm_acle.h"], &i);
            file.write_all(c_code.into_bytes().as_slice()).unwrap();
            compile_c(&c_filename, &i, compiler)
        })
        .find_any(|x| !x)
        .is_none()
}

fn build_rust(intrinsics: &Vec<Intrinsic>, toolchain: &str) -> bool {
    intrinsics.iter().for_each(|i| {
        let rust_dir = format!(r#"rust_programs/{}"#, i.name);
        let _ = std::fs::create_dir_all(&rust_dir);
        let rust_filename = format!(r#"{}/main.rs"#, rust_dir);
        let mut file = File::create(&rust_filename).unwrap();

        let c_code = generate_rust_program(&i);
        file.write_all(c_code.into_bytes().as_slice()).unwrap();
    });

    let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
    cargo
        .write_all(
            format!(
                r#"[package]
name = "intrinsic-test"
version = "{version}"
authors = ["{authors}"]
edition = "2018"
[workspace]
[dependencies]
core_arch = {{ path = "../crates/core_arch" }}
{binaries}"#,
                version = env!("CARGO_PKG_VERSION"),
                authors = env!("CARGO_PKG_AUTHORS"),
                binaries = intrinsics
                    .iter()
                    .map(|i| {
                        format!(
                            r#"[[bin]]
name = "{intrinsic}"
path = "{intrinsic}/main.rs""#,
                            intrinsic = i.name
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )
            .into_bytes()
            .as_slice(),
        )
        .unwrap();

    let output = Command::new("sh")
        .current_dir("rust_programs")
        .arg("-c")
        .arg(format!(
            "cargo {toolchain} build --release --target {target}",
            toolchain = toolchain,
            target = "aarch64-unknown-linux-gnu",
        ))
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            true
        } else {
            error!(
                "Failed to compile code for intrinsics\n\nstdout:\n{}\n\nstderr:\n{}",
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

fn main() {
    pretty_env_logger::init();

    let matches = App::new("Intrinsic test tool")
        .about("Generates Rust and C programs for intrinsics and compares the output")
        .arg(
            Arg::with_name("INPUT")
                .help("The input file containing the intrinsics")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("TOOLCHAIN")
                .takes_value(true)
                .long("toolchain")
                .help("The rust toolchain to use for building the rust code"),
        )
        .arg(
            Arg::with_name("CPPCOMPILER")
                .takes_value(true)
                .default_value("clang++")
                .long("cppcompiler")
                .help("The C++ compiler to use for compiling the c++ code"),
        )
        .arg(
            Arg::with_name("RUNNER")
                .takes_value(true)
                .long("runner")
                .help("Run the C programs under emulation with this command"),
        )
        .arg(
            Arg::with_name("SKIP")
                .takes_value(true)
                .long("skip")
                .help("Filename for a list of intrinsics to skip (one per line)"),
        )
        .get_matches();

    let filename = matches.value_of("INPUT").unwrap();
    let toolchain = matches
        .value_of("TOOLCHAIN")
        .map_or("".into(), |t| format!("+{}", t));

    let cpp_compiler = matches.value_of("CPPCOMPILER").unwrap();
    let c_runner = matches.value_of("RUNNER").unwrap_or("");
    let skip = if let Some(filename) = matches.value_of("SKIP") {
        let data = std::fs::read_to_string(&filename).expect("Failed to open file");
        data.lines().map(String::from).collect_vec()
    } else {
        Default::default()
    };

    let intrinsics = get_acle_intrinsics(filename);

    let mut intrinsics = intrinsics
        .into_iter()
        // Not sure how we would compare intrinsic that returns void.
        .filter(|i| i.results.kind() != TypeKind::Void)
        .filter(|i| i.results.kind() != TypeKind::BFloat)
        .filter(|i| !(i.results.kind() == TypeKind::Float && i.results.inner_size() == 16))
        .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
        .filter(|i| {
            !i.arguments
                .iter()
                .any(|a| a.ty.kind() == TypeKind::Float && a.ty.inner_size() == 16)
        })
        // Skip pointers for now, we would probably need to look at the return
        // type to work out how many elements we need to point to.
        .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
        .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
        .filter(|i| !skip.contains(&i.name))
        .collect::<Vec<_>>();
    intrinsics.dedup();

    if !build_c(&intrinsics, cpp_compiler) {
        std::process::exit(2);
    }

    if !build_rust(&intrinsics, &toolchain) {
        std::process::exit(3);
    }

    if !compare_outputs(&intrinsics, &toolchain, &c_runner) {
        std::process::exit(1)
    }
}

enum FailureReason {
    RunC(String),
    RunRust(String),
    Difference(String, String, String),
}

fn compare_outputs(intrinsics: &Vec<Intrinsic>, toolchain: &str, runner: &str) -> bool {
    let intrinsics = intrinsics
        .par_iter()
        .filter_map(|intrinsic| {
            let c = Command::new("sh")
                .arg("-c")
                .arg(format!(
                    "{runner} ./c_programs/{intrinsic}",
                    runner = runner,
                    intrinsic = intrinsic.name,
                ))
                .output();
            let rust = Command::new("sh")
                .current_dir("rust_programs")
                .arg("-c")
                .arg(format!(
                    "cargo {toolchain} run --release --target {target} --bin {intrinsic}",
                    intrinsic = intrinsic.name,
                    toolchain = toolchain,
                    target = "aarch64-unknown-linux-gnu",
                ))
                .output();

            let (c, rust) = match (c, rust) {
                (Ok(c), Ok(rust)) => (c, rust),
                a => panic!("{:#?}", a),
            };

            if !c.status.success() {
                error!("Failed to run C program for intrinsic {}", intrinsic.name);
                return Some(FailureReason::RunC(intrinsic.name.clone()));
            }

            if !rust.status.success() {
                error!(
                    "Failed to run rust program for intrinsic {}",
                    intrinsic.name
                );
                return Some(FailureReason::RunRust(intrinsic.name.clone()));
            }

            info!("Comparing intrinsic: {}", intrinsic.name);

            let c = std::str::from_utf8(&c.stdout)
                .unwrap()
                .to_lowercase()
                .replace("-nan", "nan");
            let rust = std::str::from_utf8(&rust.stdout)
                .unwrap()
                .to_lowercase()
                .replace("-nan", "nan");

            if c == rust {
                None
            } else {
                Some(FailureReason::Difference(intrinsic.name.clone(), c, rust))
            }
        })
        .collect::<Vec<_>>();

    intrinsics.iter().for_each(|reason| match reason {
        FailureReason::Difference(intrinsic, c, rust) => {
            println!("Difference for intrinsic: {}", intrinsic);
            let diff = diff::lines(c, rust);
            diff.iter().for_each(|diff| match diff {
                diff::Result::Left(c) => println!("C: {}", c),
                diff::Result::Right(rust) => println!("Rust: {}", rust),
                diff::Result::Both(_, _) => (),
            });
            println!("****************************************************************");
        }
        FailureReason::RunC(intrinsic) => {
            println!("Failed to run C program for intrinsic {}", intrinsic)
        }
        FailureReason::RunRust(intrinsic) => {
            println!("Failed to run rust program for intrinsic {}", intrinsic)
        }
    });
    println!("{} differences found", intrinsics.len());
    intrinsics.is_empty()
}
