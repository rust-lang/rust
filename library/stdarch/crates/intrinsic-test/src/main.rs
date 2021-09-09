#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;

use std::fs::File;
use std::io::Write;
use std::process::Command;

use clap::{App, Arg};
use intrinsic::Intrinsic;
use rayon::prelude::*;
use types::TypeKind;

mod argument;
mod intrinsic;
mod types;
mod values;

#[derive(Debug, PartialEq)]
pub enum Language {
    Rust,
    C,
}

fn generate_c_program(header_files: &[&str], intrinsic: &Intrinsic) -> String {
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
        passes = (1..20)
            .map(|idx| intrinsic.generate_pass_c(idx))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn generate_rust_program(intrinsic: &Intrinsic) -> String {
    format!(
        r#"#![feature(simd_ffi)]
#![feature(link_llvm_intrinsics)]
#![feature(stdsimd)]
#![allow(overflowing_literals)]
use core_arch::arch::aarch64::*;

fn main() {{
{passes}
}}
"#,
        passes = (1..20)
            .map(|idx| intrinsic.generate_pass_rust(idx))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn compile_c(c_filename: &str, intrinsic: &Intrinsic, compiler: &str) -> bool {
    let flags = std::env::var("CPPFLAGS").unwrap_or("".into());

    let output = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "{cpp} {cppflags} {arch_flags} -Wno-narrowing -O2 -target {target} -o c_programs/{intrinsic} {filename}",
            target = "aarch64-unknown-linux-gnu",
            arch_flags = "-march=armv8-a+crypto+crc",
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
        .get_matches();

    let filename = matches.value_of("INPUT").unwrap();
    let toolchain = matches
        .value_of("TOOLCHAIN")
        .map_or("".into(), |t| format!("+{}", t));

    let cpp_compiler = matches.value_of("CPPCOMPILER").unwrap();
    let c_runner = matches.value_of("RUNNER").unwrap_or("");
    let mut csv_reader = csv::Reader::from_reader(std::fs::File::open(filename).unwrap());

    let mut intrinsics = csv_reader
        .deserialize()
        .filter_map(|x| -> Option<Intrinsic> {
            debug!("Processing {:#?}", x);
            match x {
                Ok(a) => Some(a),
                e => {
                    error!("{:#?}", e);
                    None
                }
            }
        })
        // Only perform the test for intrinsics that are enabled...
        .filter(|i| i.enabled)
        // Not sure how we would compare intrinsic that returns void.
        .filter(|i| i.results.kind() != TypeKind::Void)
        .filter(|i| i.results.kind() != TypeKind::BFloat)
        .filter(|i| !(i.results.kind() == TypeKind::Float && i.results.inner_size() == 16))
        .filter(|i| {
            i.arguments
                .iter()
                .find(|a| a.ty.kind() == TypeKind::BFloat)
                .is_none()
        })
        .filter(|i| {
            i.arguments
                .iter()
                .find(|a| a.ty.kind() == TypeKind::Float && a.ty.inner_size() == 16)
                .is_none()
        })
        // Skip pointers for now, we would probably need to look at the return
        // type to work out how many elements we need to point to.
        .filter(|i| i.arguments.iter().find(|a| a.is_ptr()).is_none())
        // intrinsics with a lane parameter have constraints, deal with them later.
        .filter(|i| {
            i.arguments
                .iter()
                .find(|a| a.name.starts_with("lane"))
                .is_none()
        })
        .filter(|i| i.arguments.iter().find(|a| a.name == "n").is_none())
        // clang-12 fails to compile this intrinsic due to an error.
        // fatal error: error in backend: Cannot select: 0x2b99c30: i64 = AArch64ISD::VSHL Constant:i64<1>, Constant:i32<1>
        // 0x2b9a520: i64 = Constant<1>
        // 0x2b9a248: i32 = Constant<1>
        .filter(|i| !["vshld_s64", "vshld_u64"].contains(&i.name.as_str()))
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
