#![feature(slice_partition_dedup)]
#[macro_use]
extern crate log;

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use intrinsic::Intrinsic;
use itertools::Itertools;
use rayon::prelude::*;
use types::TypeKind;

use crate::argument::Argument;
use crate::format::Indentation;
use crate::json_parser::get_neon_intrinsics;

mod argument;
mod format;
mod intrinsic;
mod json_parser;
mod types;
mod values;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;

#[derive(Debug, PartialEq)]
pub enum Language {
    Rust,
    C,
}

fn gen_code_c(
    indentation: Indentation,
    intrinsic: &Intrinsic,
    constraints: &[&Argument],
    name: String,
    target: &str,
) -> String {
    if let Some((current, constraints)) = constraints.split_last() {
        let range = current
            .constraints
            .iter()
            .map(|c| c.to_range())
            .flat_map(|r| r.into_iter());

        let body_indentation = indentation.nested();
        range
            .map(|i| {
                format!(
                    "{indentation}{{\n\
                        {body_indentation}{ty} {name} = {val};\n\
                        {pass}\n\
                    {indentation}}}",
                    name = current.name,
                    ty = current.ty.c_type(),
                    val = i,
                    pass = gen_code_c(
                        body_indentation,
                        intrinsic,
                        constraints,
                        format!("{name}-{i}"),
                        target,
                    )
                )
            })
            .join("\n")
    } else {
        intrinsic.generate_loop_c(indentation, &name, PASSES, target)
    }
}

fn generate_c_program(
    notices: &str,
    header_files: &[&str],
    intrinsic: &Intrinsic,
    target: &str,
) -> String {
    let constraints = intrinsic
        .arguments
        .iter()
        .filter(|i| i.has_constraint())
        .collect_vec();

    let indentation = Indentation::default();
    format!(
        r#"{notices}{header_files}
#include <iostream>
#include <cstring>
#include <iomanip>
#include <sstream>

template<typename T1, typename T2> T1 cast(T2 x) {{
  static_assert(sizeof(T1) == sizeof(T2), "sizeof T1 and T2 must be the same");
  T1 ret{{}};
  memcpy(&ret, &x, sizeof(T1));
  return ret;
}}

#ifdef __aarch64__
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
#endif

std::ostream& operator<<(std::ostream& os, float16_t value) {{
    uint16_t temp = 0;
    memcpy(&temp, &value, sizeof(float16_t));
    std::stringstream ss;
    ss << "0x" << std::setfill('0') << std::setw(4) << std::hex << temp;
    os << ss.str();
    return os;
}}

{arglists}

int main(int argc, char **argv) {{
{passes}
    return 0;
}}"#,
        header_files = header_files
            .iter()
            .map(|header| format!("#include <{header}>"))
            .collect::<Vec<_>>()
            .join("\n"),
        arglists = intrinsic.arguments.gen_arglists_c(indentation, PASSES),
        passes = gen_code_c(
            indentation.nested(),
            intrinsic,
            constraints.as_slice(),
            Default::default(),
            target,
        ),
    )
}

fn gen_code_rust(
    indentation: Indentation,
    intrinsic: &Intrinsic,
    constraints: &[&Argument],
    name: String,
) -> String {
    if let Some((current, constraints)) = constraints.split_last() {
        let range = current
            .constraints
            .iter()
            .map(|c| c.to_range())
            .flat_map(|r| r.into_iter());

        let body_indentation = indentation.nested();
        range
            .map(|i| {
                format!(
                    "{indentation}{{\n\
                        {body_indentation}const {name}: {ty} = {val};\n\
                        {pass}\n\
                    {indentation}}}",
                    name = current.name,
                    ty = current.ty.rust_type(),
                    val = i,
                    pass = gen_code_rust(
                        body_indentation,
                        intrinsic,
                        constraints,
                        format!("{name}-{i}")
                    )
                )
            })
            .join("\n")
    } else {
        intrinsic.generate_loop_rust(indentation, &name, PASSES)
    }
}

fn generate_rust_program(notices: &str, intrinsic: &Intrinsic, target: &str) -> String {
    let constraints = intrinsic
        .arguments
        .iter()
        .filter(|i| i.has_constraint())
        .collect_vec();

    let indentation = Indentation::default();
    format!(
        r#"{notices}#![feature(simd_ffi)]
#![feature(link_llvm_intrinsics)]
#![feature(f16)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_neon_intrinsics))]
#![cfg_attr(target_arch = "arm", feature(stdarch_aarch32_crc32))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_fcma))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_dotprod))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_i8mm))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_sha3))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_sm4))]
#![cfg_attr(any(target_arch = "aarch64", target_arch = "arm64ec"), feature(stdarch_neon_ftts))]
#![feature(stdarch_neon_f16)]
#![allow(non_upper_case_globals)]
use core_arch::arch::{target_arch}::*;

fn main() {{
{arglists}
{passes}
}}
"#,
        target_arch = if target.contains("v7") {
            "arm"
        } else {
            "aarch64"
        },
        arglists = intrinsic
            .arguments
            .gen_arglists_rust(indentation.nested(), PASSES),
        passes = gen_code_rust(
            indentation.nested(),
            intrinsic,
            &constraints,
            Default::default()
        )
    )
}

fn compile_c(
    c_filename: &str,
    intrinsic: &Intrinsic,
    compiler: &str,
    target: &str,
    cxx_toolchain_dir: Option<&str>,
) -> bool {
    let flags = std::env::var("CPPFLAGS").unwrap_or("".into());
    let arch_flags = if target.contains("v7") {
        "-march=armv8.6-a+crypto+crc+dotprod+fp16"
    } else {
        "-march=armv8.6-a+crypto+sha3+crc+dotprod+fp16+faminmax+lut"
    };

    let intrinsic_name = &intrinsic.name;

    let compiler_command = if target == "aarch64_be-unknown-linux-gnu" {
        let Some(cxx_toolchain_dir) = cxx_toolchain_dir else {
            panic!(
                "When setting `--target aarch64_be-unknown-linux-gnu` the C++ compilers toolchain directory must be set with `--cxx-toolchain-dir <dest>`"
            );
        };

        /* clang++ cannot link an aarch64_be object file, so we invoke
         * aarch64_be-unknown-linux-gnu's C++ linker. This ensures that we
         * are testing the intrinsics against LLVM.
         *
         * Note: setting `--sysroot=<...>` which is the obvious thing to do
         * does not work as it gets caught up with `#include_next <stdlib.h>`
         * not existing... */
        format!(
            "{compiler} {flags} {arch_flags} \
            -ffp-contract=off \
            -Wno-narrowing \
            -O2 \
            --target=aarch64_be-unknown-linux-gnu \
            -I{cxx_toolchain_dir}/include \
            -I{cxx_toolchain_dir}/aarch64_be-none-linux-gnu/include \
            -I{cxx_toolchain_dir}/aarch64_be-none-linux-gnu/include/c++/14.2.1 \
            -I{cxx_toolchain_dir}/aarch64_be-none-linux-gnu/include/c++/14.2.1/aarch64_be-none-linux-gnu \
            -I{cxx_toolchain_dir}/aarch64_be-none-linux-gnu/include/c++/14.2.1/backward \
            -I{cxx_toolchain_dir}/aarch64_be-none-linux-gnu/libc/usr/include \
            -c {c_filename} \
            -o c_programs/{intrinsic_name}.o && \
            {cxx_toolchain_dir}/bin/aarch64_be-none-linux-gnu-g++ c_programs/{intrinsic_name}.o -o c_programs/{intrinsic_name} && \
            rm c_programs/{intrinsic_name}.o",
        )
    } else {
        // -ffp-contract=off emulates Rust's approach of not fusing separate mul-add operations
        let base_compiler_command = format!(
            "{compiler} {flags} {arch_flags} -o c_programs/{intrinsic_name} {c_filename} -ffp-contract=off -Wno-narrowing -O2"
        );

        /* `-target` can be passed to some c++ compilers, however if we want to
         *   use a c++ compiler does not support this flag we do not want to pass
         *   the flag. */
        if compiler.contains("clang") {
            format!("{base_compiler_command} -target {target}")
        } else {
            format!("{base_compiler_command} -flax-vector-conversions")
        }
    };

    let output = Command::new("sh").arg("-c").arg(compiler_command).output();
    if let Ok(output) = output {
        if output.status.success() {
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
    } else {
        error!("Command failed: {:#?}", output);
        false
    }
}

fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from a JSON specification, published under the same license as the
{line_prefix}`intrinsic-test` crate.\n
"
    )
}

fn build_c(
    notices: &str,
    intrinsics: &Vec<Intrinsic>,
    compiler: Option<&str>,
    target: &str,
    cxx_toolchain_dir: Option<&str>,
) -> bool {
    let _ = std::fs::create_dir("c_programs");
    intrinsics
        .par_iter()
        .map(|i| {
            let c_filename = format!(r#"c_programs/{}.cpp"#, i.name);
            let mut file = File::create(&c_filename).unwrap();

            let c_code = generate_c_program(
                notices,
                &["arm_neon.h", "arm_acle.h", "arm_fp16.h"],
                i,
                target,
            );
            file.write_all(c_code.into_bytes().as_slice()).unwrap();
            match compiler {
                None => true,
                Some(compiler) => compile_c(&c_filename, i, compiler, target, cxx_toolchain_dir),
            }
        })
        .find_any(|x| !x)
        .is_none()
}

fn build_rust(
    notices: &str,
    intrinsics: &[Intrinsic],
    toolchain: Option<&str>,
    target: &str,
    linker: Option<&str>,
) -> bool {
    intrinsics.iter().for_each(|i| {
        let rust_dir = format!(r#"rust_programs/{}"#, i.name);
        let _ = std::fs::create_dir_all(&rust_dir);
        let rust_filename = format!(r#"{rust_dir}/main.rs"#);
        let mut file = File::create(&rust_filename).unwrap();

        let c_code = generate_rust_program(notices, i, target);
        file.write_all(c_code.into_bytes().as_slice()).unwrap();
    });

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

/// Intrinsic test tool
#[derive(clap::Parser)]
#[command(
    name = "Intrinsic test tool",
    about = "Generates Rust and C programs for intrinsics and compares the output"
)]
struct Cli {
    /// The input file containing the intrinsics
    input: PathBuf,

    /// The rust toolchain to use for building the rust code
    #[arg(long)]
    toolchain: Option<String>,

    /// The C++ compiler to use for compiling the c++ code
    #[arg(long, default_value_t = String::from("clang++"))]
    cppcompiler: String,

    /// Run the C programs under emulation with this command
    #[arg(long)]
    runner: Option<String>,

    /// Filename for a list of intrinsics to skip (one per line)
    #[arg(long)]
    skip: Option<PathBuf>,

    /// Regenerate test programs, but don't build or run them
    #[arg(long)]
    generate_only: bool,

    /// Pass a target the test suite
    #[arg(long, default_value_t = String::from("aarch64-unknown-linux-gnu"))]
    target: String,

    /// Set the linker
    #[arg(long)]
    linker: Option<String>,

    /// Set the sysroot for the C++ compiler
    #[arg(long)]
    cxx_toolchain_dir: Option<String>,
}

fn main() {
    pretty_env_logger::init();

    let args: Cli = clap::Parser::parse();

    let filename = args.input;
    let c_runner = args.runner.unwrap_or_default();
    let target: &str = args.target.as_str();
    let linker = args.linker.as_deref();
    let cxx_toolchain_dir = args.cxx_toolchain_dir;

    let skip = if let Some(filename) = args.skip {
        let data = std::fs::read_to_string(&filename).expect("Failed to open file");
        data.lines()
            .map(str::trim)
            .filter(|s| !s.contains('#'))
            .map(String::from)
            .collect_vec()
    } else {
        Default::default()
    };
    let a32 = target.contains("v7");
    let mut intrinsics = get_neon_intrinsics(&filename).expect("Error parsing input file");

    intrinsics.sort_by(|a, b| a.name.cmp(&b.name));

    let mut intrinsics = intrinsics
        .into_iter()
        // Not sure how we would compare intrinsic that returns void.
        .filter(|i| i.results.kind() != TypeKind::Void)
        .filter(|i| i.results.kind() != TypeKind::BFloat)
        .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
        // Skip pointers for now, we would probably need to look at the return
        // type to work out how many elements we need to point to.
        .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
        .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
        .filter(|i| !skip.contains(&i.name))
        .filter(|i| !(a32 && i.a64_only))
        .collect::<Vec<_>>();
    intrinsics.dedup();

    let (toolchain, cpp_compiler) = if args.generate_only {
        (None, None)
    } else {
        (
            Some(args.toolchain.map_or_else(String::new, |t| format!("+{t}"))),
            Some(args.cppcompiler),
        )
    };

    let notices = build_notices("// ");

    if !build_c(
        &notices,
        &intrinsics,
        cpp_compiler.as_deref(),
        target,
        cxx_toolchain_dir.as_deref(),
    ) {
        std::process::exit(2);
    }

    if !build_rust(&notices, &intrinsics, toolchain.as_deref(), target, linker) {
        std::process::exit(3);
    }

    if let Some(ref toolchain) = toolchain {
        if !compare_outputs(&intrinsics, toolchain, &c_runner, target) {
            std::process::exit(1)
        }
    }
}

enum FailureReason {
    RunC(String),
    RunRust(String),
    Difference(String, String, String),
}

fn compare_outputs(
    intrinsics: &Vec<Intrinsic>,
    toolchain: &str,
    runner: &str,
    target: &str,
) -> bool {
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

            let rust = if target != "aarch64_be-unknown-linux-gnu" {
                Command::new("sh")
                    .current_dir("rust_programs")
                    .arg("-c")
                    .arg(format!(
                        "cargo {toolchain} run --target {target} --bin {intrinsic} --release",
                        intrinsic = intrinsic.name,
                        toolchain = toolchain,
                        target = target
                    ))
                    .env("RUSTFLAGS", "-Cdebuginfo=0")
                    .output()
            } else {
                Command::new("sh")
                    .arg("-c")
                    .arg(format!(
                        "{runner} ./rust_programs/target/{target}/release/{intrinsic}",
                        runner = runner,
                        target = target,
                        intrinsic = intrinsic.name,
                    ))
                    .output()
            };

            let (c, rust) = match (c, rust) {
                (Ok(c), Ok(rust)) => (c, rust),
                a => panic!("{a:#?}"),
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
            println!("Difference for intrinsic: {intrinsic}");
            let diff = diff::lines(c, rust);
            diff.iter().for_each(|diff| match diff {
                diff::Result::Left(c) => println!("C: {c}"),
                diff::Result::Right(rust) => println!("Rust: {rust}"),
                diff::Result::Both(_, _) => (),
            });
            println!("****************************************************************");
        }
        FailureReason::RunC(intrinsic) => {
            println!("Failed to run C program for intrinsic {intrinsic}")
        }
        FailureReason::RunRust(intrinsic) => {
            println!("Failed to run rust program for intrinsic {intrinsic}")
        }
    });
    println!("{} differences found", intrinsics.len());
    intrinsics.is_empty()
}
