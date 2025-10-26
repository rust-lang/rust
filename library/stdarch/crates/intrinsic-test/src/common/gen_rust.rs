use itertools::Itertools;
use std::process::Command;

use crate::common::intrinsic::Intrinsic;

use super::indentation::Indentation;
use super::intrinsic::format_f16_return_value;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;

fn write_cargo_toml_header(w: &mut impl std::io::Write, name: &str) -> std::io::Result<()> {
    writeln!(
        w,
        concat!(
            "[package]\n",
            "name = \"{name}\"\n",
            "version = \"{version}\"\n",
            "authors = [{authors}]\n",
            "license = \"{license}\"\n",
            "edition = \"2018\"\n",
        ),
        name = name,
        version = env!("CARGO_PKG_VERSION"),
        authors = env!("CARGO_PKG_AUTHORS")
            .split(":")
            .format_with(", ", |author, fmt| fmt(&format_args!("\"{author}\""))),
        license = env!("CARGO_PKG_LICENSE"),
    )
}

pub fn write_bin_cargo_toml(
    w: &mut impl std::io::Write,
    module_count: usize,
) -> std::io::Result<()> {
    write_cargo_toml_header(w, "intrinsic-test-programs")?;

    writeln!(w, "[dependencies]")?;

    for i in 0..module_count {
        writeln!(w, "mod_{i} = {{ path = \"mod_{i}/\" }}")?;
    }

    Ok(())
}

pub fn write_lib_cargo_toml(w: &mut impl std::io::Write, name: &str) -> std::io::Result<()> {
    write_cargo_toml_header(w, name)?;

    writeln!(w, "[dependencies]")?;
    writeln!(w, "core_arch = {{ path = \"../../crates/core_arch\" }}")?;

    Ok(())
}

pub fn write_main_rs<'a>(
    w: &mut impl std::io::Write,
    chunk_count: usize,
    cfg: &str,
    definitions: &str,
    intrinsics: impl Iterator<Item = &'a str> + Clone,
) -> std::io::Result<()> {
    writeln!(w, "#![feature(simd_ffi)]")?;
    writeln!(w, "#![feature(f16)]")?;
    writeln!(w, "#![allow(unused)]")?;

    // Cargo will spam the logs if these warnings are not silenced.
    writeln!(w, "#![allow(non_upper_case_globals)]")?;
    writeln!(w, "#![allow(non_camel_case_types)]")?;
    writeln!(w, "#![allow(non_snake_case)]")?;

    writeln!(w, "{cfg}")?;
    writeln!(w, "{definitions}")?;

    for module in 0..chunk_count {
        writeln!(w, "use mod_{module}::*;")?;
    }

    writeln!(w, "fn main() {{")?;

    writeln!(w, "    match std::env::args().nth(1).unwrap().as_str() {{")?;

    for binary in intrinsics {
        writeln!(w, "        \"{binary}\" => run_{binary}(),")?;
    }

    writeln!(
        w,
        "        other => panic!(\"unknown intrinsic `{{}}`\", other),"
    )?;

    writeln!(w, "    }}")?;
    writeln!(w, "}}")?;

    Ok(())
}

pub fn write_lib_rs<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    notice: &str,
    cfg: &str,
    definitions: &str,
    intrinsics: &[Intrinsic<T>],
) -> std::io::Result<()> {
    write!(w, "{notice}")?;

    writeln!(w, "#![feature(simd_ffi)]")?;
    writeln!(w, "#![feature(f16)]")?;
    writeln!(w, "#![allow(unused)]")?;

    // Cargo will spam the logs if these warnings are not silenced.
    writeln!(w, "#![allow(non_upper_case_globals)]")?;
    writeln!(w, "#![allow(non_camel_case_types)]")?;
    writeln!(w, "#![allow(non_snake_case)]")?;

    writeln!(w, "{cfg}")?;

    writeln!(w, "{definitions}")?;

    for intrinsic in intrinsics {
        crate::common::gen_rust::create_rust_test_module(w, intrinsic)?;
    }

    Ok(())
}

pub fn compile_rust_programs(toolchain: Option<&str>, target: &str, linker: Option<&str>) -> bool {
    /* If there has been a linker explicitly set from the command line then
     * we want to set it via setting it in the RUSTFLAGS*/

    // This is done because `toolchain` is None when
    // the --generate-only flag is passed
    if toolchain.is_none() {
        return true;
    }

    trace!("Building cargo command");

    let mut cargo_command = Command::new("cargo");
    cargo_command.current_dir("rust_programs");

    // Do not use the target directory of the workspace please.
    cargo_command.env("CARGO_TARGET_DIR", "target");

    if toolchain.is_some_and(|val| !val.is_empty()) {
        cargo_command.arg(toolchain.unwrap());
    }
    cargo_command.args(["build", "--target", target, "--release"]);

    let mut rust_flags = "-Cdebuginfo=0".to_string();
    if let Some(linker) = linker {
        rust_flags.push_str(" -C linker=");
        rust_flags.push_str(linker);
        rust_flags.push_str(" -C link-args=-static");

        cargo_command.env("CPPFLAGS", "-fuse-ld=lld");
    }

    cargo_command.env("RUSTFLAGS", rust_flags);

    trace!("running cargo");

    if log::log_enabled!(log::Level::Trace) {
        cargo_command.stdout(std::process::Stdio::inherit());
        cargo_command.stderr(std::process::Stdio::inherit());
    }

    let output = cargo_command.output();
    trace!("cargo is done");

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
        error!("Command failed: {output:#?}");
        false
    }
}

pub fn generate_rust_test_loop<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
    indentation: Indentation,
    specializations: &[Vec<u8>],
    passes: u32,
) -> std::io::Result<()> {
    let intrinsic_name = &intrinsic.name;

    // Each function (and each specialization) has its own type. Erase that type with a cast.
    let mut coerce = String::from("unsafe fn(");
    for _ in intrinsic.arguments.iter().filter(|a| !a.has_constraint()) {
        coerce += "_, ";
    }
    coerce += ") -> _";

    match specializations {
        [] => {
            writeln!(w, "    let specializations = [(\"\", {intrinsic_name})];")?;
        }
        [const_args] if const_args.is_empty() => {
            writeln!(w, "    let specializations = [(\"\", {intrinsic_name})];")?;
        }
        _ => {
            writeln!(w, "    let specializations = [")?;

            for specialization in specializations {
                let mut specialization: Vec<_> =
                    specialization.iter().map(|d| d.to_string()).collect();

                let const_args = specialization.join(",");

                // The identifier is reversed.
                specialization.reverse();
                let id = specialization.join("-");

                writeln!(
                    w,
                    "        (\"-{id}\", {intrinsic_name}::<{const_args}> as {coerce}),"
                )?;
            }

            writeln!(w, "    ];")?;
        }
    }

    let return_value = format_f16_return_value(intrinsic);
    let indentation2 = indentation.nested();
    let indentation3 = indentation2.nested();
    writeln!(
        w,
        "\
            for (id, f) in specializations {{\n\
                for i in 0..{passes} {{\n\
                    unsafe {{\n\
                        {loaded_args}\
                        let __return_value = f({args});\n\
                        println!(\"Result {{id}}-{{}}: {{:?}}\", i + 1, {return_value});\n\
                    }}\n\
                }}\n\
            }}",
        loaded_args = intrinsic.arguments.load_values_rust(indentation3),
        args = intrinsic.arguments.as_call_param_rust(),
    )
}

/// Generate the specializations (unique sequences of const-generic arguments) for this intrinsic.
fn generate_rust_specializations(
    constraints: &mut impl Iterator<Item = impl Iterator<Item = i64>>,
) -> Vec<Vec<u8>> {
    let mut specializations = vec![vec![]];

    for constraint in constraints {
        specializations = constraint
            .flat_map(|right| {
                specializations.iter().map(move |left| {
                    let mut left = left.clone();
                    left.push(u8::try_from(right).unwrap());
                    left
                })
            })
            .collect();
    }

    specializations
}

// Top-level function to create complete test program
pub fn create_rust_test_module<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    intrinsic: &Intrinsic<T>,
) -> std::io::Result<()> {
    trace!("generating `{}`", intrinsic.name);
    let indentation = Indentation::default();

    writeln!(w, "pub fn run_{}() {{", intrinsic.name)?;

    // Define the arrays of arguments.
    let arguments = &intrinsic.arguments;
    arguments.gen_arglists_rust(w, indentation.nested(), PASSES)?;

    // Define any const generics as `const` items, then generate the actual test loop.
    let specializations = generate_rust_specializations(
        &mut arguments
            .iter()
            .filter_map(|i| i.constraint.as_ref().map(|v| v.iter())),
    );

    generate_rust_test_loop(w, intrinsic, indentation, &specializations, PASSES)?;

    writeln!(w, "}}")?;

    Ok(())
}
