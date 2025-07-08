use itertools::Itertools;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::process::Command;

use super::argument::Argument;
use super::indentation::Indentation;
use super::intrinsic::{IntrinsicDefinition, format_f16_return_value};
use super::intrinsic_helpers::IntrinsicTypeDefinition;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;

pub fn format_rust_main_template(
    notices: &str,
    definitions: &str,
    configurations: &str,
    arch_definition: &str,
    arglists: &str,
    passes: &str,
) -> String {
    format!(
        r#"{notices}#![feature(simd_ffi)]
#![feature(f16)]
#![allow(unused)]
{configurations}
{definitions}

use core_arch::arch::{arch_definition}::*;

fn main() {{
{arglists}
{passes}
}}
"#,
    )
}

fn write_cargo_toml(w: &mut impl std::io::Write, binaries: &[String]) -> std::io::Result<()> {
    writeln!(
        w,
        concat!(
            "[package]\n",
            "name = \"intrinsic-test-programs\"\n",
            "version = \"{version}\"\n",
            "authors = [{authors}]\n",
            "license = \"{license}\"\n",
            "edition = \"2018\"\n",
            "[workspace]\n",
            "[dependencies]\n",
            "core_arch = {{ path = \"../crates/core_arch\" }}",
        ),
        version = env!("CARGO_PKG_VERSION"),
        authors = env!("CARGO_PKG_AUTHORS")
            .split(":")
            .format_with(", ", |author, fmt| fmt(&format_args!("\"{author}\""))),
        license = env!("CARGO_PKG_LICENSE"),
    )?;

    for binary in binaries {
        writeln!(
            w,
            concat!(
                "[[bin]]\n",
                "name = \"{binary}\"\n",
                "path = \"{binary}/main.rs\"\n",
            ),
            binary = binary,
        )?;
    }

    Ok(())
}

pub fn compile_rust_programs(
    binaries: Vec<String>,
    toolchain: Option<&str>,
    target: &str,
    linker: Option<&str>,
) -> bool {
    let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
    write_cargo_toml(&mut cargo, &binaries).unwrap();

    /* If there has been a linker explicitly set from the command line then
     * we want to set it via setting it in the RUSTFLAGS*/

    let mut cargo_command = Command::new("cargo");
    cargo_command.current_dir("rust_programs");

    if let Some(toolchain) = toolchain {
        if !toolchain.is_empty() {
            cargo_command.arg(toolchain);
        }
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
    let output = cargo_command.output();

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

// Creates directory structure and file path mappings
pub fn setup_rust_file_paths(identifiers: &Vec<String>) -> BTreeMap<&String, String> {
    identifiers
        .par_iter()
        .map(|identifier| {
            let rust_dir = format!("rust_programs/{identifier}");
            let _ = std::fs::create_dir_all(&rust_dir);
            let rust_filename = format!("{rust_dir}/main.rs");

            (identifier, rust_filename)
        })
        .collect::<BTreeMap<&String, String>>()
}

pub fn generate_rust_test_loop<T: IntrinsicTypeDefinition>(
    intrinsic: &dyn IntrinsicDefinition<T>,
    indentation: Indentation,
    additional: &str,
    passes: u32,
) -> String {
    let constraints = intrinsic.arguments().as_constraint_parameters_rust();
    let constraints = if !constraints.is_empty() {
        format!("::<{constraints}>")
    } else {
        constraints
    };

    let return_value = format_f16_return_value(intrinsic);
    let indentation2 = indentation.nested();
    let indentation3 = indentation2.nested();
    format!(
        "{indentation}for i in 0..{passes} {{\n\
            {indentation2}unsafe {{\n\
                {loaded_args}\
                {indentation3}let __return_value = {intrinsic_call}{const}({args});\n\
                {indentation3}println!(\"Result {additional}-{{}}: {{:?}}\", i + 1, {return_value});\n\
            {indentation2}}}\n\
        {indentation}}}",
        loaded_args = intrinsic.arguments().load_values_rust(indentation3),
        intrinsic_call = intrinsic.name(),
        const = constraints,
        args = intrinsic.arguments().as_call_param_rust(),
    )
}

pub fn generate_rust_constraint_blocks<T: IntrinsicTypeDefinition>(
    intrinsic: &dyn IntrinsicDefinition<T>,
    indentation: Indentation,
    constraints: &[&Argument<T>],
    name: String,
) -> String {
    if let Some((current, constraints)) = constraints.split_last() {
        let range = current
            .constraint
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
                    pass = generate_rust_constraint_blocks(
                        intrinsic,
                        body_indentation,
                        constraints,
                        format!("{name}-{i}")
                    )
                )
            })
            .join("\n")
    } else {
        generate_rust_test_loop(intrinsic, indentation, &name, PASSES)
    }
}

// Top-level function to create complete test program
pub fn create_rust_test_program<T: IntrinsicTypeDefinition>(
    intrinsic: &dyn IntrinsicDefinition<T>,
    target: &str,
    notice: &str,
    definitions: &str,
    cfg: &str,
) -> String {
    let arguments = intrinsic.arguments();
    let constraints = arguments
        .iter()
        .filter(|i| i.has_constraint())
        .collect_vec();

    let indentation = Indentation::default();
    format_rust_main_template(
        notice,
        definitions,
        cfg,
        target,
        intrinsic
            .arguments()
            .gen_arglists_rust(indentation.nested(), PASSES)
            .as_str(),
        generate_rust_constraint_blocks(
            intrinsic,
            indentation.nested(),
            &constraints,
            Default::default(),
        )
        .as_str(),
    )
}
