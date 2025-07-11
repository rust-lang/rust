use itertools::Itertools;
use std::process::Command;

use super::argument::Argument;
use super::indentation::Indentation;
use super::intrinsic::{IntrinsicDefinition, format_f16_return_value};
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
    architecture: &str,
    notice: &str,
    cfg: &str,
    definitions: &str,
    intrinsics: &[impl IntrinsicDefinition<T>],
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

    writeln!(w, "use core_arch::arch::{architecture}::*;")?;

    writeln!(w, "{definitions}")?;

    for intrinsic in intrinsics {
        crate::common::gen_rust::create_rust_test_module(w, intrinsic)?;
    }

    Ok(())
}

pub fn compile_rust_programs(toolchain: Option<&str>, target: &str, linker: Option<&str>) -> bool {
    /* If there has been a linker explicitly set from the command line then
     * we want to set it via setting it in the RUSTFLAGS*/

    trace!("Building cargo command");

    let mut cargo_command = Command::new("cargo");
    cargo_command.current_dir("rust_programs");

    // Do not use the target directory of the workspace please.
    cargo_command.env("CARGO_TARGET_DIR", "target");

    if let Some(toolchain) = toolchain
        && !toolchain.is_empty()
    {
        cargo_command.arg(toolchain);
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
    intrinsic: &dyn IntrinsicDefinition<T>,
    indentation: Indentation,
    additional: &str,
    passes: u32,
) -> std::io::Result<()> {
    let constraints = intrinsic.arguments().as_constraint_parameters_rust();
    let constraints = if !constraints.is_empty() {
        format!("::<{constraints}>")
    } else {
        constraints
    };

    let return_value = format_f16_return_value(intrinsic);
    let indentation2 = indentation.nested();
    let indentation3 = indentation2.nested();
    writeln!(
        w,
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

fn generate_rust_constraint_blocks<'a, T: IntrinsicTypeDefinition + 'a>(
    w: &mut impl std::io::Write,
    intrinsic: &dyn IntrinsicDefinition<T>,
    indentation: Indentation,
    constraints: &mut (impl Iterator<Item = &'a Argument<T>> + Clone),
    name: String,
) -> std::io::Result<()> {
    let Some(current) = constraints.next() else {
        return generate_rust_test_loop(w, intrinsic, indentation, &name, PASSES);
    };

    let body_indentation = indentation.nested();
    for i in current.constraint.iter().flat_map(|c| c.to_range()) {
        let ty = current.ty.rust_type();

        writeln!(w, "{indentation}{{")?;

        writeln!(w, "{body_indentation}const {}: {ty} = {i};", current.name)?;

        generate_rust_constraint_blocks(
            w,
            intrinsic,
            body_indentation,
            &mut constraints.clone(),
            format!("{name}-{i}"),
        )?;

        writeln!(w, "{indentation}}}")?;
    }

    Ok(())
}

// Top-level function to create complete test program
pub fn create_rust_test_module<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    intrinsic: &dyn IntrinsicDefinition<T>,
) -> std::io::Result<()> {
    trace!("generating `{}`", intrinsic.name());
    let indentation = Indentation::default();

    writeln!(w, "pub fn run_{}() {{", intrinsic.name())?;

    // Define the arrays of arguments.
    let arguments = intrinsic.arguments();
    arguments.gen_arglists_rust(w, indentation.nested(), PASSES)?;

    // Define any const generics as `const` items, then generate the actual test loop.
    generate_rust_constraint_blocks(
        w,
        intrinsic,
        indentation.nested(),
        &mut arguments.iter().rev().filter(|i| i.has_constraint()),
        Default::default(),
    )?;

    writeln!(w, "}}")?;

    Ok(())
}
