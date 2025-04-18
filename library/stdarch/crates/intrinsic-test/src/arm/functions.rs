use super::config::{AARCH_CONFIGURATIONS, POLY128_OSTREAM_DEF, build_notices};
use super::intrinsic::ArmIntrinsicType;
use crate::arm::constraint::Constraint;
use crate::common::argument::Argument;
use crate::common::compile_c::CompilationCommandBuilder;
use crate::common::format::Indentation;
use crate::common::gen_c::{compile_c, create_c_filenames, generate_c_program};
use crate::common::gen_rust::{compile_rust, create_rust_filenames, generate_rust_program};
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_types::IntrinsicTypeDefinition;
use crate::common::write_file;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::BTreeMap;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;

fn gen_code_c(
    indentation: Indentation,
    intrinsic: &Intrinsic<ArmIntrinsicType, Constraint>,
    constraints: &[&Argument<ArmIntrinsicType, Constraint>],
    name: String,
    target: &str,
) -> String {
    if let Some((current, constraints)) = constraints.split_last() {
        let range = current
            .metadata
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

fn generate_c_program_arm(
    header_files: &[&str],
    intrinsic: &Intrinsic<ArmIntrinsicType, Constraint>,
    target: &str,
) -> String {
    let constraints = intrinsic
        .arguments
        .iter()
        .filter(|&i| i.has_constraint())
        .collect_vec();

    let indentation = Indentation::default();
    generate_c_program(
        build_notices("// ").as_str(),
        header_files,
        "aarch64",
        &[POLY128_OSTREAM_DEF],
        intrinsic
            .arguments
            .gen_arglists_c(indentation, PASSES)
            .as_str(),
        gen_code_c(
            indentation.nested(),
            intrinsic,
            constraints.as_slice(),
            Default::default(),
            target,
        )
        .as_str(),
    )
}

fn gen_code_rust(
    indentation: Indentation,
    intrinsic: &Intrinsic<ArmIntrinsicType, Constraint>,
    constraints: &[&Argument<ArmIntrinsicType, Constraint>],
    name: String,
) -> String {
    if let Some((current, constraints)) = constraints.split_last() {
        let range = current
            .metadata
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

fn generate_rust_program_arm(
    intrinsic: &Intrinsic<ArmIntrinsicType, Constraint>,
    target: &str,
) -> String {
    let constraints = intrinsic
        .arguments
        .iter()
        .filter(|i| i.has_constraint())
        .collect_vec();

    let indentation = Indentation::default();
    let final_target = if target.contains("v7") {
        "arm"
    } else {
        "aarch64"
    };
    generate_rust_program(
        build_notices("// ").as_str(),
        AARCH_CONFIGURATIONS,
        final_target,
        intrinsic
            .arguments
            .gen_arglists_rust(indentation.nested(), PASSES)
            .as_str(),
        gen_code_rust(
            indentation.nested(),
            intrinsic,
            &constraints,
            Default::default(),
        )
        .as_str(),
    )
}

fn compile_c_arm(
    intrinsics_name_list: &Vec<String>,
    _filename_mapping: BTreeMap<&String, String>,
    compiler: &str,
    target: &str,
    cxx_toolchain_dir: Option<&str>,
) -> bool {
    let mut command = CompilationCommandBuilder::new()
        .add_arch_flags(vec!["armv8.6-a", "crypto", "crc", "dotprod", "fp16"])
        .set_compiler(compiler)
        .set_target(target)
        .set_opt_level("2")
        .set_cxx_toolchain_dir(cxx_toolchain_dir)
        .set_project_root("c_programs")
        .add_extra_flags(vec!["-ffp-contract=off", "-Wno-narrowing"]);

    if !target.contains("v7") {
        command = command.add_arch_flags(vec!["faminmax", "lut", "sha3"]);
    }

    command = if target == "aarch64_be-unknown-linux-gnu" {
        command
            .set_linker(
                cxx_toolchain_dir.unwrap_or("").to_string() + "/bin/aarch64_be-none-linux-gnu-g++",
            )
            .set_include_paths(vec![
                "/include",
                "/aarch64_be-none-linux-gnu/include",
                "/aarch64_be-none-linux-gnu/include/c++/14.2.1",
                "/aarch64_be-none-linux-gnu/include/c++/14.2.1/aarch64_be-none-linux-gnu",
                "/aarch64_be-none-linux-gnu/include/c++/14.2.1/backward",
                "/aarch64_be-none-linux-gnu/libc/usr/include",
            ])
    } else {
        if compiler.contains("clang") {
            command.add_extra_flag(format!("-target {target}").as_str())
        } else {
            command.add_extra_flag("-flax-vector-conversions")
        }
    };

    let compiler_commands = intrinsics_name_list
        .iter()
        .map(|intrinsic_name| {
            command
                .clone()
                .set_input_name(intrinsic_name)
                .set_output_name(intrinsic_name)
                .to_string()
        })
        .collect::<Vec<_>>();

    compile_c(&compiler_commands)
}

pub fn build_c(
    intrinsics: &Vec<Intrinsic<ArmIntrinsicType, Constraint>>,
    compiler: Option<&str>,
    target: &str,
    cxx_toolchain_dir: Option<&str>,
) -> bool {
    let intrinsics_name_list = intrinsics
        .par_iter()
        .map(|i| i.name.clone())
        .collect::<Vec<_>>();
    let filename_mapping = create_c_filenames(&intrinsics_name_list);

    intrinsics.par_iter().for_each(|i| {
        let c_code = generate_c_program_arm(&["arm_neon.h", "arm_acle.h", "arm_fp16.h"], i, target);
        match filename_mapping.get(&i.name) {
            Some(filename) => write_file(filename, c_code),
            None => {}
        };
    });

    match compiler {
        None => true,
        Some(compiler) => compile_c_arm(
            &intrinsics_name_list,
            filename_mapping,
            compiler,
            target,
            cxx_toolchain_dir,
        ),
    }
}

pub fn build_rust(
    intrinsics: &[Intrinsic<ArmIntrinsicType, Constraint>],
    toolchain: Option<&str>,
    target: &str,
    linker: Option<&str>,
) -> bool {
    let intrinsics_name_list = intrinsics
        .par_iter()
        .map(|i| i.name.clone())
        .collect::<Vec<_>>();
    let filename_mapping = create_rust_filenames(&intrinsics_name_list);

    intrinsics.par_iter().for_each(|i| {
        let rust_code = generate_rust_program_arm(i, target);
        match filename_mapping.get(&i.name) {
            Some(filename) => write_file(filename, rust_code),
            None => {}
        }
    });

    let intrinsics_name_list = intrinsics.iter().map(|i| i.name.as_str()).collect_vec();

    compile_rust(&intrinsics_name_list, toolchain, target, linker)
}
