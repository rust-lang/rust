use super::argument::Argument;
use super::config::{AARCH_CONFIGURATIONS, POLY128_OSTREAM_DEF, build_notices};
use super::format::Indentation;
use super::intrinsic::Intrinsic;
use crate::common::gen_c::{compile_c, create_c_files, generate_c_program};
use crate::common::gen_rust::{compile_rust, create_rust_files, generate_rust_program};
use itertools::Itertools;
use rayon::prelude::*;
use std::io::Write;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;

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

fn generate_c_program_arm(header_files: &[&str], intrinsic: &Intrinsic, target: &str) -> String {
    let constraints = intrinsic
        .arguments
        .iter()
        .filter(|i| i.has_constraint())
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

fn generate_rust_program_arm(intrinsic: &Intrinsic, target: &str) -> String {
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
    intrinsics_name_list: Vec<String>,
    compiler: &str,
    target: &str,
    cxx_toolchain_dir: Option<&str>,
) -> bool {
    let compiler_commands = intrinsics_name_list.iter().map(|intrinsic_name|{
        let c_filename = format!(r#"c_programs/{intrinsic_name}.cpp"#);
        let flags = std::env::var("CPPFLAGS").unwrap_or("".into());
        let arch_flags = if target.contains("v7") {
            "-march=armv8.6-a+crypto+crc+dotprod+fp16"
        } else {
            "-march=armv8.6-a+crypto+sha3+crc+dotprod+fp16+faminmax+lut"
        };

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

        compiler_command
    })
    .collect::<Vec<_>>();

    compile_c(&compiler_commands)
}

pub fn build_c(
    intrinsics: &Vec<Intrinsic>,
    compiler: Option<&str>,
    target: &str,
    cxx_toolchain_dir: Option<&str>,
) -> bool {
    let _ = std::fs::create_dir("c_programs");
    let intrinsics_name_list = intrinsics
        .par_iter()
        .map(|i| i.name.clone())
        .collect::<Vec<_>>();
    let file_mapping = create_c_files(&intrinsics_name_list);

    intrinsics.par_iter().for_each(|i| {
        let c_code = generate_c_program_arm(&["arm_neon.h", "arm_acle.h", "arm_fp16.h"], i, target);
        match file_mapping.get(&i.name) {
            Some(mut file) => file.write_all(c_code.into_bytes().as_slice()).unwrap(),
            None => {}
        };
    });

    match compiler {
        None => true,
        Some(compiler) => compile_c_arm(intrinsics_name_list, compiler, target, cxx_toolchain_dir),
    }
}

pub fn build_rust(
    intrinsics: &[Intrinsic],
    toolchain: Option<&str>,
    target: &str,
    linker: Option<&str>,
) -> bool {
    let intrinsics_name_list = intrinsics
        .par_iter()
        .map(|i| i.name.clone())
        .collect::<Vec<_>>();
    let file_mapping = create_rust_files(&intrinsics_name_list);

    intrinsics.par_iter().for_each(|i| {
        let c_code = generate_rust_program_arm(i, target);
        match file_mapping.get(&i.name) {
            Some(mut file) => file.write_all(c_code.into_bytes().as_slice()).unwrap(),
            None => {}
        }
    });

    let intrinsics_name_list = intrinsics.iter().map(|i| i.name.as_str()).collect_vec();

    compile_rust(&intrinsics_name_list, toolchain, target, linker)
}
