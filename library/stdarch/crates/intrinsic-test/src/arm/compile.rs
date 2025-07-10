use crate::common::cli::ProcessedCli;
use crate::common::compile_c::CompilationCommandBuilder;
use crate::common::gen_c::compile_c_programs;

pub fn compile_c_arm(config: &ProcessedCli, intrinsics_name_list: &[String]) -> bool {
    let Some(ref cpp_compiler) = config.cpp_compiler else {
        return true;
    };

    // -ffp-contract=off emulates Rust's approach of not fusing separate mul-add operations
    let mut command = CompilationCommandBuilder::new()
        .add_arch_flags(vec!["armv8.6-a", "crypto", "crc", "dotprod", "fp16"])
        .set_compiler(cpp_compiler)
        .set_target(&config.target)
        .set_opt_level("2")
        .set_cxx_toolchain_dir(config.cxx_toolchain_dir.as_deref())
        .set_project_root("c_programs")
        .add_extra_flags(vec!["-ffp-contract=off", "-Wno-narrowing"]);

    if !config.target.contains("v7") {
        command = command.add_arch_flags(vec!["faminmax", "lut", "sha3"]);
    }

    /*
     * clang++ cannot link an aarch64_be object file, so we invoke
     * aarch64_be-unknown-linux-gnu's C++ linker. This ensures that we
     * are testing the intrinsics against LLVM.
     *
     * Note: setting `--sysroot=<...>` which is the obvious thing to do
     * does not work as it gets caught up with `#include_next <stdlib.h>`
     * not existing...
     */
    if config.target.contains("aarch64_be") {
        let Some(ref cxx_toolchain_dir) = config.cxx_toolchain_dir else {
            panic!(
                "target `{}` must specify `cxx_toolchain_dir`",
                config.target
            )
        };

        let linker = if let Some(ref linker) = config.linker {
            linker.to_owned()
        } else {
            format!("{cxx_toolchain_dir}/bin/aarch64_be-none-linux-gnu-g++")
        };

        trace!("using linker: {linker}");

        command = command.set_linker(linker).set_include_paths(vec![
                "/include",
                "/aarch64_be-none-linux-gnu/include",
                "/aarch64_be-none-linux-gnu/include/c++/14.3.1",
                "/aarch64_be-none-linux-gnu/include/c++/14.3.1/aarch64_be-none-linux-gnu",
                "/aarch64_be-none-linux-gnu/include/c++/14.3.1/backward",
                "/aarch64_be-none-linux-gnu/libc/usr/include",
        ]);
    }

    if !cpp_compiler.contains("clang") {
        command = command.add_extra_flag("-flax-vector-conversions");
    }

    let compiler_commands = intrinsics_name_list
        .iter()
        .map(|intrinsic_name| {
            command
                .clone()
                .set_input_name(intrinsic_name)
                .set_output_name(intrinsic_name)
                .make_string()
        })
        .collect::<Vec<_>>();

    compile_c_programs(&compiler_commands)
}
