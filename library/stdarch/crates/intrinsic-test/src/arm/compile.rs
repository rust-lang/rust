use crate::common::cli::ProcessedCli;
use crate::common::compile_c::{CompilationCommandBuilder, CppCompilation};

pub fn build_cpp_compilation(config: &ProcessedCli) -> Option<CppCompilation> {
    let cpp_compiler = config.cpp_compiler.as_ref()?;

    // -ffp-contract=off emulates Rust's approach of not fusing separate mul-add operations
    let mut command = CompilationCommandBuilder::new()
        .add_arch_flags(["armv8.6-a", "crypto", "crc", "dotprod", "fp16"])
        .set_compiler(cpp_compiler)
        .set_target(&config.target)
        .set_opt_level("2")
        .set_cxx_toolchain_dir(config.cxx_toolchain_dir.as_deref())
        .set_project_root("c_programs")
        .add_extra_flags(["-ffp-contract=off", "-Wno-narrowing"]);

    if !config.target.contains("v7") {
        command = command.add_arch_flags(["faminmax", "lut", "sha3"]);
    }

    if !cpp_compiler.contains("clang") {
        command = command.add_extra_flag("-flax-vector-conversions");
    }

    let mut cpp_compiler = command.into_cpp_compilation();

    if config.target.contains("aarch64_be") {
        let Some(ref cxx_toolchain_dir) = config.cxx_toolchain_dir else {
            panic!(
                "target `{}` must specify `cxx_toolchain_dir`",
                config.target
            )
        };

        cpp_compiler.command_mut().args([
            &format!("--sysroot={cxx_toolchain_dir}/aarch64_be-none-linux-gnu/libc"),
            "--include-directory",
            &format!("{cxx_toolchain_dir}/aarch64_be-none-linux-gnu/include/c++/14.3.1"),
            "--include-directory",
            &format!("{cxx_toolchain_dir}/aarch64_be-none-linux-gnu/include/c++/14.3.1/aarch64_be-none-linux-gnu"),
            "-L",
            &format!("{cxx_toolchain_dir}/lib/gcc/aarch64_be-none-linux-gnu/14.3.1"),
            "-L",
            &format!("{cxx_toolchain_dir}/aarch64_be-none-linux-gnu/libc/usr/lib"),
            "-B",
            &format!("{cxx_toolchain_dir}/lib/gcc/aarch64_be-none-linux-gnu/14.3.1"),
        ]);
    }

    Some(cpp_compiler)
}
