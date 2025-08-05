use crate::common::cli::ProcessedCli;
use crate::common::compile_c::{CompilationCommandBuilder, CppCompilation};

pub fn build_cpp_compilation(config: &ProcessedCli) -> Option<CppCompilation> {
    let cpp_compiler = config.cpp_compiler.as_ref()?;

    // -ffp-contract=off emulates Rust's approach of not fusing separate mul-add operations
    let mut command = CompilationCommandBuilder::new()
        .add_arch_flags([
            "avx",
            "avx2",
            "avx512f",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512bw",
            "avx512bf16",
            "avx512bitalg",
            "lzcnt",
            "popcnt",
            "adx",
            "aes",
        ])
        .set_compiler(cpp_compiler)
        .set_target(&config.target)
        .set_opt_level("2")
        .set_cxx_toolchain_dir(config.cxx_toolchain_dir.as_deref())
        .set_project_root("c_programs")
        .add_extra_flags(vec!["-ffp-contract=off", "-Wno-narrowing"]);

    if !cpp_compiler.contains("clang") {
        command = command.add_extra_flag("-flax-vector-conversions");
    }

    let cpp_compiler = command.into_cpp_compilation();

    Some(cpp_compiler)
}
