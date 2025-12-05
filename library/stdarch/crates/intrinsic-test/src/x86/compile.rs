use crate::common::cli::ProcessedCli;
use crate::common::compile_c::{CompilationCommandBuilder, CppCompilation};

pub fn build_cpp_compilation(config: &ProcessedCli) -> Option<CppCompilation> {
    let cpp_compiler = config.cpp_compiler.as_ref()?;

    // -ffp-contract=off emulates Rust's approach of not fusing separate mul-add operations
    let mut command = CompilationCommandBuilder::new()
        .add_arch_flags(["icelake-client"])
        .set_compiler(cpp_compiler)
        .set_target(&config.target)
        .set_opt_level("2")
        .set_cxx_toolchain_dir(config.cxx_toolchain_dir.as_deref())
        .set_project_root("c_programs")
        .add_extra_flags(vec![
            "-ffp-contract=off",
            "-Wno-narrowing",
            "-mavx",
            "-mavx2",
            "-mavx512f",
            "-msse2",
            "-mavx512vl",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512cd",
            "-mavx512fp16",
            "-msha512",
            "-msm4",
            "-mavxvnni",
            "-mavx512bitalg",
            "-mavx512ifma",
            "-mavx512vbmi",
            "-mavx512vbmi2",
            "-mavx512vnni",
            "-mavx512vpopcntdq",
            "-ferror-limit=1000",
            "-std=c++23",
        ]);

    if !cpp_compiler.contains("clang") {
        command = command.add_extra_flag("-flax-vector-conversions");
    }

    let cpp_compiler = command.into_cpp_compilation();

    Some(cpp_compiler)
}
