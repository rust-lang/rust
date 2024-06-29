use crate::core::build_steps::compile::{Std, Sysroot};
use crate::core::build_steps::tool::{RustcPerf, Tool};
use crate::core::builder::Builder;
use crate::core::config::DebuginfoLevel;

/// Performs profiling using `rustc-perf` on a built version of the compiler.
pub fn perf(builder: &Builder<'_>) {
    let collector = builder.ensure(RustcPerf {
        compiler: builder.compiler(0, builder.config.build),
        target: builder.config.build,
    });

    if builder.build.config.rust_debuginfo_level_rustc == DebuginfoLevel::None {
        builder.info(r#"WARNING: You are compiling rustc without debuginfo, this will make profiling less useful.
Consider setting `rust.debuginfo-level = 1` in `config.toml`."#);
    }

    let compiler = builder.compiler(builder.top_stage, builder.config.build);
    builder.ensure(Std::new(compiler, builder.config.build));
    let sysroot = builder.ensure(Sysroot::new(compiler));
    let rustc = sysroot.join("bin/rustc");

    let rustc_perf_dir = builder.build.tempdir().join("rustc-perf");
    let profile_results_dir = rustc_perf_dir.join("results");

    // We need to take args passed after `--` and pass them to `rustc-perf-wrapper`
    let args = std::env::args().skip_while(|a| a != "--").skip(1);

    let mut cmd = builder.tool_cmd(Tool::RustcPerfWrapper);
    cmd.env("RUSTC_REAL", rustc)
        .env("PERF_COLLECTOR", collector)
        .env("PERF_RESULT_DIR", profile_results_dir)
        .args(args);
    builder.run(&mut cmd);
}
