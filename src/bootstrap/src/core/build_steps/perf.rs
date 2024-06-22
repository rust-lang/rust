use std::process::Command;

use crate::core::build_steps::compile::{Std, Sysroot};
use crate::core::build_steps::tool::RustcPerf;
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

    let results_dir = builder.build.tempdir().join("rustc-perf");

    let mut cmd = Command::new(collector);
    let cmd = cmd
        .arg("profile_local")
        .arg("eprintln")
        .arg("--out-dir")
        .arg(&results_dir)
        .arg("--include")
        .arg("helloworld")
        .arg(&rustc);

    builder.info(&format!("Running `rustc-perf` using `{}`", rustc.display()));

    // We need to set the working directory to `src/tools/perf`, so that it can find the directory
    // with compile-time benchmarks.
    let cmd = cmd.current_dir(builder.src.join("src/tools/rustc-perf"));
    builder.build.run(cmd);

    builder.info(&format!("You can find the results at `{}`", results_dir.display()));
}
