use std::env;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::path::Dirs;
use crate::prepare::GitRepo;
use crate::rustc_info::get_file_name;
use crate::utils::{Compiler, spawn_and_wait};

static SIMPLE_RAYTRACER_REPO: GitRepo = GitRepo::github(
    "ebobby",
    "simple-raytracer",
    "804a7a21b9e673a482797aa289a18ed480e4d813",
    "ad6f59a2331a3f56",
    "<none>",
);

pub(crate) fn benchmark(dirs: &Dirs, compiler: &Compiler) {
    if std::process::Command::new("hyperfine").output().is_err() {
        eprintln!("Hyperfine not installed");
        eprintln!("Hint: Try `cargo install hyperfine` to install hyperfine");
        std::process::exit(1);
    }

    SIMPLE_RAYTRACER_REPO.fetch(dirs);
    SIMPLE_RAYTRACER_REPO.patch(dirs);

    let bench_runs = env::var("BENCH_RUNS").unwrap_or_else(|_| "10".to_string()).parse().unwrap();

    let mut gha_step_summary = if let Ok(file) = std::env::var("GITHUB_STEP_SUMMARY") {
        Some(std::fs::OpenOptions::new().append(true).open(file).unwrap())
    } else {
        None
    };

    eprintln!("[BENCH COMPILE] ebobby/simple-raytracer");
    let cargo_clif = &compiler.cargo;
    let rustc_clif = &compiler.rustc;
    let rustflags = &compiler.rustflags.join("\x1f");
    let manifest_path = SIMPLE_RAYTRACER_REPO.source_dir().to_path(dirs).join("Cargo.toml");
    let target_dir = dirs.build_dir.join("simple_raytracer");

    let clean_cmd = format!(
        "RUSTC=rustc cargo clean --manifest-path {manifest_path} --target-dir {target_dir}",
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );
    let llvm_build_cmd = format!(
        "RUSTC=rustc cargo build --manifest-path {manifest_path} --target-dir {target_dir} && (rm build/raytracer_cg_llvm || true) && ln build/simple_raytracer/debug/main build/raytracer_cg_llvm",
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );
    let clif_build_cmd = format!(
        "RUSTC={rustc_clif} CARGO_ENCODED_RUSTFLAGS=\"{rustflags}\" {cargo_clif} build --manifest-path {manifest_path} --target-dir {target_dir} && (rm build/raytracer_cg_clif || true) && ln build/simple_raytracer/debug/main build/raytracer_cg_clif",
        cargo_clif = cargo_clif.display(),
        rustc_clif = rustc_clif.display(),
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );
    let clif_build_opt_cmd = format!(
        "RUSTC={rustc_clif} CARGO_ENCODED_RUSTFLAGS=\"{rustflags}\" {cargo_clif} build --manifest-path {manifest_path} --target-dir {target_dir} --release && (rm build/raytracer_cg_clif_opt || true) && ln build/simple_raytracer/release/main build/raytracer_cg_clif_opt",
        cargo_clif = cargo_clif.display(),
        rustc_clif = rustc_clif.display(),
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );

    let bench_compile_markdown = dirs.build_dir.join("bench_compile.md");

    let bench_compile = hyperfine_command(
        0,
        bench_runs,
        Some(&clean_cmd),
        &[
            ("cargo build", &llvm_build_cmd),
            ("cargo-clif build", &clif_build_cmd),
            ("cargo-clif build --release", &clif_build_opt_cmd),
        ],
        &bench_compile_markdown,
    );

    spawn_and_wait(bench_compile);

    if let Some(gha_step_summary) = gha_step_summary.as_mut() {
        gha_step_summary.write_all(b"## Compile ebobby/simple-raytracer\n\n").unwrap();
        gha_step_summary.write_all(&std::fs::read(bench_compile_markdown).unwrap()).unwrap();
        gha_step_summary.write_all(b"\n").unwrap();
    }

    eprintln!("[BENCH RUN] ebobby/simple-raytracer");

    let bench_run_markdown = dirs.build_dir.join("bench_run.md");

    let raytracer_cg_llvm =
        Path::new(".").join(get_file_name(&compiler.rustc, "raytracer_cg_llvm", "bin"));
    let raytracer_cg_clif =
        Path::new(".").join(get_file_name(&compiler.rustc, "raytracer_cg_clif", "bin"));
    let raytracer_cg_clif_opt =
        Path::new(".").join(get_file_name(&compiler.rustc, "raytracer_cg_clif_opt", "bin"));
    let mut bench_run = hyperfine_command(
        0,
        bench_runs,
        None,
        &[
            ("", raytracer_cg_llvm.to_str().unwrap()),
            ("", raytracer_cg_clif.to_str().unwrap()),
            ("", raytracer_cg_clif_opt.to_str().unwrap()),
        ],
        &bench_run_markdown,
    );
    bench_run.current_dir(&dirs.build_dir);
    spawn_and_wait(bench_run);

    if let Some(gha_step_summary) = gha_step_summary.as_mut() {
        gha_step_summary.write_all(b"## Run ebobby/simple-raytracer\n\n").unwrap();
        gha_step_summary.write_all(&std::fs::read(bench_run_markdown).unwrap()).unwrap();
        gha_step_summary.write_all(b"\n").unwrap();
    }
}

#[must_use]
fn hyperfine_command(
    warmup: u64,
    runs: u64,
    prepare: Option<&str>,
    cmds: &[(&str, &str)],
    markdown_export: &Path,
) -> Command {
    let mut bench = Command::new("hyperfine");

    bench.arg("--export-markdown").arg(markdown_export);

    if warmup != 0 {
        bench.arg("--warmup").arg(warmup.to_string());
    }

    if runs != 0 {
        bench.arg("--runs").arg(runs.to_string());
    }

    if let Some(prepare) = prepare {
        bench.arg("--prepare").arg(prepare);
    }

    for &(name, cmd) in cmds {
        if name != "" {
            bench.arg("-n").arg(name);
        }
        bench.arg(cmd);
    }

    bench
}
