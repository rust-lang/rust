use std::env;
use std::io::Write;
use std::path::Path;

use super::path::{Dirs, RelPath};
use super::prepare::GitRepo;
use super::rustc_info::get_file_name;
use super::utils::{hyperfine_command, spawn_and_wait, Compiler};

static SIMPLE_RAYTRACER_REPO: GitRepo = GitRepo::github(
    "ebobby",
    "simple-raytracer",
    "804a7a21b9e673a482797aa289a18ed480e4d813",
    "ad6f59a2331a3f56",
    "<none>",
);

pub(crate) fn benchmark(dirs: &Dirs, bootstrap_host_compiler: &Compiler) {
    benchmark_simple_raytracer(dirs, bootstrap_host_compiler);
}

fn benchmark_simple_raytracer(dirs: &Dirs, bootstrap_host_compiler: &Compiler) {
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
    let cargo_clif = RelPath::DIST
        .to_path(dirs)
        .join(get_file_name(&bootstrap_host_compiler.rustc, "cargo_clif", "bin").replace('_', "-"));
    let manifest_path = SIMPLE_RAYTRACER_REPO.source_dir().to_path(dirs).join("Cargo.toml");
    let target_dir = RelPath::BUILD.join("simple_raytracer").to_path(dirs);

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
        "RUSTC=rustc {cargo_clif} build --manifest-path {manifest_path} --target-dir {target_dir} && (rm build/raytracer_cg_clif || true) && ln build/simple_raytracer/debug/main build/raytracer_cg_clif",
        cargo_clif = cargo_clif.display(),
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );
    let clif_build_opt_cmd = format!(
        "RUSTC=rustc {cargo_clif} build --manifest-path {manifest_path} --target-dir {target_dir} --release && (rm build/raytracer_cg_clif_opt || true) && ln build/simple_raytracer/release/main build/raytracer_cg_clif_opt",
        cargo_clif = cargo_clif.display(),
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );

    let bench_compile_markdown = RelPath::DIST.to_path(dirs).join("bench_compile.md");

    let bench_compile = hyperfine_command(
        1,
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

    let bench_run_markdown = RelPath::DIST.to_path(dirs).join("bench_run.md");

    let raytracer_cg_llvm = Path::new(".").join(get_file_name(
        &bootstrap_host_compiler.rustc,
        "raytracer_cg_llvm",
        "bin",
    ));
    let raytracer_cg_clif = Path::new(".").join(get_file_name(
        &bootstrap_host_compiler.rustc,
        "raytracer_cg_clif",
        "bin",
    ));
    let raytracer_cg_clif_opt = Path::new(".").join(get_file_name(
        &bootstrap_host_compiler.rustc,
        "raytracer_cg_clif_opt",
        "bin",
    ));
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
    bench_run.current_dir(RelPath::BUILD.to_path(dirs));
    spawn_and_wait(bench_run);

    if let Some(gha_step_summary) = gha_step_summary.as_mut() {
        gha_step_summary.write_all(b"## Run ebobby/simple-raytracer\n\n").unwrap();
        gha_step_summary.write_all(&std::fs::read(bench_run_markdown).unwrap()).unwrap();
        gha_step_summary.write_all(b"\n").unwrap();
    }
}
