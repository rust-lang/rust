use std::env;
use std::fs;
use std::path::Path;

use super::path::{Dirs, RelPath};
use super::prepare::GitRepo;
use super::rustc_info::get_file_name;
use super::utils::{hyperfine_command, spawn_and_wait, CargoProject, Compiler};

static SIMPLE_RAYTRACER_REPO: GitRepo = GitRepo::github(
    "ebobby",
    "simple-raytracer",
    "804a7a21b9e673a482797aa289a18ed480e4d813",
    "<none>",
);

// Use a separate target dir for the initial LLVM build to reduce unnecessary recompiles
static SIMPLE_RAYTRACER_LLVM: CargoProject =
    CargoProject::new(&SIMPLE_RAYTRACER_REPO.source_dir(), "simple_raytracer_llvm");

static SIMPLE_RAYTRACER: CargoProject =
    CargoProject::new(&SIMPLE_RAYTRACER_REPO.source_dir(), "simple_raytracer");

pub(crate) fn benchmark(dirs: &Dirs, bootstrap_host_compiler: &Compiler) {
    benchmark_simple_raytracer(dirs, bootstrap_host_compiler);
}

fn benchmark_simple_raytracer(dirs: &Dirs, bootstrap_host_compiler: &Compiler) {
    if std::process::Command::new("hyperfine").output().is_err() {
        eprintln!("Hyperfine not installed");
        eprintln!("Hint: Try `cargo install hyperfine` to install hyperfine");
        std::process::exit(1);
    }

    if !SIMPLE_RAYTRACER_REPO.source_dir().to_path(dirs).exists() {
        SIMPLE_RAYTRACER_REPO.fetch(dirs);
        spawn_and_wait(SIMPLE_RAYTRACER.fetch(
            &bootstrap_host_compiler.cargo,
            &bootstrap_host_compiler.rustc,
            dirs,
        ));
    }

    eprintln!("[LLVM BUILD] simple-raytracer");
    let build_cmd = SIMPLE_RAYTRACER_LLVM.build(bootstrap_host_compiler, dirs);
    spawn_and_wait(build_cmd);
    fs::copy(
        SIMPLE_RAYTRACER_LLVM
            .target_dir(dirs)
            .join(&bootstrap_host_compiler.triple)
            .join("debug")
            .join(get_file_name("main", "bin")),
        RelPath::BUILD.to_path(dirs).join(get_file_name("raytracer_cg_llvm", "bin")),
    )
    .unwrap();

    let bench_runs = env::var("BENCH_RUNS").unwrap_or_else(|_| "10".to_string()).parse().unwrap();

    eprintln!("[BENCH COMPILE] ebobby/simple-raytracer");
    let cargo_clif =
        RelPath::DIST.to_path(dirs).join(get_file_name("cargo_clif", "bin").replace('_', "-"));
    let manifest_path = SIMPLE_RAYTRACER.manifest_path(dirs);
    let target_dir = SIMPLE_RAYTRACER.target_dir(dirs);

    let clean_cmd = format!(
        "RUSTC=rustc cargo clean --manifest-path {manifest_path} --target-dir {target_dir}",
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );
    let llvm_build_cmd = format!(
        "RUSTC=rustc cargo build --manifest-path {manifest_path} --target-dir {target_dir}",
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );
    let clif_build_cmd = format!(
        "RUSTC=rustc {cargo_clif} build --manifest-path {manifest_path} --target-dir {target_dir}",
        cargo_clif = cargo_clif.display(),
        manifest_path = manifest_path.display(),
        target_dir = target_dir.display(),
    );

    let bench_compile =
        hyperfine_command(1, bench_runs, Some(&clean_cmd), &llvm_build_cmd, &clif_build_cmd);

    spawn_and_wait(bench_compile);

    eprintln!("[BENCH RUN] ebobby/simple-raytracer");
    fs::copy(
        target_dir.join("debug").join(get_file_name("main", "bin")),
        RelPath::BUILD.to_path(dirs).join(get_file_name("raytracer_cg_clif", "bin")),
    )
    .unwrap();

    let mut bench_run = hyperfine_command(
        0,
        bench_runs,
        None,
        Path::new(".").join(get_file_name("raytracer_cg_llvm", "bin")).to_str().unwrap(),
        Path::new(".").join(get_file_name("raytracer_cg_clif", "bin")).to_str().unwrap(),
    );
    bench_run.current_dir(RelPath::BUILD.to_path(dirs));
    spawn_and_wait(bench_run);
}
