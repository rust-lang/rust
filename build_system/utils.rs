use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};

use super::path::{Dirs, RelPath};

#[derive(Clone, Debug)]
pub(crate) struct Compiler {
    pub(crate) cargo: PathBuf,
    pub(crate) rustc: PathBuf,
    pub(crate) rustdoc: PathBuf,
    pub(crate) rustflags: String,
    pub(crate) rustdocflags: String,
    pub(crate) triple: String,
    pub(crate) runner: Vec<String>,
}

impl Compiler {
    pub(crate) fn set_cross_linker_and_runner(&mut self) {
        match self.triple.as_str() {
            "aarch64-unknown-linux-gnu" => {
                // We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
                self.rustflags += " -Clinker=aarch64-linux-gnu-gcc";
                self.rustdocflags += " -Clinker=aarch64-linux-gnu-gcc";
                self.runner = vec![
                    "qemu-aarch64".to_owned(),
                    "-L".to_owned(),
                    "/usr/aarch64-linux-gnu".to_owned(),
                ];
            }
            "s390x-unknown-linux-gnu" => {
                // We are cross-compiling for s390x. Use the correct linker and run tests in qemu.
                self.rustflags += " -Clinker=s390x-linux-gnu-gcc";
                self.rustdocflags += " -Clinker=s390x-linux-gnu-gcc";
                self.runner = vec![
                    "qemu-s390x".to_owned(),
                    "-L".to_owned(),
                    "/usr/s390x-linux-gnu".to_owned(),
                ];
            }
            "x86_64-pc-windows-gnu" => {
                // We are cross-compiling for Windows. Run tests in wine.
                self.runner = vec!["wine".to_owned()];
            }
            _ => {
                println!("Unknown non-native platform");
            }
        }
    }
}

pub(crate) struct CargoProject {
    source: &'static RelPath,
    target: &'static str,
}

impl CargoProject {
    pub(crate) const fn new(path: &'static RelPath, target: &'static str) -> CargoProject {
        CargoProject { source: path, target }
    }

    pub(crate) fn source_dir(&self, dirs: &Dirs) -> PathBuf {
        self.source.to_path(dirs)
    }

    pub(crate) fn manifest_path(&self, dirs: &Dirs) -> PathBuf {
        self.source_dir(dirs).join("Cargo.toml")
    }

    pub(crate) fn target_dir(&self, dirs: &Dirs) -> PathBuf {
        RelPath::BUILD.join(self.target).to_path(dirs)
    }

    #[must_use]
    fn base_cmd(&self, command: &str, cargo: &Path, dirs: &Dirs) -> Command {
        let mut cmd = Command::new(cargo);

        cmd.arg(command)
            .arg("--manifest-path")
            .arg(self.manifest_path(dirs))
            .arg("--target-dir")
            .arg(self.target_dir(dirs))
            .arg("--locked");

        if dirs.frozen {
            cmd.arg("--frozen");
        }

        cmd
    }

    #[must_use]
    fn build_cmd(&self, command: &str, compiler: &Compiler, dirs: &Dirs) -> Command {
        let mut cmd = self.base_cmd(command, &compiler.cargo, dirs);

        cmd.arg("--target").arg(&compiler.triple);

        cmd.env("RUSTC", &compiler.rustc);
        cmd.env("RUSTDOC", &compiler.rustdoc);
        cmd.env("RUSTFLAGS", &compiler.rustflags);
        cmd.env("RUSTDOCFLAGS", &compiler.rustdocflags);
        if !compiler.runner.is_empty() {
            cmd.env(
                format!("CARGO_TARGET_{}_RUNNER", compiler.triple.to_uppercase().replace('-', "_")),
                compiler.runner.join(" "),
            );
        }

        cmd
    }

    pub(crate) fn clean(&self, dirs: &Dirs) {
        let _ = fs::remove_dir_all(self.target_dir(dirs));
    }

    #[must_use]
    pub(crate) fn build(&self, compiler: &Compiler, dirs: &Dirs) -> Command {
        self.build_cmd("build", compiler, dirs)
    }

    #[must_use]
    pub(crate) fn test(&self, compiler: &Compiler, dirs: &Dirs) -> Command {
        self.build_cmd("test", compiler, dirs)
    }

    #[must_use]
    pub(crate) fn run(&self, compiler: &Compiler, dirs: &Dirs) -> Command {
        self.build_cmd("run", compiler, dirs)
    }
}

#[must_use]
pub(crate) fn hyperfine_command(
    warmup: u64,
    runs: u64,
    prepare: Option<&str>,
    cmds: &[&str],
) -> Command {
    let mut bench = Command::new("hyperfine");

    if warmup != 0 {
        bench.arg("--warmup").arg(warmup.to_string());
    }

    if runs != 0 {
        bench.arg("--runs").arg(runs.to_string());
    }

    if let Some(prepare) = prepare {
        bench.arg("--prepare").arg(prepare);
    }

    bench.args(cmds);

    bench
}

#[must_use]
pub(crate) fn git_command<'a>(repo_dir: impl Into<Option<&'a Path>>, cmd: &str) -> Command {
    let mut git_cmd = Command::new("git");
    git_cmd
        .arg("-c")
        .arg("user.name=Dummy")
        .arg("-c")
        .arg("user.email=dummy@example.com")
        .arg("-c")
        .arg("core.autocrlf=false")
        .arg(cmd);
    if let Some(repo_dir) = repo_dir.into() {
        git_cmd.current_dir(repo_dir);
    }
    git_cmd
}

#[track_caller]
pub(crate) fn try_hard_link(src: impl AsRef<Path>, dst: impl AsRef<Path>) {
    let src = src.as_ref();
    let dst = dst.as_ref();
    if let Err(_) = fs::hard_link(src, dst) {
        fs::copy(src, dst).unwrap(); // Fallback to copying if hardlinking failed
    }
}

#[track_caller]
pub(crate) fn spawn_and_wait(mut cmd: Command) {
    if !cmd.spawn().unwrap().wait().unwrap().success() {
        process::exit(1);
    }
}

// Based on the retry function in rust's src/ci/shared.sh
#[track_caller]
pub(crate) fn retry_spawn_and_wait(tries: u64, mut cmd: Command) {
    for i in 1..tries + 1 {
        if i != 1 {
            println!("Command failed. Attempt {i}/{tries}:");
        }
        if cmd.spawn().unwrap().wait().unwrap().success() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(i * 5));
    }
    println!("The command has failed after {tries} attempts.");
    process::exit(1);
}

#[track_caller]
pub(crate) fn spawn_and_wait_with_input(mut cmd: Command, input: String) -> String {
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn child process");

    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    std::thread::spawn(move || {
        stdin.write_all(input.as_bytes()).expect("Failed to write to stdin");
    });

    let output = child.wait_with_output().expect("Failed to read stdout");
    if !output.status.success() {
        process::exit(1);
    }

    String::from_utf8(output.stdout).unwrap()
}

pub(crate) fn remove_dir_if_exists(path: &Path) {
    match fs::remove_dir_all(&path) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => panic!("Failed to remove {path}: {err}", path = path.display()),
    }
}

pub(crate) fn copy_dir_recursively(from: &Path, to: &Path) {
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let filename = entry.file_name();
        if filename == "." || filename == ".." {
            continue;
        }
        if entry.metadata().unwrap().is_dir() {
            fs::create_dir(to.join(&filename)).unwrap();
            copy_dir_recursively(&from.join(&filename), &to.join(&filename));
        } else {
            fs::copy(from.join(&filename), to.join(&filename)).unwrap();
        }
    }
}

pub(crate) fn is_ci() -> bool {
    env::var("CI").is_ok()
}

pub(crate) fn is_ci_opt() -> bool {
    env::var("CI_OPT").is_ok()
}

pub(crate) fn maybe_incremental(cmd: &mut Command) {
    if is_ci() || std::env::var("CARGO_BUILD_INCREMENTAL").map_or(false, |val| val == "false") {
        // Disabling incr comp reduces cache size and incr comp doesn't save as much on CI anyway
        cmd.env("CARGO_BUILD_INCREMENTAL", "false");
    } else {
        // Force incr comp even in release mode unless in CI or incremental builds are explicitly disabled
        cmd.env("CARGO_BUILD_INCREMENTAL", "true");
    }
}
