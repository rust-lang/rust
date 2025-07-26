use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::{env, fs, io};

use crate::path::{Dirs, RelPath};
use crate::shared_utils::rustflags_to_cmd_env;

#[derive(Clone, Debug)]
pub(crate) struct Compiler {
    pub(crate) cargo: PathBuf,
    pub(crate) rustc: PathBuf,
    pub(crate) rustdoc: PathBuf,
    pub(crate) rustflags: Vec<String>,
    pub(crate) rustdocflags: Vec<String>,
    pub(crate) triple: String,
    pub(crate) runner: Vec<String>,
}

impl Compiler {
    pub(crate) fn set_cross_linker_and_runner(&mut self) {
        match self.triple.as_str() {
            "aarch64-unknown-linux-gnu" => {
                // We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
                self.rustflags.push("-Clinker=aarch64-linux-gnu-gcc".to_owned());
                self.rustdocflags.push("-Clinker=aarch64-linux-gnu-gcc".to_owned());
                self.runner = vec![
                    "qemu-aarch64".to_owned(),
                    "-L".to_owned(),
                    "/usr/aarch64-linux-gnu".to_owned(),
                ];
            }
            "s390x-unknown-linux-gnu" => {
                // We are cross-compiling for s390x. Use the correct linker and run tests in qemu.
                self.rustflags.push("-Clinker=s390x-linux-gnu-gcc".to_owned());
                self.rustdocflags.push("-Clinker=s390x-linux-gnu-gcc".to_owned());
                self.runner = vec![
                    "qemu-s390x".to_owned(),
                    "-L".to_owned(),
                    "/usr/s390x-linux-gnu".to_owned(),
                ];
            }
            "riscv64gc-unknown-linux-gnu" => {
                // We are cross-compiling for riscv64. Use the correct linker and run tests in qemu.
                self.rustflags.push("-Clinker=riscv64-linux-gnu-gcc".to_owned());
                self.rustdocflags.push("-Clinker=riscv64-linux-gnu-gcc".to_owned());
                self.runner = vec![
                    "qemu-riscv64".to_owned(),
                    "-L".to_owned(),
                    "/usr/riscv64-linux-gnu".to_owned(),
                ];
            }
            "x86_64-pc-windows-gnu" => {
                // We are cross-compiling for Windows. Run tests in wine.
                self.runner = vec!["wine".to_owned()];
            }
            _ => {
                eprintln!("Unknown non-native platform");
            }
        }
    }

    pub(crate) fn run_with_runner(&self, program: impl AsRef<OsStr>) -> Command {
        if self.runner.is_empty() {
            Command::new(program)
        } else {
            let mut runner_iter = self.runner.iter();
            let mut cmd = Command::new(runner_iter.next().unwrap());
            cmd.args(runner_iter);
            cmd.arg(program);
            cmd
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
        dirs.build_dir.join(self.target)
    }

    #[must_use]
    fn base_cmd(&self, command: &str, cargo: &Path, dirs: &Dirs) -> Command {
        let mut cmd = Command::new(cargo);

        cmd.arg(command)
            .arg("--manifest-path")
            .arg(self.manifest_path(dirs))
            .arg("--target-dir")
            .arg(self.target_dir(dirs))
            .arg("--locked")
            // bootstrap sets both RUSTC and RUSTC_WRAPPER to the same wrapper. RUSTC is already
            // respected by the rustc-clif wrapper, but RUSTC_WRAPPER will misinterpret rustc-clif
            // as filename, so we need to unset it.
            .env_remove("RUSTC_WRAPPER");

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
        rustflags_to_cmd_env(&mut cmd, "RUSTFLAGS", &compiler.rustflags);
        rustflags_to_cmd_env(&mut cmd, "RUSTDOCFLAGS", &compiler.rustdocflags);
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
    let status = cmd.spawn().unwrap().wait().unwrap();
    if !status.success() {
        eprintln!("{cmd:?} exited with status {:?}", status);
        process::exit(1);
    }
}

/// Create the specified directory if it doesn't exist yet and delete all contents.
pub(crate) fn ensure_empty_dir(path: &Path) {
    fs::create_dir_all(path).unwrap();
    let read_dir = match fs::read_dir(&path) {
        Ok(read_dir) => read_dir,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return;
        }
        Err(err) => {
            panic!("Failed to read contents of {path}: {err}", path = path.display())
        }
    };
    for entry in read_dir {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            match fs::remove_dir_all(entry.path()) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => panic!("Failed to remove {path}: {err}", path = entry.path().display()),
            }
        } else {
            match fs::remove_file(entry.path()) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => panic!("Failed to remove {path}: {err}", path = entry.path().display()),
            }
        }
    }
}

pub(crate) fn copy_dir_recursively(from: &Path, to: &Path) {
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let filename = entry.file_name();
        if filename == "." || filename == ".." {
            continue;
        }
        let src = from.join(&filename);
        let dst = to.join(&filename);
        if entry.metadata().unwrap().is_dir() {
            fs::create_dir(&dst).unwrap_or_else(|e| panic!("failed to create {dst:?}: {e}"));
            copy_dir_recursively(&src, &dst);
        } else {
            fs::copy(&src, &dst).unwrap_or_else(|e| panic!("failed to copy {src:?}->{dst:?}: {e}"));
        }
    }
}

static IN_GROUP: AtomicBool = AtomicBool::new(false);
pub(crate) struct LogGroup {
    is_gha: bool,
}

impl LogGroup {
    pub(crate) fn guard(name: &str) -> LogGroup {
        let is_gha = env::var("GITHUB_ACTIONS").is_ok();

        assert!(!IN_GROUP.swap(true, Ordering::SeqCst));
        if is_gha {
            eprintln!("::group::{name}");
        }

        LogGroup { is_gha }
    }
}

impl Drop for LogGroup {
    fn drop(&mut self) {
        if self.is_gha {
            eprintln!("::endgroup::");
        }
        IN_GROUP.store(false, Ordering::SeqCst);
    }
}
