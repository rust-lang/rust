use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};

use super::path::RelPath;
use super::rustc_info::{get_cargo_path, get_host_triple, get_rustc_path, get_rustdoc_path};

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
    pub(crate) fn host() -> Compiler {
        Compiler {
            cargo: get_cargo_path(),
            rustc: get_rustc_path(),
            rustdoc: get_rustdoc_path(),
            rustflags: String::new(),
            rustdocflags: String::new(),
            triple: get_host_triple(),
            runner: vec![],
        }
    }

    pub(crate) fn with_triple(triple: String) -> Compiler {
        Compiler {
            cargo: get_cargo_path(),
            rustc: get_rustc_path(),
            rustdoc: get_rustdoc_path(),
            rustflags: String::new(),
            rustdocflags: String::new(),
            triple,
            runner: vec![],
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

    pub(crate) fn source_dir(&self) -> PathBuf {
        self.source.to_path()
    }

    pub(crate) fn manifest_path(&self) -> PathBuf {
        self.source_dir().join("Cargo.toml")
    }

    pub(crate) fn target_dir(&self) -> PathBuf {
        RelPath::BUILD.join(self.target).to_path()
    }

    fn base_cmd(&self, command: &str, cargo: &Path) -> Command {
        let mut cmd = Command::new(cargo);

        cmd.arg(command)
            .arg("--manifest-path")
            .arg(self.manifest_path())
            .arg("--target-dir")
            .arg(self.target_dir());

        cmd
    }

    fn build_cmd(&self, command: &str, compiler: &Compiler) -> Command {
        let mut cmd = self.base_cmd(command, &compiler.cargo);

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

    #[must_use]
    pub(crate) fn fetch(&self, cargo: impl AsRef<Path>) -> Command {
        let mut cmd = Command::new(cargo.as_ref());

        cmd.arg("fetch").arg("--manifest-path").arg(self.manifest_path());

        cmd
    }

    #[must_use]
    pub(crate) fn clean(&self, cargo: &Path) -> Command {
        self.base_cmd("clean", cargo)
    }

    #[must_use]
    pub(crate) fn build(&self, compiler: &Compiler) -> Command {
        self.build_cmd("build", compiler)
    }

    #[must_use]
    pub(crate) fn test(&self, compiler: &Compiler) -> Command {
        self.build_cmd("test", compiler)
    }

    #[must_use]
    pub(crate) fn run(&self, compiler: &Compiler) -> Command {
        self.build_cmd("run", compiler)
    }
}

#[must_use]
pub(crate) fn hyperfine_command(
    warmup: u64,
    runs: u64,
    prepare: Option<&str>,
    a: &str,
    b: &str,
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

    bench.arg(a).arg(b);

    bench
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
    env::var("CI").as_ref().map(|val| &**val) == Ok("true")
}
