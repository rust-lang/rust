use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub use wasmparser;

pub fn out_dir() -> PathBuf {
    env::var_os("TMPDIR").unwrap().into()
}

fn setup_common_build_cmd() -> Command {
    let rustc = env::var("RUSTC").unwrap();
    let mut cmd = Command::new(rustc);
    cmd.arg("--out-dir").arg(out_dir()).arg("-L").arg(out_dir());
    cmd
}

fn handle_failed_output(cmd: &str, output: Output, caller_line_number: u32) -> ! {
    eprintln!("command failed at line {caller_line_number}");
    eprintln!("{cmd}");
    eprintln!("output status: `{}`", output.status);
    eprintln!("=== STDOUT ===\n{}\n\n", String::from_utf8(output.stdout).unwrap());
    eprintln!("=== STDERR ===\n{}\n\n", String::from_utf8(output.stderr).unwrap());
    std::process::exit(1)
}

pub fn rustc() -> RustcInvocationBuilder {
    RustcInvocationBuilder::new()
}

pub fn aux_build() -> AuxBuildInvocationBuilder {
    AuxBuildInvocationBuilder::new()
}

#[derive(Debug)]
pub struct RustcInvocationBuilder {
    cmd: Command,
}

impl RustcInvocationBuilder {
    fn new() -> Self {
        let cmd = setup_common_build_cmd();
        Self { cmd }
    }

    pub fn arg(&mut self, arg: &str) -> &mut RustcInvocationBuilder {
        self.cmd.arg(arg);
        self
    }

    pub fn args(&mut self, args: &[&str]) -> &mut RustcInvocationBuilder {
        self.cmd.args(args);
        self
    }

    #[track_caller]
    pub fn run(&mut self) -> Output {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.cmd.output().unwrap();
        if !output.status.success() {
            handle_failed_output(&format!("{:#?}", self.cmd), output, caller_line_number);
        }
        output
    }
}

#[derive(Debug)]
pub struct AuxBuildInvocationBuilder {
    cmd: Command,
}

impl AuxBuildInvocationBuilder {
    fn new() -> Self {
        let mut cmd = setup_common_build_cmd();
        cmd.arg("--crate-type=lib");
        Self { cmd }
    }

    pub fn arg(&mut self, arg: &str) -> &mut AuxBuildInvocationBuilder {
        self.cmd.arg(arg);
        self
    }

    #[track_caller]
    pub fn run(&mut self) -> Output {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.cmd.output().unwrap();
        if !output.status.success() {
            handle_failed_output(&format!("{:#?}", self.cmd), output, caller_line_number);
        }
        output
    }
}

fn run_common(bin_name: &str) -> (Command, Output) {
    let target = env::var("TARGET").unwrap();

    let bin_name =
        if target.contains("windows") { format!("{}.exe", bin_name) } else { bin_name.to_owned() };

    let mut bin_path = PathBuf::new();
    bin_path.push(env::var("TMPDIR").unwrap());
    bin_path.push(&bin_name);
    let ld_lib_path_envvar = env::var("LD_LIB_PATH_ENVVAR").unwrap();
    let mut cmd = Command::new(bin_path);
    cmd.env(&ld_lib_path_envvar, {
        let mut paths = vec![];
        paths.push(PathBuf::from(env::var("TMPDIR").unwrap()));
        for p in env::split_paths(&env::var("TARGET_RPATH_ENV").unwrap()) {
            paths.push(p.to_path_buf());
        }
        for p in env::split_paths(&env::var(&ld_lib_path_envvar).unwrap()) {
            paths.push(p.to_path_buf());
        }
        env::join_paths(paths.iter()).unwrap()
    });

    if target.contains("windows") {
        let mut paths = vec![];
        for p in env::split_paths(&std::env::var("PATH").unwrap_or(String::new())) {
            paths.push(p.to_path_buf());
        }
        paths.push(Path::new(&std::env::var("TARGET_RPATH_DIR").unwrap()).to_path_buf());
        cmd.env("PATH", env::join_paths(paths.iter()).unwrap());
    }

    let output = cmd.output().unwrap();
    (cmd, output)
}

/// Run a built binary and make sure it succeeds.
#[track_caller]
pub fn run(bin_name: &str) -> Output {
    let caller_location = std::panic::Location::caller();
    let caller_line_number = caller_location.line();

    let (cmd, output) = run_common(bin_name);
    if !output.status.success() {
        handle_failed_output(&format!("{:#?}", cmd), output, caller_line_number);
    }
    output
}

/// Run a built binary and make sure it fails.
#[track_caller]
pub fn run_fail(bin_name: &str) -> Output {
    let caller_location = std::panic::Location::caller();
    let caller_line_number = caller_location.line();

    let (cmd, output) = run_common(bin_name);
    if output.status.success() {
        handle_failed_output(&format!("{:#?}", cmd), output, caller_line_number);
    }
    output
}
