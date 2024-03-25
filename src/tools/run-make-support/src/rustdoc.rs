use std::env;
use std::path::Path;
use std::process::{Command, Output};

use crate::{add_host_rpath_env, handle_failed_output};

pub fn rustdoc() -> RustdocInvocationBuilder {
    RustdocInvocationBuilder::new()
}

#[derive(Debug)]
pub struct RustdocInvocationBuilder {
    cmd: Command,
}

impl RustdocInvocationBuilder {
    fn new() -> Self {
        let cmd = setup_common_rustdoc_build_cmd();
        Self { cmd }
    }

    pub fn arg(&mut self, arg: &str) -> &mut RustdocInvocationBuilder {
        self.cmd.arg(arg);
        self
    }

    pub fn arg_file(&mut self, arg: &Path) -> &mut RustdocInvocationBuilder {
        self.cmd.arg(arg);
        self
    }

    pub fn env(&mut self, key: &str, value: &str) -> &mut RustdocInvocationBuilder {
        self.cmd.env(key, value);
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

    #[track_caller]
    pub fn run_fail(&mut self) -> Output {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.cmd.output().unwrap();
        if output.status.success() {
            handle_failed_output(&format!("{:#?}", self.cmd), output, caller_line_number);
        }
        output
    }

    #[track_caller]
    pub fn run_fail_assert_exit_code(&mut self, code: i32) -> Output {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.cmd.output().unwrap();
        if output.status.code().unwrap() != code {
            handle_failed_output(&format!("{:#?}", self.cmd), output, caller_line_number);
        }
        output
    }
}

fn setup_common_rustdoc_build_cmd() -> Command {
    use std::env::VarError;

    let rustdoc = env::var("RUSTDOC").unwrap();
    let target_rpath_dir = env::var("TARGET_RPATH_DIR").unwrap();

    let mut cmd = Command::new(rustdoc);

    add_host_rpath_env(&mut cmd);

    cmd.arg("-L").arg(target_rpath_dir);

    match std::env::var("RUSTC_LINKER") {
        Ok(rustc_linker) => {
            cmd.arg(&format!("-Clinker='{rustc_linker}'"));
        }
        Err(VarError::NotPresent) => {}
        Err(VarError::NotUnicode(s)) => {
            panic!("RUSTC_LINKER was found, but set to non-unicode string {s:?}");
        }
    }

    cmd
}
