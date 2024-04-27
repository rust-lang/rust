use std::env;
use std::path::Path;
use std::process::{Command, Output};

use crate::{handle_failed_output, set_host_rpath};

/// Construct a plain `rustdoc` invocation with no flags set.
pub fn bare_rustdoc() -> Rustdoc {
    Rustdoc::bare()
}

/// Construct a new `rustdoc` invocation with `-L $(TARGET_RPATH_DIR)` set.
pub fn rustdoc() -> Rustdoc {
    Rustdoc::new()
}

#[derive(Debug)]
pub struct Rustdoc {
    cmd: Command,
}

crate::impl_common_helpers!(Rustdoc);

fn setup_common() -> Command {
    let rustdoc = env::var("RUSTDOC").unwrap();
    let mut cmd = Command::new(rustdoc);
    set_host_rpath(&mut cmd);
    cmd
}

impl Rustdoc {
    /// Construct a bare `rustdoc` invocation.
    pub fn bare() -> Self {
        let cmd = setup_common();
        Self { cmd }
    }

    /// Construct a `rustdoc` invocation with `-L $(TARGET_RPATH_DIR)` set.
    pub fn new() -> Self {
        let mut cmd = setup_common();
        let target_rpath_dir = env::var_os("TARGET_RPATH_DIR").unwrap();
        cmd.arg(format!("-L{}", target_rpath_dir.to_string_lossy()));
        Self { cmd }
    }

    /// Specify path to the input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify output directory.
    pub fn out_dir<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("--out-dir").arg(path.as_ref());
        self
    }

    /// Given a `path`, pass `@{path}` to `rustdoc` as an
    /// [arg file](https://doc.rust-lang.org/rustdoc/command-line-arguments.html#path-load-command-line-flags-from-a-path).
    pub fn arg_file<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(format!("@{}", path.as_ref().display()));
        self
    }

    #[track_caller]
    pub fn run_fail_assert_exit_code(&mut self, code: i32) -> Output {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.cmd.output().unwrap();
        if output.status.code().unwrap() != code {
            handle_failed_output(&self.cmd, output, caller_line_number);
        }
        output
    }
}
