use std::env;
use std::path::Path;
use std::process::{Command, Output};

use crate::{handle_failed_output, tmp_dir};

/// Construct a new `rustc` invocation.
pub fn rustc() -> Rustc {
    Rustc::new()
}

/// Construct a new `rustc` aux-build invocation.
pub fn aux_build() -> Rustc {
    Rustc::new_aux_build()
}

/// A `rustc` invocation builder.
#[derive(Debug)]
pub struct Rustc {
    cmd: Command,
}

fn setup_common() -> Command {
    let rustc = env::var("RUSTC").unwrap();
    let mut cmd = Command::new(rustc);
    cmd.arg("--out-dir").arg(tmp_dir()).arg("-L").arg(tmp_dir());
    cmd
}

impl Rustc {
    // `rustc` invocation constructor methods

    /// Construct a new `rustc` invocation.
    pub fn new() -> Self {
        let cmd = setup_common();
        Self { cmd }
    }

    /// Construct a new `rustc` invocation with `aux_build` preset (setting `--crate-type=lib`).
    pub fn new_aux_build() -> Self {
        let mut cmd = setup_common();
        cmd.arg("--crate-type=lib");
        Self { cmd }
    }

    // Argument provider methods

    /// Configure the compilation environment.
    pub fn cfg(&mut self, s: &str) -> &mut Self {
        self.cmd.arg("--cfg");
        self.cmd.arg(s);
        self
    }

    /// Specify default optimization level `-O` (alias for `-C opt-level=2`).
    pub fn opt(&mut self) -> &mut Self {
        self.cmd.arg("-O");
        self
    }

    /// Specify type(s) of output files to generate.
    pub fn emit(&mut self, kinds: &str) -> &mut Self {
        self.cmd.arg(format!("--emit={kinds}"));
        self
    }

    /// Specify where an external library is located.
    pub fn extern_<P: AsRef<Path>>(&mut self, crate_name: &str, path: P) -> &mut Self {
        assert!(
            !crate_name.contains(|c: char| c.is_whitespace() || c == '\\' || c == '/'),
            "crate name cannot contain whitespace or path separators"
        );

        let path = path.as_ref().to_string_lossy();

        self.cmd.arg("--extern");
        self.cmd.arg(format!("{crate_name}={path}"));

        self
    }

    /// Specify path to the input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify target triple.
    pub fn target(&mut self, target: &str) -> &mut Self {
        assert!(!target.contains(char::is_whitespace), "target triple cannot contain spaces");
        self.cmd.arg(format!("--target={target}"));
        self
    }

    /// Generic command argument provider. Use `.arg("-Zname")` over `.arg("-Z").arg("arg")`.
    /// This method will panic if a plain `-Z` or `-C` is passed, or if `-Z <name>` or `-C <name>`
    /// is passed (note the space).
    pub fn arg(&mut self, arg: &str) -> &mut Self {
        assert!(
            !(["-Z", "-C"].contains(&arg) || arg.starts_with("-Z ") || arg.starts_with("-C ")),
            "use `-Zarg` or `-Carg` over split `-Z` `arg` or `-C` `arg`"
        );
        self.cmd.arg(arg);
        self
    }

    /// Generic command arguments provider. Use `.arg("-Zname")` over `.arg("-Z").arg("arg")`.
    /// This method will panic if a plain `-Z` or `-C` is passed, or if `-Z <name>` or `-C <name>`
    /// is passed (note the space).
    pub fn args(&mut self, args: &[&str]) -> &mut Self {
        for arg in args {
            assert!(
                !(["-Z", "-C"].contains(&arg) || arg.starts_with("-Z ") || arg.starts_with("-C ")),
                "use `-Zarg` or `-Carg` over split `-Z` `arg` or `-C` `arg`"
            );
        }

        self.cmd.args(args);
        self
    }

    // Command inspection, output and running helper methods

    /// Get the [`Output`][std::process::Output] of the finished `rustc` process.
    pub fn output(&mut self) -> Output {
        self.cmd.output().unwrap()
    }

    /// Run the constructed `rustc` command and assert that it is successfully run.
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

    /// Inspect what the underlying [`Command`] is up to the current construction.
    pub fn inspect(&mut self, f: impl FnOnce(&Command)) -> &mut Self {
        f(&self.cmd);
        self
    }
}
