use std::env;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use crate::{handle_failed_output, set_host_rpath, tmp_dir};

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
    stdin: Option<Box<[u8]>>,
}

crate::impl_common_helpers!(Rustc);

fn setup_common() -> Command {
    let rustc = env::var("RUSTC").unwrap();
    let mut cmd = Command::new(rustc);
    set_host_rpath(&mut cmd);
    cmd.arg("--out-dir").arg(tmp_dir()).arg("-L").arg(tmp_dir());
    cmd
}

impl Rustc {
    // `rustc` invocation constructor methods

    /// Construct a new `rustc` invocation.
    pub fn new() -> Self {
        let cmd = setup_common();
        Self { cmd, stdin: None }
    }

    /// Construct a new `rustc` invocation with `aux_build` preset (setting `--crate-type=lib`).
    pub fn new_aux_build() -> Self {
        let mut cmd = setup_common();
        cmd.arg("--crate-type=lib");
        Self { cmd, stdin: None }
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

    /// Specify a specific optimization level.
    pub fn opt_level(&mut self, option: &str) -> &mut Self {
        self.cmd.arg(format!("-Copt-level={option}"));
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

    /// Specify path to the output file. Equivalent to `-o`` in rustc.
    pub fn output<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(path.as_ref());
        self
    }

    /// This flag defers LTO optimizations to the linker.
    pub fn linker_plugin_lto(&mut self, option: &str) -> &mut Self {
        self.cmd.arg(format!("-Clinker-plugin-lto={option}"));
        self
    }

    /// Specify what happens when the code panics.
    pub fn panic(&mut self, option: &str) -> &mut Self {
        self.cmd.arg(format!("-Cpanic={option}"));
        self
    }

    /// Specify number of codegen units
    pub fn codegen_units(&mut self, units: usize) -> &mut Self {
        self.cmd.arg(format!("-Ccodegen-units={units}"));
        self
    }

    /// Specify directory path used for incremental cache
    pub fn incremental<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        let mut arg = OsString::new();
        arg.push("-Cincremental=");
        arg.push(path.as_ref());
        self.cmd.arg(&arg);
        self
    }

    /// Specify error format to use
    pub fn error_format(&mut self, format: &str) -> &mut Self {
        self.cmd.arg(format!("--error-format={format}"));
        self
    }

    /// Specify json messages printed by the compiler
    pub fn json(&mut self, items: &str) -> &mut Self {
        self.cmd.arg(format!("--json={items}"));
        self
    }

    /// Specify the target triple, or a path to a custom target json spec file.
    pub fn target(&mut self, target: &str) -> &mut Self {
        self.cmd.arg(format!("--target={target}"));
        self
    }

    /// Specify the crate type.
    pub fn crate_type(&mut self, crate_type: &str) -> &mut Self {
        self.cmd.arg("--crate-type");
        self.cmd.arg(crate_type);
        self
    }

    /// Add a directory to the library search path. Equivalent to `-L` in rustc.
    pub fn library_search_path<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-L");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Override the system root. Equivalent to `--sysroot` in rustc.
    pub fn sysroot<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("--sysroot");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify the edition year.
    pub fn edition(&mut self, edition: &str) -> &mut Self {
        self.cmd.arg("--edition");
        self.cmd.arg(edition);
        self
    }

    /// Specify the print request.
    pub fn print(&mut self, request: &str) -> &mut Self {
        self.cmd.arg("--print");
        self.cmd.arg(request);
        self
    }

    /// Add an extra argument to the linker invocation, via `-Clink-arg`.
    pub fn link_arg(&mut self, link_arg: &str) -> &mut Self {
        self.cmd.arg(format!("-Clink-arg={link_arg}"));
        self
    }

    /// Specify a stdin input
    pub fn stdin<I: AsRef<[u8]>>(&mut self, input: I) -> &mut Self {
        self.stdin = Some(input.as_ref().to_vec().into_boxed_slice());
        self
    }

    /// Specify the crate name.
    pub fn crate_name<S: AsRef<OsStr>>(&mut self, name: S) -> &mut Self {
        self.cmd.arg("--crate-name");
        self.cmd.arg(name.as_ref());
        self
    }

    /// Get the [`Output`][::std::process::Output] of the finished process.
    #[track_caller]
    pub fn command_output(&mut self) -> ::std::process::Output {
        // let's make sure we piped all the input and outputs
        self.cmd.stdin(Stdio::piped());
        self.cmd.stdout(Stdio::piped());
        self.cmd.stderr(Stdio::piped());

        if let Some(input) = &self.stdin {
            let mut child = self.cmd.spawn().unwrap();

            {
                let mut stdin = child.stdin.take().unwrap();
                stdin.write_all(input.as_ref()).unwrap();
            }

            child.wait_with_output().expect("failed to get output of finished process")
        } else {
            self.cmd.output().expect("failed to get output of finished process")
        }
    }

    #[track_caller]
    pub fn run_fail_assert_exit_code(&mut self, code: i32) -> Output {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.command_output();
        if output.status.code().unwrap() != code {
            handle_failed_output(&self.cmd, output, caller_line_number);
        }
        output
    }
}
