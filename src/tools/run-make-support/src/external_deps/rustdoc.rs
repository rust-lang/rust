use std::ffi::OsStr;
use std::path::Path;

use crate::command::Command;
use crate::env::{env_var, env_var_os};
use crate::util::set_host_rpath;

/// Construct a plain `rustdoc` invocation with no flags set.
#[track_caller]
pub fn bare_rustdoc() -> Rustdoc {
    Rustdoc::bare()
}

/// Construct a new `rustdoc` invocation with `-L $(TARGET_RPATH_DIR)` set.
#[track_caller]
pub fn rustdoc() -> Rustdoc {
    Rustdoc::new()
}

#[derive(Debug)]
#[must_use]
pub struct Rustdoc {
    cmd: Command,
}

crate::macros::impl_common_helpers!(Rustdoc);

#[track_caller]
fn setup_common() -> Command {
    let rustdoc = env_var("RUSTDOC");
    let mut cmd = Command::new(rustdoc);
    set_host_rpath(&mut cmd);
    cmd
}

impl Rustdoc {
    /// Construct a bare `rustdoc` invocation.
    #[track_caller]
    pub fn bare() -> Self {
        let cmd = setup_common();
        Self { cmd }
    }

    /// Construct a `rustdoc` invocation with `-L $(TARGET_RPATH_DIR)` set.
    #[track_caller]
    pub fn new() -> Self {
        let mut cmd = setup_common();
        cmd.arg("-L").arg(env_var_os("TARGET_RPATH_DIR"));
        Self { cmd }
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

    /// Specify output directory.
    #[doc(alias = "output")]
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

    /// Specify a stdin input buffer.
    pub fn stdin_buf<I: AsRef<[u8]>>(&mut self, input: I) -> &mut Self {
        self.cmd.stdin_buf(input);
        self
    }

    /// Specify the edition year.
    pub fn edition(&mut self, edition: &str) -> &mut Self {
        self.cmd.arg("--edition");
        self.cmd.arg(edition);
        self
    }

    /// Specify the target triple, or a path to a custom target json spec file.
    pub fn target<S: AsRef<str>>(&mut self, target: S) -> &mut Self {
        let target = target.as_ref();
        self.cmd.arg(format!("--target={target}"));
        self
    }

    /// Specify the crate type.
    pub fn crate_type(&mut self, crate_type: &str) -> &mut Self {
        self.cmd.arg("--crate-type");
        self.cmd.arg(crate_type);
        self
    }

    /// Specify the crate name.
    pub fn crate_name<S: AsRef<OsStr>>(&mut self, name: S) -> &mut Self {
        self.cmd.arg("--crate-name");
        self.cmd.arg(name.as_ref());
        self
    }

    /// Add a directory to the library search path. It corresponds to the `-L`
    /// rustdoc option.
    pub fn library_search_path<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-L");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify the output format.
    pub fn output_format(&mut self, format: &str) -> &mut Self {
        self.cmd.arg("--output-format");
        self.cmd.arg(format);
        self
    }

    /// Specify type(s) of output files to generate.
    pub fn emit<S: AsRef<str>>(&mut self, kinds: S) -> &mut Self {
        let kinds = kinds.as_ref();
        self.cmd.arg(format!("--emit={kinds}"));
        self
    }
}
