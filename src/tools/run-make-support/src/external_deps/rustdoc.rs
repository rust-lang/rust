use std::ffi::OsStr;
use std::path::Path;

use crate::command::Command;
use crate::env::env_var;
use crate::target;
use crate::util::set_host_compiler_dylib_path;

/// Construct a new `rustdoc` invocation with target automatically set to cross-compile target and
/// with host compiler runtime libs configured. Use [`bare_rustdoc`] to avoid automatically setting
/// cross-compile target.
#[track_caller]
pub fn rustdoc() -> Rustdoc {
    Rustdoc::new()
}

/// Bare `rustdoc` invocation, no args set.
#[track_caller]
pub fn bare_rustdoc() -> Rustdoc {
    Rustdoc::bare()
}

#[derive(Debug)]
#[must_use]
pub struct Rustdoc {
    cmd: Command,
    target: Option<String>,
}

// Only fill in the target just before execution, so that it can be overridden.
crate::macros::impl_common_helpers!(Rustdoc, |rustdoc: &mut Rustdoc| {
    if let Some(target) = &rustdoc.target {
        rustdoc.cmd.arg(&format!("--target={target}"));
    }
});

#[track_caller]
fn setup_common() -> Command {
    let rustdoc = env_var("RUSTDOC");
    let mut cmd = Command::new(rustdoc);
    set_host_compiler_dylib_path(&mut cmd);
    cmd
}

impl Rustdoc {
    /// Construct a new `rustdoc` invocation with target automatically set to cross-compile target
    /// and with host compiler runtime libs configured. Use [`bare_rustdoc`] to avoid automatically
    /// setting cross-compile target.
    #[track_caller]
    pub fn new() -> Self {
        let cmd = setup_common();
        Self { cmd, target: Some(target()) }
    }

    /// Bare `rustdoc` invocation, no args set.
    #[track_caller]
    pub fn bare() -> Self {
        let cmd = setup_common();
        Self { cmd, target: None }
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
        // We store the target as a separate field, so that it can be specified multiple times.
        // This is in particular useful to override the default target set in `Rustdoc::new()`.
        self.target = Some(target.as_ref().to_string());
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
