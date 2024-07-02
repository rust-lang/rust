use command::Command;
use std::ffi::{OsStr, OsString};
use std::path::Path;

use crate::{command, cwd, env_var, set_host_rpath};

/// Construct a new `rustc` invocation.
#[track_caller]
pub fn rustc() -> Rustc {
    Rustc::new()
}

/// Construct a new `rustc` aux-build invocation.
#[track_caller]
pub fn aux_build() -> Rustc {
    Rustc::new_aux_build()
}

/// A `rustc` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct Rustc {
    cmd: Command,
}

crate::impl_common_helpers!(Rustc);

#[track_caller]
fn setup_common() -> Command {
    let rustc = env_var("RUSTC");
    let mut cmd = Command::new(rustc);
    set_host_rpath(&mut cmd);
    cmd.arg("-L").arg(cwd());
    cmd
}

impl Rustc {
    // `rustc` invocation constructor methods

    /// Construct a new `rustc` invocation.
    #[track_caller]
    pub fn new() -> Self {
        let cmd = setup_common();
        Self { cmd }
    }

    /// Construct a new `rustc` invocation with `aux_build` preset (setting `--crate-type=lib`).
    #[track_caller]
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

    /// Specify a specific optimization level.
    pub fn opt_level(&mut self, option: &str) -> &mut Self {
        self.cmd.arg(format!("-Copt-level={option}"));
        self
    }

    /// Incorporate a hashed string to mangled symbols.
    pub fn metadata(&mut self, meta: &str) -> &mut Self {
        self.cmd.arg(format!("-Cmetadata={meta}"));
        self
    }

    /// Add a suffix in each output filename.
    pub fn extra_filename(&mut self, suffix: &str) -> &mut Self {
        self.cmd.arg(format!("-Cextra-filename={suffix}"));
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

    /// Remap source path prefixes in all output.
    pub fn remap_path_prefix<P: AsRef<Path>>(&mut self, from: P, to: P) -> &mut Self {
        let from = from.as_ref().to_string_lossy();
        let to = to.as_ref().to_string_lossy();

        self.cmd.arg("--remap-path-prefix");
        self.cmd.arg(format!("{from}={to}"));

        self
    }

    /// Specify path to the input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    //Adjust the backtrace level, displaying more detailed information at higher levels.
    pub fn set_backtrace_level<R: AsRef<OsStr>>(&mut self, level: R) -> &mut Self {
        self.cmd.env("RUST_BACKTRACE", level);
        self
    }

    /// Specify path to the output file. Equivalent to `-o`` in rustc.
    pub fn output<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify path to the output directory. Equivalent to `--out-dir`` in rustc.
    pub fn out_dir<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("--out-dir");
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

    /// Specify directory path used for profile generation
    pub fn profile_generate<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        let mut arg = OsString::new();
        arg.push("-Cprofile-generate=");
        arg.push(path.as_ref());
        self.cmd.arg(&arg);
        self
    }

    /// Specify directory path used for profile usage
    pub fn profile_use<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        let mut arg = OsString::new();
        arg.push("-Cprofile-use=");
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

    /// Add multiple extra arguments to the linker invocation, via `-Clink-args`.
    pub fn link_args(&mut self, link_args: &str) -> &mut Self {
        self.cmd.arg(format!("-Clink-args={link_args}"));
        self
    }

    /// Specify a stdin input
    pub fn stdin<I: AsRef<[u8]>>(&mut self, input: I) -> &mut Self {
        self.cmd.stdin(input);
        self
    }

    /// Specify the crate name.
    pub fn crate_name<S: AsRef<OsStr>>(&mut self, name: S) -> &mut Self {
        self.cmd.arg("--crate-name");
        self.cmd.arg(name.as_ref());
        self
    }

    /// Specify the linker
    pub fn linker(&mut self, linker: &str) -> &mut Self {
        self.cmd.arg(format!("-Clinker={linker}"));
        self
    }

    /// Specify the linker flavor
    pub fn linker_flavor(&mut self, linker_flavor: &str) -> &mut Self {
        self.cmd.arg(format!("-Clinker-flavor={linker_flavor}"));
        self
    }
}
