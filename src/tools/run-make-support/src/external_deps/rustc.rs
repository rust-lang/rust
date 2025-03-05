use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::str::FromStr as _;

use crate::command::Command;
use crate::env::env_var;
use crate::path_helpers::cwd;
use crate::util::set_host_compiler_dylib_path;
use crate::{is_aix, is_darwin, is_msvc, is_windows, target, uname};

/// Construct a new `rustc` invocation. This will automatically set the library
/// search path as `-L cwd()`. Use [`bare_rustc`] to avoid this.
#[track_caller]
pub fn rustc() -> Rustc {
    Rustc::new()
}

/// Construct a plain `rustc` invocation with no flags set. Note that [`set_host_compiler_dylib_path`]
/// still presets the environment variable `HOST_RUSTC_DYLIB_PATH` by default.
#[track_caller]
pub fn bare_rustc() -> Rustc {
    Rustc::bare()
}

/// A `rustc` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct Rustc {
    cmd: Command,
    target: Option<String>,
}

// Only fill in the target just before execution, so that it can be overridden.
crate::macros::impl_common_helpers!(Rustc, |rustc: &mut Rustc| {
    if let Some(target) = &rustc.target {
        rustc.cmd.arg(&format!("--target={target}"));
    }
});

pub fn rustc_path() -> String {
    env_var("RUSTC")
}

#[track_caller]
fn setup_common() -> Command {
    let mut cmd = Command::new(rustc_path());
    set_host_compiler_dylib_path(&mut cmd);
    cmd
}

impl Rustc {
    // `rustc` invocation constructor methods

    /// Construct a new `rustc` invocation. This will automatically set the library
    /// search path as `-L cwd()` and also the compilation target.
    /// Use [`bare_rustc`] to avoid this.
    #[track_caller]
    pub fn new() -> Self {
        let mut cmd = setup_common();
        cmd.arg("-L").arg(cwd());

        // Automatically default to cross-compilation
        Self { cmd, target: Some(target()) }
    }

    /// Construct a bare `rustc` invocation with no flags set.
    #[track_caller]
    pub fn bare() -> Self {
        let cmd = setup_common();
        Self { cmd, target: None }
    }

    // Argument provider methods

    /// Configure the compilation environment.
    pub fn cfg(&mut self, s: &str) -> &mut Self {
        self.cmd.arg("--cfg");
        self.cmd.arg(s);
        self
    }

    /// Specify default optimization level `-O` (alias for `-C opt-level=3`).
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
    pub fn emit<S: AsRef<str>>(&mut self, kinds: S) -> &mut Self {
        let kinds = kinds.as_ref();
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
    pub fn remap_path_prefix<P: AsRef<Path>, P2: AsRef<Path>>(
        &mut self,
        from: P,
        to: P2,
    ) -> &mut Self {
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

    /// Specify option of `-C symbol-mangling-version`.
    pub fn symbol_mangling_version(&mut self, option: &str) -> &mut Self {
        self.cmd.arg(format!("-Csymbol-mangling-version={option}"));
        self
    }

    /// Specify `-C prefer-dynamic`.
    pub fn prefer_dynamic(&mut self) -> &mut Self {
        self.cmd.arg(format!("-Cprefer-dynamic"));
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

    /// Normalize the line number in the stderr output
    pub fn ui_testing(&mut self) -> &mut Self {
        self.cmd.arg(format!("-Zui-testing"));
        self
    }

    /// Specify the target triple, or a path to a custom target json spec file.
    pub fn target<S: AsRef<str>>(&mut self, target: S) -> &mut Self {
        // We store the target as a separate field, so that it can be specified multiple times.
        // This is in particular useful to override the default target set in Rustc::new().
        self.target = Some(target.as_ref().to_string());
        self
    }

    /// Specify the target CPU.
    pub fn target_cpu<S: AsRef<str>>(&mut self, target_cpu: S) -> &mut Self {
        let target_cpu = target_cpu.as_ref();
        self.cmd.arg(format!("-Ctarget-cpu={target_cpu}"));
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

    /// Add a directory to the library search path with a restriction, where `kind` is a dependency
    /// type. Equivalent to `-L KIND=PATH` in rustc.
    pub fn specific_library_search_path<P: AsRef<Path>>(
        &mut self,
        kind: &str,
        path: P,
    ) -> &mut Self {
        assert!(["dependency", "native", "all", "framework", "crate"].contains(&kind));
        let path = path.as_ref().to_string_lossy();
        self.cmd.arg(format!("-L{kind}={path}"));
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

    /// Specify a stdin input buffer.
    pub fn stdin_buf<I: AsRef<[u8]>>(&mut self, input: I) -> &mut Self {
        self.cmd.stdin_buf(input);
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

    /// Specify `-C debuginfo=...`.
    pub fn debuginfo(&mut self, level: &str) -> &mut Self {
        self.cmd.arg(format!("-Cdebuginfo={level}"));
        self
    }

    /// Specify `-C split-debuginfo={packed,unpacked,off}`.
    pub fn split_debuginfo(&mut self, split_kind: &str) -> &mut Self {
        self.cmd.arg(format!("-Csplit-debuginfo={split_kind}"));
        self
    }

    /// Pass the `--verbose` flag.
    pub fn verbose(&mut self) -> &mut Self {
        self.cmd.arg("--verbose");
        self
    }

    /// `EXTRARSCXXFLAGS`
    pub fn extra_rs_cxx_flags(&mut self) -> &mut Self {
        if is_windows() {
            // So this is a bit hacky: we can't use the DLL version of libstdc++ because
            // it pulls in the DLL version of libgcc, which means that we end up with 2
            // instances of the DW2 unwinding implementation. This is a problem on
            // i686-pc-windows-gnu because each module (DLL/EXE) needs to register its
            // unwind information with the unwinding implementation, and libstdc++'s
            // __cxa_throw won't see the unwinding info we registered with our statically
            // linked libgcc.
            //
            // Now, simply statically linking libstdc++ would fix this problem, except
            // that it is compiled with the expectation that pthreads is dynamically
            // linked as a DLL and will fail to link with a statically linked libpthread.
            //
            // So we end up with the following hack: we link use static:-bundle to only
            // link the parts of libstdc++ that we actually use, which doesn't include
            // the dependency on the pthreads DLL.
            if !is_msvc() {
                self.cmd.arg("-lstatic:-bundle=stdc++");
            };
        } else if is_darwin() {
            self.cmd.arg("-lc++");
        } else if is_aix() {
            self.cmd.arg("-lc++");
            self.cmd.arg("-lc++abi");
        } else {
            if !matches!(&uname()[..], "FreeBSD" | "SunOS" | "OpenBSD") {
                self.cmd.arg("-lstdc++");
            };
        };
        self
    }
}

/// Query the sysroot path corresponding `rustc --print=sysroot`.
#[track_caller]
pub fn sysroot() -> PathBuf {
    let path = rustc().print("sysroot").run().stdout_utf8();
    PathBuf::from_str(path.trim()).unwrap()
}
