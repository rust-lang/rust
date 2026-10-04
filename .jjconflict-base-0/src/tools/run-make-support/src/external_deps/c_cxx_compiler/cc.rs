use std::path::Path;

use crate::command::Command;
use crate::{env_var, is_windows_msvc};

/// Construct a new platform-specific C compiler invocation.
///
/// WARNING: This means that what flags are accepted by the underlying C compiler is
/// platform- AND compiler-specific. Consult the relevant docs for `gcc`, `clang` and `mvsc`.
#[track_caller]
pub fn cc() -> Cc {
    Cc::new()
}

/// Construct a new platform-specific CXX compiler invocation.
/// CXX_DEFAULT_FLAGS is passed from compiletest.
#[track_caller]
pub fn cxx() -> Cc {
    Cc::new_cxx()
}

/// A platform-specific C compiler invocation builder. The specific C compiler used is
/// passed down from compiletest.
#[derive(Debug)]
#[must_use]
pub struct Cc {
    cmd: Command,
}

crate::macros::impl_common_helpers!(Cc);

impl Cc {
    /// Construct a new platform-specific C compiler invocation.
    ///
    /// WARNING: This means that what flags are accepted by the underlying C compile is
    /// platform- AND compiler-specific. Consult the relevant docs for `gcc`, `clang` and `mvsc`.
    #[track_caller]
    pub fn new() -> Self {
        let compiler = env_var("CC");

        let mut cmd = Command::new(compiler);

        let default_cflags = env_var("CC_DEFAULT_FLAGS");
        for flag in default_cflags.split(char::is_whitespace) {
            cmd.arg(flag);
        }

        Self { cmd }
    }

    /// Construct a new platform-specific CXX compiler invocation.
    /// CXX_DEFAULT_FLAGS is passed from compiletest.
    #[track_caller]
    pub fn new_cxx() -> Self {
        let compiler = env_var("CXX");

        let mut cmd = Command::new(compiler);

        let default_cflags = env_var("CXX_DEFAULT_FLAGS");
        for flag in default_cflags.split(char::is_whitespace) {
            cmd.arg(flag);
        }

        Self { cmd }
    }

    /// Specify path of the input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Adds directories to the list that the linker searches for libraries.
    /// Equivalent to `-L`.
    pub fn library_search_path<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-L");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify `-o` or `-Fe`/`-Fo` depending on platform/compiler.
    pub fn out_exe(&mut self, name: &str) -> &mut Self {
        let mut path = std::path::PathBuf::from(name);

        if is_windows_msvc() {
            path.set_extension("exe");
            let fe_path = path.clone();
            path.set_extension("");
            path.set_extension("obj");
            let fo_path = path;
            self.cmd.arg(format!("-Fe:{}", fe_path.to_str().unwrap()));
            self.cmd.arg(format!("-Fo:{}", fo_path.to_str().unwrap()));
        } else {
            self.cmd.arg("-o");
            self.cmd.arg(name);
        }

        self
    }

    /// Specify path of the output binary.
    pub fn output<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Optimize the output.
    /// Equivalent to `-O3` for GNU-compatible linkers or `-O2` for MSVC linkers.
    pub fn optimize(&mut self) -> &mut Self {
        if is_windows_msvc() {
            self.cmd.arg("-O2");
        } else {
            self.cmd.arg("-O3");
        }
        self
    }
}
