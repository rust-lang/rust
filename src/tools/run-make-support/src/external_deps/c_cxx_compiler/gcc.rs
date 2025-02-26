use std::path::Path;

use crate::command::Command;

/// Construct a gcc invocation.
///
/// WARNING: This assumes *a* `gcc` exists in the environment and is suitable for use.
#[track_caller]
pub fn gcc() -> Gcc {
    Gcc::new()
}

/// A specific `gcc`.
#[derive(Debug)]
#[must_use]
pub struct Gcc {
    cmd: Command,
}

crate::macros::impl_common_helpers!(Gcc);

impl Gcc {
    /// Construct a `gcc` invocation. This assumes that *a* suitable `gcc` is available in the
    /// environment.
    ///
    /// Note that this does **not** prepopulate the `gcc` invocation with `CC_DEFAULT_FLAGS`.
    #[track_caller]
    pub fn new() -> Self {
        let cmd = Command::new("gcc");
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

    /// Specify `-o`.
    pub fn out_exe(&mut self, name: &str) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(name);
        self
    }

    /// Specify path of the output binary.
    pub fn output<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(path.as_ref());
        self
    }

    /// Optimize the output at `-O3`.
    pub fn optimize(&mut self) -> &mut Self {
        self.cmd.arg("-O3");
        self
    }
}
