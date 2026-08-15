use std::path::Path;

use crate::command::Command;
use crate::{bin_name, cwd, env_var};

/// Construct a new `clang` invocation. `clang` is not always available for all targets.
#[track_caller]
pub fn clang() -> Clang {
    Clang::new()
}

/// A `clang` invocation builder.
#[derive(Debug)]
#[must_use]
pub struct Clang {
    cmd: Command,
}

crate::macros::impl_common_helpers!(Clang);

impl Clang {
    /// Construct a new `clang` invocation. `clang` is not always available for all targets.
    #[track_caller]
    pub fn new() -> Self {
        let clang = env_var("CLANG");
        let mut cmd = Command::new(clang);
        cmd.arg("-L").arg(cwd());
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify the name of the executable. The executable will be placed under the current directory
    /// and the extension will be determined by [`bin_name`].
    pub fn out_exe(&mut self, name: &str) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(bin_name(name));
        self
    }

    /// Specify which target triple clang should target.
    pub fn target(&mut self, target_triple: &str) -> &mut Self {
        self.cmd.arg("-target");
        self.cmd.arg(target_triple);
        self
    }

    /// Pass `-nostdlib` to disable linking the C standard library.
    pub fn no_stdlib(&mut self) -> &mut Self {
        self.cmd.arg("-nostdlib");
        self
    }

    /// Specify architecture.
    pub fn arch(&mut self, arch: &str) -> &mut Self {
        self.cmd.arg(format!("-march={arch}"));
        self
    }

    /// Specify LTO settings.
    pub fn lto(&mut self, lto: &str) -> &mut Self {
        self.cmd.arg(format!("-flto={lto}"));
        self
    }

    /// Specify which ld to use.
    pub fn use_ld(&mut self, ld: &str) -> &mut Self {
        self.cmd.arg(format!("-fuse-ld={ld}"));
        self
    }
}
