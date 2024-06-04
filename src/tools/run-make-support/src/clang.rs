use std::path::Path;
use std::process::Command;

use crate::{bin_name, env_var, handle_failed_output, tmp_dir};

/// Construct a new `clang` invocation. `clang` is not always available for all targets.
pub fn clang() -> Clang {
    Clang::new()
}

/// A `clang` invocation builder.
#[derive(Debug)]
pub struct Clang {
    cmd: Command,
}

crate::impl_common_helpers!(Clang);

impl Clang {
    /// Construct a new `clang` invocation. `clang` is not always available for all targets.
    pub fn new() -> Self {
        let clang = env_var("CLANG");
        let cmd = Command::new(clang);
        Self { cmd }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify the name of the executable. The executable will be placed under `$TMPDIR`, and the
    /// extension will be determined by [`bin_name`].
    pub fn out_exe(&mut self, name: &str) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(tmp_dir().join(bin_name(name)));
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

    /// Get the [`Output`][::std::process::Output] of the finished process.
    pub fn command_output(&mut self) -> ::std::process::Output {
        self.cmd.output().expect("failed to get output of finished process")
    }
}
