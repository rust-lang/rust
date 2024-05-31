use std::env;
use std::path::Path;
use std::process::Command;

use crate::drop_bomb::DropBomb;
use crate::{bin_name, handle_failed_output, tmp_dir};

/// Construct a new `clang` invocation. `clang` is not always available for all targets.
pub fn clang() -> Clang {
    Clang::new()
}

/// A `clang` invocation builder.
#[must_use]
#[derive(Debug)]
pub struct Clang {
    cmd: Command,
    drop_bomb: DropBomb,
}

crate::impl_common_helpers!(Clang);

impl Clang {
    /// Construct a new `clang` invocation. `clang` is not always available for all targets.
    #[must_use]
    pub fn new() -> Self {
        let clang =
            env::var("CLANG").expect("`CLANG` not specified, but this is required to find `clang`");
        let cmd = Command::new(clang);
        Self { cmd, drop_bomb: DropBomb::arm("clang invocation must be executed") }
    }

    /// Provide an input file.
    #[must_use]
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Specify the name of the executable. The executable will be placed under `$TMPDIR`, and the
    /// extension will be determined by [`bin_name`].
    #[must_use]
    pub fn out_exe(&mut self, name: &str) -> &mut Self {
        self.cmd.arg("-o");
        self.cmd.arg(tmp_dir().join(bin_name(name)));
        self
    }

    /// Specify which target triple clang should target.
    #[must_use]
    pub fn target(&mut self, target_triple: &str) -> &mut Self {
        self.cmd.arg("-target");
        self.cmd.arg(target_triple);
        self
    }

    /// Pass `-nostdlib` to disable linking the C standard library.
    #[must_use]
    pub fn no_stdlib(&mut self) -> &mut Self {
        self.cmd.arg("-nostdlib");
        self
    }

    /// Specify architecture.
    #[must_use]
    pub fn arch(&mut self, arch: &str) -> &mut Self {
        self.cmd.arg(format!("-march={arch}"));
        self
    }

    /// Specify LTO settings.
    #[must_use]
    pub fn lto(&mut self, lto: &str) -> &mut Self {
        self.cmd.arg(format!("-flto={lto}"));
        self
    }

    /// Specify which ld to use.
    #[must_use]
    pub fn use_ld(&mut self, ld: &str) -> &mut Self {
        self.cmd.arg(format!("-fuse-ld={ld}"));
        self
    }

    /// Get the [`Output`][::std::process::Output] of the finished process.
    pub fn command_output(&mut self) -> ::std::process::Output {
        self.cmd.output().expect("failed to get output of finished process")
    }
}
