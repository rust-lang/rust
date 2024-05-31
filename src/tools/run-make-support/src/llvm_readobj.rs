use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::drop_bomb::DropBomb;
use crate::handle_failed_output;

/// Construct a new `llvm-readobj` invocation. This assumes that `llvm-readobj` is available
/// at `$LLVM_BIN_DIR/llvm-readobj`.
pub fn llvm_readobj() -> LlvmReadobj {
    LlvmReadobj::new()
}

/// A `llvm-readobj` invocation builder.
#[must_use]
#[derive(Debug)]
pub struct LlvmReadobj {
    cmd: Command,
    drop_bomb: DropBomb,
}

crate::impl_common_helpers!(LlvmReadobj);

impl LlvmReadobj {
    /// Construct a new `llvm-readobj` invocation. This assumes that `llvm-readobj` is available
    /// at `$LLVM_BIN_DIR/llvm-readobj`.
    pub fn new() -> Self {
        let llvm_bin_dir = env::var("LLVM_BIN_DIR")
            .expect("`LLVM_BIN_DIR` not specified, but this is required to find `llvm-readobj`");
        let llvm_bin_dir = PathBuf::from(llvm_bin_dir);
        let llvm_readobj = llvm_bin_dir.join("llvm-readobj");
        let cmd = Command::new(llvm_readobj);
        Self { cmd, drop_bomb: DropBomb::arm("llvm-readobj invocation must be executed") }
    }

    /// Provide an input file.
    pub fn input<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.arg(path.as_ref());
        self
    }

    /// Pass `--file-header` to display file headers.
    pub fn file_header(&mut self) -> &mut Self {
        self.cmd.arg("--file-header");
        self
    }

    /// Get the [`Output`][::std::process::Output] of the finished process.
    #[track_caller]
    pub fn command_output(&mut self) -> ::std::process::Output {
        self.cmd.output().expect("failed to get output of finished process")
    }
}
