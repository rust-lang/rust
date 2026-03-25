use std::path::{Path, PathBuf};

/// Encapsulates paths to the Triton build used by the MLIR/Triton codegen backend.
///
/// Triton is built at a fixed location (`target/build/triton-build/`) so that both
/// `rustc_llvm` (which builds it) and `rustc_mlir` (which consumes it) agree on the path.
pub struct Triton {
    /// Root of the Triton cmake output directory.
    /// cmake artifacts are in `<out_dir>/build/`.
    pub out_dir: PathBuf,

    /// Path to the Triton source tree (`src/triton/`).
    source: PathBuf,
}

impl Triton {
    pub fn new(root_dir: &Path, _target_dir: &Path) -> Self {
        let out_dir = root_dir.join("target/build/triton-build");
        let source = root_dir.join("src/triton");
        Triton { out_dir, source }
    }

    /// Path to the Triton source directory (`src/triton/`).
    pub fn source_dir(&self) -> &Path {
        &self.source
    }

    /// Directory containing `libtriton.a` (i.e. `<out_dir>/build/`).
    pub fn link_dir(&self) -> PathBuf {
        self.out_dir.join("build")
    }

    /// List of linker arguments for Triton (e.g. `"static=triton"`).
    pub fn link_libs(&self) -> Vec<String> {
        vec!["static=triton".to_string()]
    }
}
