use std::path::{Path, PathBuf};
use std::process::Command;

/// Encapsulates paths to the LLVM installation used by the build.
///
/// In a bootstrap (`x.py`) build, LLVM is built by bootstrap and `LLVM_CONFIG` is set by the
/// bootstrap build system. For standalone cargo builds, falls back to
/// `target/build/llvm-build/build/bin/llvm-config`.
pub struct Llvm {
    /// Parent of the LLVM cmake build directory.
    /// e.g. `build/<target>/llvm/` for bootstrap, or `target/build/llvm-build/` standalone.
    pub out_dir: PathBuf,

    /// LLVM install/prefix directory (output of `llvm-config --prefix`).
    /// e.g. `build/<target>/llvm/build/` or `target/build/llvm-build/build/`.
    pub install_dir: PathBuf,
}

impl Llvm {
    pub fn new(root_dir: &Path, _target_dir: &Path) -> Self {
        // Bootstrap sets LLVM_CONFIG; fall back to the standalone cargo build location.
        let llvm_config = std::env::var_os("LLVM_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                root_dir.join("target/build/llvm-build/build/bin/llvm-config")
            });

        let output = Command::new(&llvm_config)
            .arg("--prefix")
            .output()
            .unwrap_or_else(|e| panic!("failed to run llvm-config {:?}: {}", llvm_config, e));
        assert!(output.status.success(), "llvm-config --prefix failed");

        let prefix = String::from_utf8(output.stdout).expect("llvm-config output not UTF-8");
        let install_dir = PathBuf::from(prefix.trim());

        // `out_dir` is the parent of `install_dir`:
        //   bootstrap:  install_dir = build/<target>/llvm/build/  → out_dir = build/<target>/llvm/
        //   standalone: install_dir = target/build/llvm-build/build/ → out_dir = target/build/llvm-build/
        let out_dir = install_dir
            .parent()
            .unwrap_or(&install_dir)
            .to_path_buf();

        Llvm { out_dir, install_dir }
    }
}
