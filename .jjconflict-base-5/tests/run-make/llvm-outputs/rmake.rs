//@ needs-target-std
// test that directories get created when emitting llvm bitcode and IR

use std::path::PathBuf;

use run_make_support::{cwd, run_in_tmpdir, rustc};

fn main() {
    let mut path_bc = PathBuf::new();
    let mut path_ir = PathBuf::new();
    run_in_tmpdir(|| {
        let p = cwd();
        path_bc = p.join("nonexistent_dir_bc");
        path_ir = p.join("nonexistent_dir_ir");
        rustc().input("-").stdin_buf("fn main() {}").out_dir(&path_bc).emit("llvm-bc").run();
        rustc().input("-").stdin_buf("fn main() {}").out_dir(&path_ir).emit("llvm-ir").run();
        assert!(path_bc.exists());
        assert!(path_ir.exists());
    });
    assert!(!path_bc.exists());
    assert!(!path_ir.exists());
}
