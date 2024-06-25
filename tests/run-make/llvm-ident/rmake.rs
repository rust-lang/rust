//@ only-linux
//@ ignore-cross-compile

use run_make_support::llvm::llvm_bin_dir;
use run_make_support::{cmd, env_var, llvm_filecheck, read_dir, rustc, source_root};

use std::ffi::OsStr;

fn main() {
    // `-Ccodegen-units=16 -Copt-level=2` is used here to trigger thin LTO
    // across codegen units to test deduplication of the named metadata
    // (see `LLVMRustPrepareThinLTOImport` for details).
    rustc()
        .emit("link,obj")
        .arg("-")
        .arg("-Csave-temps")
        .codegen_units(16)
        .opt_level("2")
        .target(&env_var("TARGET"))
        .stdin("fn main(){}")
        .run();

    // `llvm-dis` is used here since `--emit=llvm-ir` does not emit LLVM IR
    // for temporary outputs.
    let mut files = Vec::new();
    read_dir(".", |path| {
        if path.is_file() && path.extension().is_some_and(|ext| ext == OsStr::new("bc")) {
            files.push(path.to_path_buf());
        }
    });
    cmd(llvm_bin_dir().join("llvm-dis")).args(&files).run();

    // Check LLVM IR files (including temporary outputs) have `!llvm.ident`
    // named metadata, reusing the related codegen test.
    let llvm_ident_path = source_root().join("tests/codegen/llvm-ident.rs");
    read_dir(".", |path| {
        if path.is_file() && path.extension().is_some_and(|ext| ext == OsStr::new("ll")) {
            llvm_filecheck().input_file(path).arg(&llvm_ident_path).run();
        }
    });
}
