//@ only-linux
//@ ignore-cross-compile

use run_make_support::llvm::llvm_bin_dir;
use run_make_support::{
    cmd, env_var, has_extension, llvm_filecheck, rustc, shallow_find_files, source_root,
};

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
        .stdin_buf("fn main(){}")
        .run();

    // `llvm-dis` is used here since `--emit=llvm-ir` does not emit LLVM IR
    // for temporary outputs.
    let files = shallow_find_files(".", |path| has_extension(path, "bc"));
    cmd(llvm_bin_dir().join("llvm-dis")).args(files).run();

    // Check LLVM IR files (including temporary outputs) have `!llvm.ident`
    // named metadata, reusing the related codegen test.
    let llvm_ident_path = source_root().join("tests/codegen-llvm/llvm-ident.rs");
    let files = shallow_find_files(".", |path| has_extension(path, "ll"));
    for file in files {
        llvm_filecheck().input_file(file).arg(&llvm_ident_path).run();
    }
}
