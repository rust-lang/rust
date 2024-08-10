// Regression test for https://github.com/rust-lang/rust/issues/107407

use run_make_support::{llvm_ar, rustc, static_lib_name};

fn main() {
    rustc().input("simple_obj.rs").emit("obj").run();
    llvm_ar().obj_to_thin_ar().output_input(static_lib_name("thin_archive"), "simple_obj.o").run();
    rustc().input("rust_archive.rs").run();
    // Disable lld as it ignores the symbol table in the archive file.
    rustc()
        .input("bin.rs") /*.arg("-Zlinker-features=-lld")*/
        .run();
}
