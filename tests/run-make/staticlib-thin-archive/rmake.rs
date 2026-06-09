//@ ignore-cross-compile

// Regression test for https://github.com/rust-lang/rust/issues/107407 which
// checks that rustc can read thin archive. Before the object crate added thin
// archive support rustc would add emit object files to the staticlib and after
// the object crate added thin archive support it would previously crash the
// compiler due to a missing special case for thin archive members.
use run_make_support::{llvm_ar, path, rfs, rust_lib_name, rustc, static_lib_name};

fn main() {
    rfs::create_dir("archive");

    // Build a thin archive
    rustc().input("simple_obj.rs").emit("obj").output("archive/simple_obj.o").run();
    llvm_ar()
        .obj_to_thin_ar()
        .output_input(path("archive").join(static_lib_name("thin_archive")), "archive/simple_obj.o")
        .run();

    // Build an rlib which includes the members of this thin archive
    rustc().input("rust_lib.rs").library_search_path("archive").run();

    // Build a binary which requires a symbol from the thin archive
    rustc().input("bin.rs").extern_("rust_lib", rust_lib_name("rust_lib")).run();
}
