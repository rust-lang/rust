//@ needs-target-std
//
// Regression test for https://github.com/rust-lang/rust/issues/128955#issuecomment-2657811196
// which checks that rustc can read an archive containing LLVM bitcode with a
// newer version from the one rustc links against.
use run_make_support::{llvm_ar, path, rfs, rustc, static_lib_name};

fn main() {
    rfs::create_dir("archive");

    let mut bitcode = b"BC\xC0\xDE".to_vec();
    bitcode.extend(std::iter::repeat(b'a').take(50));
    rfs::write("archive/invalid_bitcode.o", &bitcode);

    llvm_ar()
        .arg("rcuS") // like obj_to_ar() except skips creating a symbol table
        .output_input(
            path("archive").join(static_lib_name("thin_archive")),
            "archive/invalid_bitcode.o",
        )
        .run();

    // Build an rlib which includes the members of this thin archive
    rustc().input("rust_lib.rs").library_search_path("archive").run();
}
