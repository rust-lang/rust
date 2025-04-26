// macOS (and iOS) has a concept of universal (fat) binaries which contain code for multiple CPU
// architectures in the same file. Apple is migrating from x86_64 to aarch64 CPUs,
// so for the next few years it will be important for macOS developers to
// build "fat" binaries (executables and cdylibs).

// Rustc used to be unable to handle these special libraries, which was fixed in #98736. If
// compilation in this test is successful, the native fat library was successfully linked to.
// See https://github.com/rust-lang/rust/issues/55235

//@ only-apple

use run_make_support::{cc, llvm_ar, rustc};

fn main() {
    cc().args(&["-arch", "arm64", "-arch", "x86_64", "native-library.c", "-c"])
        .out_exe("native-library.o")
        .run();
    llvm_ar().obj_to_ar().output_input("libnative-library.a", "native-library.o").run();
    rustc().input("lib.rs").crate_type("lib").arg("-lstatic=native-library").run();
}
