// This file has very long lines, but there is no way to avoid it as we are testing
// long crate names. so:
// ignore-tidy-linelength

// A variant of the smoke test to check that link time optimization
// (LTO) is accepted by the compiler, and that
// passing its various flags still results in successful compilation, even for very long crate names.
// See https://github.com/rust-lang/rust/issues/49914

//@ ignore-cross-compile

use std::fs;

use run_make_support::{rfs, rustc};

// This test make sure we don't get such following error:
// error: could not write output to generated_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_crate_name.generated_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_crate_name.9384edb61bfd127c-cgu.0.rcgu.o: File name too long
// as reported in issue #49914
fn main() {
    let lto_flags = ["-Clto", "-Clto=yes", "-Clto=off", "-Clto=thin", "-Clto=fat"];
    let aux_file = "generated_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_crate_name.rs";
    // The auxiliary file is used to test long crate names.
    // The file name is intentionally long to test the handling of long filenames.
    // We don't commit it to avoid issues with Windows paths which have known limitations for the full path length.
    // Posix usually only have a limit for the length of the file name.
    rfs::write(aux_file, "#![crate_type = \"rlib\"]\n");

    for flag in lto_flags {
        rustc().input(aux_file).arg(flag).run();
        rustc().input("main.rs").arg(flag).run();
    }
}
