// In order to prevent temporary files from overwriting each other in parallel
// compilation, rustc was changed to mix an extra filename with temporary
// outputs. However, as this is a similar behavior with the codegen flag
// -C extra-filename, this test checks that the manually passed flag
// is not overwritten by this feature, and that the output files
// are named as expected.
// See https://github.com/rust-lang/rust/pull/15686

//@ ignore-cross-compile (relocations in generic ELF against `arm-unknown-linux-gnueabihf`)

use run_make_support::{bin_name, cwd, has_prefix, has_suffix, rfs, rustc, shallow_find_files};

fn main() {
    rustc().extra_filename("bar").input("foo.rs").arg("-Csave-temps").run();
    let object_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "foobar.foo") && has_suffix(path, "0.rcgu.o")
    });
    let object_file = object_files.get(0).unwrap();
    rfs::remove_file(object_file);
    rfs::remove_file(bin_name("foobar"));
}
