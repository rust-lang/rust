// If a dylib is being produced, the compiler will first check to see if it can
// be created entirely statically before falling back to dynamic dependencies. This
// behavior can be overridden with `-C prefer-dynamic`.
// In this test, bar depends on foo and is compiled fully statically despite the available
// `foo` dynamic library. This allows the main binary to be executed in the final step.
// See https://github.com/rust-lang/rust/commit/3036b001276a6e43409b08b7f2334ce72aeeb036

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{
    cc, cwd, dynamic_lib_name, extra_c_flags, has_extension, is_windows_msvc, rfs, run, rustc,
    shallow_find_files,
};

fn main() {
    rustc().input("foo.rs").run();
    rustc().input("bar.rs").run();
    // On msvc, dynamic libraries are compiled by rustc to:
    // bar.dll     // dylib
    // bar.dll.lib // import library for the dylib
    // bar.dll.exp // export library for the dylib
    // msvc's underlying link.exe requires the import library for the dynamic library as input.
    // That is why the library is bar.dll.lib, not bar.dll.
    let library = if is_windows_msvc() { "bar.dll.lib" } else { &dynamic_lib_name("bar") };
    cc().input("main.c").out_exe("main").arg(library).args(extra_c_flags()).run();
    for rlib in shallow_find_files(cwd(), |path| has_extension(path, "rlib")) {
        rfs::remove_file(rlib);
    }
    rfs::remove_file(dynamic_lib_name("foo"));
    if is_windows_msvc() {
        rfs::remove_file("foo.dll.lib");
    }
    run("main");
}
