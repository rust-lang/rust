// Previously, rustc mandated that cdylibs could only link against rlibs as dependencies,
// making linkage between cdylibs and dylibs impossible. After this was changed in #68448,
// this test attempts to link both `foo` (a cdylib) and `bar` (a dylib) and checks that
// both compilation and execution are successful.
// See https://github.com/rust-lang/rust/pull/68448

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{
    bin_name, cc, dynamic_lib_extension, dynamic_lib_name, filename_contains, has_extension,
    has_prefix, has_suffix, is_windows_msvc, msvc_import_dynamic_lib_name, path, run, rustc,
    shallow_find_files, target,
};

fn main() {
    rustc().arg("-Cprefer-dynamic").input("bar.rs").run();
    rustc().input("foo.rs").run();
    let sysroot = rustc().print("sysroot").run().stdout_utf8();
    let sysroot = sysroot.trim();
    let target_sysroot = path(sysroot).join("lib/rustlib").join(target()).join("lib");
    if is_windows_msvc() {
        let mut libs = shallow_find_files(&target_sysroot, |path| {
            has_prefix(path, "libstd-") && has_suffix(path, ".dll.lib")
        });
        libs.push(path(msvc_import_dynamic_lib_name("foo")));
        libs.push(path(msvc_import_dynamic_lib_name("bar")));
        cc().input("foo.c").args(&libs).out_exe("foo").run();
    } else {
        let stdlibs = shallow_find_files(&target_sysroot, |path| {
            has_extension(path, dynamic_lib_extension()) && filename_contains(path, "std")
        });
        cc().input("foo.c")
            .args(&[dynamic_lib_name("foo"), dynamic_lib_name("bar")])
            .arg(stdlibs.get(0).unwrap())
            .library_search_path(&target_sysroot)
            .output(bin_name("foo"))
            .run();
    }
    run("foo");
}
