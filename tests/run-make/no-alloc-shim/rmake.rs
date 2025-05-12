// This test checks the compatibility of the interaction between `--emit obj` and
// `#[global_allocator]`, as it is now possible to invoke the latter without the
// allocator shim since #86844. As this feature is unstable, it should fail if
// --cfg check_feature_gate is passed.
// See https://github.com/rust-lang/rust/pull/86844

//@ ignore-cross-compile
// Reason: the compiled binary is executed

//@ ignore-msvc
//FIXME(Oneirical): Getting this to work on MSVC requires passing libcmt.lib to CC,
// which is not trivial to do.
// Tracking issue: https://github.com/rust-lang/rust/issues/128602
// Discussion: https://github.com/rust-lang/rust/pull/128407#discussion_r1702439172

use run_make_support::{cc, has_extension, has_prefix, run, rustc, shallow_find_files};

fn main() {
    rustc().input("foo.rs").crate_type("bin").emit("obj").panic("abort").run();
    let libdir = rustc().print("target-libdir").run().stdout_utf8();
    let libdir = libdir.trim();

    let alloc_libs = shallow_find_files(&libdir, |path| {
        has_prefix(path, "liballoc-") && has_extension(path, "rlib")
    });
    let core_libs = shallow_find_files(&libdir, |path| {
        has_prefix(path, "libcore-") && has_extension(path, "rlib")
    });
    let compiler_builtins_libs = shallow_find_files(libdir, |path| {
        has_prefix(path, "libcompiler_builtins") && has_extension(path, "rlib")
    });

    cc().input("foo.o")
        .out_exe("foo")
        .args(&alloc_libs)
        .args(&core_libs)
        .args(&compiler_builtins_libs)
        .run();
    run("foo");

    // Check that linking without __rust_no_alloc_shim_is_unstable defined fails
    rustc()
        .input("foo.rs")
        .crate_type("bin")
        .emit("obj")
        .panic("abort")
        .cfg("check_feature_gate")
        .run();
    cc().input("foo.o")
        .out_exe("foo")
        .args(&alloc_libs)
        .args(&core_libs)
        .args(&compiler_builtins_libs)
        .run_fail();
}
