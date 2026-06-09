// When two object archives with the same filename are present, an iterator is supposed to
// inspect each object, recognize the duplication and extract each one to a different directory.
// This test checks that this duplicate handling behaviour has not been broken.
// See https://github.com/rust-lang/rust/pull/24439

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{cc, is_windows_msvc, llvm_ar, rfs, run, rustc};

fn main() {
    rfs::create_dir("a");
    rfs::create_dir("b");
    compile_obj_force_foo("a", "foo");
    compile_obj_force_foo("b", "bar");
    let mut ar = llvm_ar();
    ar.obj_to_ar().arg("libfoo.a");
    if is_windows_msvc() {
        ar.arg("a/foo.obj").arg("b/foo.obj").run();
    } else {
        ar.arg("a/foo.o").arg("b/foo.o").run();
    }
    rustc().input("foo.rs").run();
    rustc().input("bar.rs").run();
    run("bar");
}

#[track_caller]
pub fn compile_obj_force_foo(dir: &str, lib_name: &str) {
    let obj_file = if is_windows_msvc() { format!("{dir}/foo") } else { format!("{dir}/foo.o") };
    let src = format!("{lib_name}.c");
    if is_windows_msvc() {
        cc().arg("-c").out_exe(&obj_file).input(src).run();
    } else {
        cc().arg("-v").arg("-c").out_exe(&obj_file).input(src).run();
    };
}
