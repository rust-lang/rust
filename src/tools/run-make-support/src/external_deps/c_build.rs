use std::path::PathBuf;

use super::cygpath::get_windows_path;
use crate::artifact_names::{dynamic_lib_name, static_lib_name};
use crate::external_deps::cc::cc;
use crate::external_deps::llvm::llvm_ar;
use crate::path_helpers::path;
use crate::targets::{is_darwin, is_msvc, is_windows};

// FIXME(Oneirical): These native build functions should take a Path-based generic.

/// Builds a static lib (`.lib` on Windows MSVC and `.a` for the rest) with the given name.
#[track_caller]
pub fn build_native_static_lib(lib_name: &str) -> PathBuf {
    let obj_file = if is_msvc() { format!("{lib_name}") } else { format!("{lib_name}.o") };
    let src = format!("{lib_name}.c");
    let lib_path = static_lib_name(lib_name);
    if is_msvc() {
        cc().arg("-c").out_exe(&obj_file).input(src).run();
    } else {
        cc().arg("-v").arg("-c").out_exe(&obj_file).input(src).run();
    };
    let obj_file = if is_msvc() {
        PathBuf::from(format!("{lib_name}.obj"))
    } else {
        PathBuf::from(format!("{lib_name}.o"))
    };
    llvm_ar().obj_to_ar().output_input(&lib_path, &obj_file).run();
    path(lib_path)
}

/// Builds a dynamic lib. The filename is computed in a target-dependent manner, relying on
/// [`std::env::consts::DLL_PREFIX`] and [`std::env::consts::DLL_EXTENSION`].
#[track_caller]
pub fn build_native_dynamic_lib(lib_name: &str) -> PathBuf {
    let obj_file = if is_msvc() { format!("{lib_name}") } else { format!("{lib_name}.o") };
    let src = format!("{lib_name}.c");
    let lib_path = dynamic_lib_name(lib_name);
    if is_msvc() {
        cc().arg("-c").out_exe(&obj_file).input(src).run();
    } else {
        cc().arg("-v").arg("-c").out_exe(&obj_file).input(src).run();
    };
    let obj_file = if is_msvc() { format!("{lib_name}.obj") } else { format!("{lib_name}.o") };
    if is_msvc() {
        let mut out_arg = "-out:".to_owned();
        out_arg.push_str(&get_windows_path(&lib_path));
        cc().input(&obj_file).args(&["-link", "-dll", &out_arg]).run();
    } else if is_darwin() {
        cc().out_exe(&lib_path).input(&obj_file).args(&["-dynamiclib", "-Wl,-dylib"]).run();
    } else if is_windows() {
        cc().out_exe(&lib_path)
            .input(&obj_file)
            .args(&["-shared", &format!("-Wl,--out-implib={lib_path}.a")])
            .run();
    } else {
        cc().out_exe(&lib_path).input(&obj_file).arg("-shared").run();
    }
    path(lib_path)
}
