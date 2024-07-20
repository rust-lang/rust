use std::path::PathBuf;

use crate::artifact_names::static_lib_name;
use crate::external_deps::cc::cc;
use crate::external_deps::llvm::llvm_ar;
use crate::path_helpers::path;
use crate::targets::is_msvc;

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
