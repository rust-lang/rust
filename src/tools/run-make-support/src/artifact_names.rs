//! A collection of helpers to construct artifact names, such as names of dynamic or static
//! libraries which are target-dependent.

use crate::target;
use crate::targets::is_msvc;

/// Construct the static library name based on the target.
#[track_caller]
#[must_use]
pub fn static_lib_name(name: &str) -> String {
    assert!(!name.contains(char::is_whitespace), "static library name cannot contain whitespace");

    if is_msvc() { format!("{name}.lib") } else { format!("lib{name}.a") }
}

/// Construct the dynamic library name based on the target.
#[track_caller]
#[must_use]
pub fn dynamic_lib_name(name: &str) -> String {
    assert!(!name.contains(char::is_whitespace), "dynamic library name cannot contain whitespace");

    format!("{}{name}.{}", dynamic_lib_prefix(), dynamic_lib_extension())
}

fn dynamic_lib_prefix() -> &'static str {
    if target().contains("windows") { "" } else { "lib" }
}

/// Construct the dynamic library extension based on the target.
#[must_use]
pub fn dynamic_lib_extension() -> &'static str {
    let target = target();

    if target.contains("apple") {
        "dylib"
    } else if target.contains("windows") {
        "dll"
    } else {
        "so"
    }
}

/// Construct the name of the import library for the dynamic library, exclusive to MSVC and accepted
/// by link.exe.
#[track_caller]
#[must_use]
pub fn msvc_import_dynamic_lib_name(name: &str) -> String {
    assert!(is_msvc(), "this function is exclusive to MSVC");
    assert!(!name.contains(char::is_whitespace), "import library name cannot contain whitespace");

    format!("{name}.dll.lib")
}

/// Construct the name of a rust library (rlib).
#[track_caller]
#[must_use]
pub fn rust_lib_name(name: &str) -> String {
    format!("lib{name}.rlib")
}

/// Construct the binary (executable) name based on the target.
#[track_caller]
#[must_use]
pub fn bin_name(name: &str) -> String {
    let target = target();

    if target.contains("windows") {
        format!("{name}.exe")
    } else if target.contains("uefi") {
        format!("{name}.efi")
    } else if target.contains("wasm") {
        format!("{name}.wasm")
    } else if target.contains("nvptx") {
        format!("{name}.ptx")
    } else {
        name.to_string()
    }
}
