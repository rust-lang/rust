//! A collection of helpers to construct artifact names, such as names of dynamic or static
//! librarys which are target-dependent.

// FIXME(jieyouxu): convert these to return `PathBuf`s instead of strings!

use crate::targets::is_msvc;

/// Construct the static library name based on the target.
#[must_use]
pub fn static_lib_name(name: &str) -> String {
    assert!(!name.contains(char::is_whitespace), "static library name cannot contain whitespace");

    if is_msvc() { format!("{name}.lib") } else { format!("lib{name}.a") }
}

/// Construct the dynamic library name based on the target.
#[must_use]
pub fn dynamic_lib_name(name: &str) -> String {
    assert!(!name.contains(char::is_whitespace), "dynamic library name cannot contain whitespace");

    format!("{}{name}.{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_EXTENSION)
}

/// Construct the name of the import library for the dynamic library, exclusive to MSVC and
/// accepted by link.exe.
#[track_caller]
#[must_use]
pub fn msvc_import_dynamic_lib_name(name: &str) -> String {
    assert!(is_msvc(), "this function is exclusive to MSVC");
    assert!(!name.contains(char::is_whitespace), "import library name cannot contain whitespace");

    format!("{name}.dll.lib")
}

/// Construct the dynamic library extension based on the target.
#[must_use]
pub fn dynamic_lib_extension() -> &'static str {
    std::env::consts::DLL_EXTENSION
}

/// Construct the name of a rust library (rlib).
#[must_use]
pub fn rust_lib_name(name: &str) -> String {
    format!("lib{name}.rlib")
}

/// Construct the binary (executable) name based on the target.
#[must_use]
pub fn bin_name(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}
