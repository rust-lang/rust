//! A collection of helpers to construct artifact names, such as names of dynamic or static
//! librarys which are target-dependent.

use crate::targets::{is_darwin, is_msvc, is_windows};

/// Construct the static library name based on the target.
#[must_use]
pub fn static_lib_name(name: &str) -> String {
    // See tools.mk (irrelevant lines omitted):
    //
    // ```makefile
    // ifeq ($(UNAME),Darwin)
    //     STATICLIB = $(TMPDIR)/lib$(1).a
    // else
    //     ifdef IS_WINDOWS
    //         ifdef IS_MSVC
    //             STATICLIB = $(TMPDIR)/$(1).lib
    //         else
    //             STATICLIB = $(TMPDIR)/lib$(1).a
    //         endif
    //     else
    //         STATICLIB = $(TMPDIR)/lib$(1).a
    //     endif
    // endif
    // ```
    assert!(!name.contains(char::is_whitespace), "static library name cannot contain whitespace");

    if is_msvc() { format!("{name}.lib") } else { format!("lib{name}.a") }
}

/// Construct the dynamic library name based on the target.
#[must_use]
pub fn dynamic_lib_name(name: &str) -> String {
    // See tools.mk (irrelevant lines omitted):
    //
    // ```makefile
    // ifeq ($(UNAME),Darwin)
    //     DYLIB = $(TMPDIR)/lib$(1).dylib
    // else
    //     ifdef IS_WINDOWS
    //         DYLIB = $(TMPDIR)/$(1).dll
    //     else
    //         DYLIB = $(TMPDIR)/lib$(1).so
    //     endif
    // endif
    // ```
    assert!(!name.contains(char::is_whitespace), "dynamic library name cannot contain whitespace");

    let extension = dynamic_lib_extension();
    if is_darwin() {
        format!("lib{name}.{extension}")
    } else if is_windows() {
        format!("{name}.{extension}")
    } else {
        format!("lib{name}.{extension}")
    }
}

/// Construct the dynamic library extension based on the target.
#[must_use]
pub fn dynamic_lib_extension() -> &'static str {
    if is_darwin() {
        "dylib"
    } else if is_windows() {
        "dll"
    } else {
        "so"
    }
}

/// Construct the name of a rust library (rlib).
#[must_use]
pub fn rust_lib_name(name: &str) -> String {
    format!("lib{name}.rlib")
}

/// Construct the binary (executable) name based on the target.
#[must_use]
pub fn bin_name(name: &str) -> String {
    if is_windows() { format!("{name}.exe") } else { name.to_string() }
}
