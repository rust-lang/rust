// This test verifies that we do not produce a warning when performing LTO on a
// crate graph that contains a mix of different DWARF version settings. This
// matches Clang's behavior.

//@ ignore-msvc Platform must use DWARF
//@ aux-build:dwarf-mixed-versions-lto-aux.rs
//@ compile-flags: -C lto -g -Cdwarf-version=5
//@ no-prefer-dynamic
//@ build-pass
//@ ignore-backends: gcc

extern crate dwarf_mixed_versions_lto_aux;

fn main() {
    dwarf_mixed_versions_lto_aux::say_hi();
}
