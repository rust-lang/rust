// FIXME: find a way to add rustflags to ui-cargo tests
//@compile-flags: --remap-path-prefix {{src-base}}=/remapped

#![warn(clippy::self_named_module_files)]

mod bad;

fn main() {}
