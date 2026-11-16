//@ needs-sanitizer-support
//@ needs-sanitizer-address
//@ ignore-cross-compile
//@ ignore-backends: gcc
//
//@ build-pass
//@ compile-flags: -Z sanitizer=address -O -C codegen-units=1 --crate-type=rlib
//
// Make sure there is no redundant instrumentation warning when going through the no-LTO
// optimization pipeline.

pub fn test() {
}
