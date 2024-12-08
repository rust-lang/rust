// Verifies that `-Z instrument-xray` cannot be used with unsupported targets,
//
//@ needs-llvm-components: x86
//@ compile-flags: -Z instrument-xray --target x86_64-apple-darwin
//@ error-pattern: error: XRay instrumentation is not supported for this target

#![feature(no_core)]
#![no_core]
#![no_main]
