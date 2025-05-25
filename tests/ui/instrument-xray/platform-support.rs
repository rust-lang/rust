//@ only-nightly (flag is still unstable)
//@ needs-xray

//@ revisions: unsupported
//@[unsupported] needs-llvm-components: x86
//@[unsupported] compile-flags: -Z instrument-xray --target=x86_64-pc-windows-msvc

//@ revisions: x86_64-linux
//@[x86_64-linux] needs-llvm-components: x86
//@[x86_64-linux] compile-flags: -Z instrument-xray --target=x86_64-unknown-linux-gnu
//@[x86_64-linux] check-pass

//@ revisions: x86_64-darwin
//@[x86_64-darwin] needs-llvm-components: x86
//@[x86_64-darwin] compile-flags: -Z instrument-xray --target=x86_64-apple-darwin
//@[x86_64-darwin] check-pass

//@ revisions: aarch64-darwin
//@[aarch64-darwin] needs-llvm-components: aarch64
//@[aarch64-darwin] compile-flags: -Z instrument-xray --target=aarch64-apple-darwin
//@[aarch64-darwin] check-pass

#![feature(no_core)]
#![no_core]
#![no_main]

//[unsupported]~? ERROR XRay instrumentation is not supported for this target
