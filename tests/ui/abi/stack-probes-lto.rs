//@ revisions: aarch64 x32 x64
//@ run-pass
//@[aarch64] only-aarch64
//@[x32] only-x86
//@[x64] only-x86_64
//@ needs-subprocess
//@ ignore-musl FIXME #31506
//@ ignore-fuchsia no exception handler registered for segfault
//@ compile-flags: -C lto
//@ no-prefer-dynamic
//@ ignore-nto Crash analysis impossible at SIGSEGV in QNX Neutrino
//@ ignore-ios Stack probes are enabled, but the SIGSEGV handler isn't
//@ ignore-tvos Stack probes are enabled, but the SIGSEGV handler isn't
//@ ignore-watchos Stack probes are enabled, but the SIGSEGV handler isn't
//@ ignore-visionos Stack probes are enabled, but the SIGSEGV handler isn't
//@ ignore-backends: gcc

include!("stack-probes.rs");
