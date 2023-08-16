//@run
//@ignore-target-arm
//@ignore-target-aarch64
//@ignore-target-mips
//@ignore-target-mips64
//@ignore-target-sparc
//@ignore-target-sparc64
//@ignore-target-loongarch64
//@ignore-target-wasm
//@ignore-target-emscripten no processes
//@ignore-target-sgx no processes
//@ignore-target-musl FIXME #31506
//@ignore-target-fuchsia no exception handler registered for segfault
//@compile-flags: -C lto
// no-prefer-dynamic
//@ignore-target-nto Crash analysis impossible at SIGSEGV in QNX Neutrino

include!("stack-probes.rs");
