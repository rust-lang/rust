// run-pass
// ignore-arm
// ignore-aarch64
// ignore-mips
// ignore-mips64
// ignore-sparc
// ignore-sparc64
// ignore-wasm
// ignore-emscripten no processes
// ignore-sgx no processes
// ignore-musl FIXME #31506
// ignore-pretty
// ignore-fuchsia no exception handler registered for segfault
// compile-flags: -C lto
// no-prefer-dynamic
// ignore-nto Crash analysis impossible at SIGSEGV in QNX Neutrino

include!("stack-probes.rs");
