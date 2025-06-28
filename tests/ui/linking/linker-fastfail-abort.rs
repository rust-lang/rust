//@ build-fail
//@ only-windows The abnormal behaviour may be observed only on Windows using the MSVC linker
//@ only-msvc

//@ ignore-cross-compile because aux-bin does not yet support it
//@ aux-bin: link.rs
// Dummy linker that will always exit as if like with __fastfail abort()

//@ compile-flags: -Clinker={{build-base}}\linking\linker-fastfail-abort\auxiliary\bin\link.exe -Ctarget-feature=+crt-static

// Since the error notes are too verbose
//@ dont-check-compiler-stderr
//@ dont-require-annotations: NOTE

//~? ERROR linking with `$TEST_BUILD_DIR/auxiliary/bin/link.exe` failed: exit code: 0xc0000409
//~? NOTE 0xc0000409 is `STATUS_STACK_BUFFER_OVERRUN`
//~? NOTE This may occur when using the MSVC toolchain on Windows and can be caused by `__fastfail` termination, rather than necessarily indicating a stack buffer overrun.

fn main() {}
