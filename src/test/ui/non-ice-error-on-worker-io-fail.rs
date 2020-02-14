// Issue #66530: We would ICE if someone compiled with `-o /dev/null`,
// because we would try to generate auxiliary files in `/dev/` (which
// at least the OS X file system rejects).
//
// An attempt to `-o` into a directory we cannot write into should indeed
// be an error; but not an ICE.

// compile-flags: -o /dev/null

// The error-pattern check occurs *before* normalization, and the error patterns
// are wildly different between build environments. So this is a cop-out (and we
// rely on the checking of the normalized stderr output as our actual
// "verification" of the diagnostic).

// error-pattern: error

// On Mac OS X, we get an error like the below
// normalize-stderr-test "failed to write bytecode to /dev/null.non_ice_error_on_worker_io_fail.*" -> "io error modifying /dev/"

// On Linux, we get an error like the below
// normalize-stderr-test "couldn't create a temp dir.*" -> "io error modifying /dev/"

// ignore-tidy-linelength
// ignore-windows - this is a unix-specific test
// ignore-emscripten - the file-system issues do not replicate here
// ignore-wasm - the file-system issues do not replicate here
// ignore-arm - the file-system issues do not replicate here, at least on armhf-gnu

#![crate_type="lib"]

#![cfg_attr(not(feature = "std"), no_std)]
pub mod task {
    pub mod __internal {
        use crate::task::Waker;
    }
    pub use core::task::Waker;
}
