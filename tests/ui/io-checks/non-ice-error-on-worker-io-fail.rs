// Issue #66530: We would ICE if someone compiled with `-o /dev/null`,
// because we would try to generate auxiliary files in `/dev/` (which
// at least the OS X file system rejects).
//
// An attempt to `-o` into a directory we cannot write into should indeed
// be an error; but not an ICE.
//
// However, some folks run tests as root, which can write `/dev/` and end
// up clobbering `/dev/null`. Instead we'll use a non-existent path, which
// also used to ICE, but even root can't magically write there.

//@ compile-flags: -o ./does-not-exist/output

// The error-pattern check occurs *before* normalization, and the error patterns
// are wildly different between build environments. So this is a cop-out (and we
// rely on the checking of the normalized stderr output as our actual
// "verification" of the diagnostic).

// On Mac OS X, we get an error like the below
//@ normalize-stderr: "failed to write bytecode to ./does-not-exist/output.non_ice_error_on_worker_io_fail.*" -> "io error modifying ./does-not-exist/"

// On Linux, we get an error like the below
//@ normalize-stderr: "couldn't create a temp dir.*" -> "io error modifying ./does-not-exist/"

//@ ignore-windows - this is a unix-specific test
//@ ignore-emscripten - the file-system issues do not replicate here
//@ ignore-arm - the file-system issues do not replicate here, at least on armhf-gnu

#![crate_type = "lib"]

//~? ERROR /does-not-exist/
