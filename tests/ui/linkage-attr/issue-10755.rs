// build-fail
// dont-check-compiler-stderr
// compile-flags: -C linker=llllll
// error-pattern: `llllll`

// Before, the error-pattern checked for "not found". On WSL with appendWindowsPath=true, running
// in invalid command returns a PermissionDenied instead.

fn main() {
}
