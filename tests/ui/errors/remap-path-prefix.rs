//@ revisions: normal with-diagnostic-scope without-diagnostic-scope
//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//@ [with-diagnostic-scope]compile-flags: -Zremap-path-scope=diagnostics
//@ [without-diagnostic-scope]compile-flags: -Zremap-path-scope=object
// Manually remap, so the remapped path remains in .stderr file.

// The remapped paths are not normalized by compiletest.
//@ normalize-stderr: "\\(errors)" -> "/$1"

fn main() {
    // We cannot actually put an ERROR marker here because
    // the file name in the error message is not what the
    // test framework expects (since the filename gets remapped).
    // We still test the expected error in the stderr file.
    ferris //[without-diagnostic-scope]~ ERROR cannot find value `ferris` in this scope
}

//[normal]~? ERROR cannot find value `ferris` in this scope
//[with-diagnostic-scope]~? ERROR cannot find value `ferris` in this scope
