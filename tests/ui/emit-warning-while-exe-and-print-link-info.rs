//@ compile-flags: --print native-static-libs --crate-type staticlib  --emit metadata
//@ check-pass
//@ error-pattern: warning: skipping link step due to conflict: cannot output linkage information without emitting executable
//@ error-pattern: note: consider emitting executable to print linkage information
