//@ compile-flags: --print native-static-libs --crate-type staticlib  --emit metadata
//@ check-pass
//~? WARN skipping link step due to conflict: cannot output linkage information without emitting link
