//@ needs-llvm-components: webassembly
//@ compile-flags: --print=target-cpus --target=wasm32-unknown-unknown
//@ check-pass

// LLVM at HEAD has added support for the `lime1` CPU. Remove it from the
// output so that the stdout with LLVM-at-HEAD matches the output of the LLVM
// versions currently used by default.
// FIXME(#133919): Once Rust upgrades to LLVM 20, remove this.
//@ normalize-stdout: "(?m)^ *lime1\n" -> ""
