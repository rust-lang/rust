// The attentive may note the underscores in the target triple, making it invalid. This test
// checks that such invalid target specs are rejected by the compiler.
// See https://github.com/rust-lang/rust/issues/33329

//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64_unknown-linux-musl

fn main() {}
