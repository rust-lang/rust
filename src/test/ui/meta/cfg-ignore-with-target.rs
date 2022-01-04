// This test checks that the "ignore-cfg" compiletest option is evaluated
// based on the "--target" option in "compile-flags", rather than
// the default platform.
//
// compile-flags: --target x86_64-unknown-linux-gnu
// needs-llvm-components: x86
// ignore-cfg: target_os=linux
// check-pass

compile_error!("this should never be compiled");

fn main() {}
