// Verifies that stable and unstable supported sanitizers cannot be used without
// `-Zunstable-options`.
//
//@ needs-llvm-components: x86
//@ needs-sanitizer-support
//@ ignore-backends: gcc
//@ compile-flags: -Clto -Csanitize=address,cfi --target x86_64-unknown-linux-gnu
//@ error-pattern: error: cfi sanitizer is not supported for this target

fn main() { }
