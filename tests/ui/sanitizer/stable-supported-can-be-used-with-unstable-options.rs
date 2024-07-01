// Verifies that stable supported sanitizers can be used with `-Zunstable-options`.
//
//@ needs-llvm-components: x86
//@ needs-sanitizer-support
//@ build-pass
//@ compile-flags: -Zunstable-options -Csanitizer=address --target x86_64-unknown-linux-gnu

fn main() { }
